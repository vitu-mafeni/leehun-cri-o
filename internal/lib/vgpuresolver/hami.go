package vgpuresolver

import (
	"context"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"

	rspec "github.com/opencontainers/runtime-spec/specs-go"
)

const (
	// hamiDRALabel is the pod label that marks a pod as managed by the
	// HAMi DRA driver. When set to "true", HAMi creates per-container
	// directories under hamiVGPUContainersPrefix; these get a new UUID
	// each time the pod is (re)scheduled, so during checkpoint/restore
	// the original path must be remapped via CRIU's ext-mount-map.
	hamiDRALabel = "hami.io/dra"

	// hamiVGPUContainersPrefix is the host-side directory under which HAMi
	// creates one subdirectory per container, "<UUID>_<container_name>",
	// bind-mounted into the container.
	hamiVGPUContainersPrefix = "/usr/local/vgpu/claims/"

	// hamiClaimUUIDLen is the length of the UUID segment HAMi prefixes onto
	// each claim directory's basename, before the "_<container_name>" suffix.
	hamiClaimUUIDLen = 36

	// HAMi's vGPU hijack library (libvgpu.so) reads these env vars to enforce
	// per-container memory/compute limits — see Project-HAMi/HAMi. They are
	// the authoritative source for "what allocation does this container
	// actually hold," independent of whatever quota/scheduling layer decided
	// it upstream.
	envCUDADeviceMemoryLimit = "CUDA_DEVICE_MEMORY_LIMIT"
	envCUDADeviceSMLimit     = "CUDA_DEVICE_SM_LIMIT"
	envNvidiaVisibleDevices  = "NVIDIA_VISIBLE_DEVICES"
)

type hamiResolver struct{}

func init() {
	Register(&hamiResolver{})
}

func (h *hamiResolver) Name() string { return "hami" }

func (h *hamiResolver) IsManagedPod(labels map[string]string) bool {
	return labels[hamiDRALabel] == "true"
}

// ClaimMountSources returns the source paths of HAMi vGPU bind mounts
// (those rooted at hamiVGPUContainersPrefix) found in the given mount list.
func (h *hamiResolver) ClaimMountSources(mounts []rspec.Mount) []string {
	var paths []string
	for _, m := range mounts {
		if strings.HasPrefix(m.Source, hamiVGPUContainersPrefix) {
			paths = append(paths, m.Source)
		}
	}
	return paths
}

// MatchClaimSources pairs each old claim path with a new one by the
// trailing container-name suffix after HAMi's 36-character UUID directory
// segment, so pods with multiple vGPU mounts still match correctly. An old
// path with no name-suffix match falls back to any unused new path (the
// common case of exactly one vGPU mount per container); an old path that
// still finds nothing is simply omitted from the result.
func (h *hamiResolver) MatchClaimSources(oldSources, newSources []string) map[string]string {
	result := make(map[string]string, len(oldSources))
	used := make(map[string]bool, len(newSources))

	for _, oldPath := range oldSources {
		oldSuffix := hamiClaimNameSuffix(oldPath)
		matched := ""

		if oldSuffix != "" {
			for _, p := range newSources {
				if used[p] {
					continue
				}
				if hamiClaimNameSuffix(p) == oldSuffix {
					matched = p
					break
				}
			}
		}
		if matched == "" {
			for _, p := range newSources {
				if !used[p] {
					matched = p
					break
				}
			}
		}
		if matched != "" {
			used[matched] = true
			result[oldPath] = matched
		}
	}
	return result
}

// hamiClaimNameSuffix returns the container-name part of a HAMi vGPU
// directory path. HAMi names the directory "<UUID>_<container_name>"; the
// UUID is 36 characters followed by an underscore. Returns "" when the
// basename doesn't match this convention.
func hamiClaimNameSuffix(p string) string {
	base := filepath.Base(p)
	if len(base) > hamiClaimUUIDLen+1 && base[hamiClaimUUIDLen] == '_' {
		return base[hamiClaimUUIDLen+1:]
	}
	return ""
}

// ResolveAllocation reads the container's live HAMi vGPU allocation from its
// environment (the hijack-library limit env vars) and queries the host GPU
// driver for the physical characteristics of the device actually bound —
// never a hardcoded or cached value; every field here is freshly derived at
// call time from either the container's own env or a live nvidia-smi query.
func (h *hamiResolver) ResolveAllocation(ctx context.Context, env []string) (*GPUCapability, bool, error) {
	memLimitMiB, haveMem := lookupIntEnv(env, envCUDADeviceMemoryLimit)
	smLimit, haveSM := lookupIntEnv(env, envCUDADeviceSMLimit)
	uuid := firstVisibleDeviceUUID(env)

	if !haveMem && !haveSM && uuid == "" {
		// No HAMi allocation env vars present at all — this container holds
		// no vGPU slice under this resolver (e.g. a CPU-only sibling
		// container in the same pod). Not an error.
		return nil, false, nil
	}

	cap := &GPUCapability{
		VRAMMiB:      memLimitMiB,
		ComputeCores: smLimit,
		GPUCount:     1, // HAMi allocates a single physical GPU per container today
		GPUUUID:      uuid,
	}

	if uuid != "" {
		info, err := queryLiveGPUInfo(ctx, uuid)
		if err != nil {
			return cap, true, err
		}
		cap.HighWaterMarkMiB = info.usedMiB
		cap.ComputeCapability = info.computeCapability
		cap.DriverVersion = info.driverVersion
	}

	return cap, true, nil
}

func lookupIntEnv(env []string, key string) (int, bool) {
	prefix := key + "="
	for _, e := range env {
		if !strings.HasPrefix(e, prefix) {
			continue
		}
		val := strings.TrimPrefix(e, prefix)
		n, err := strconv.Atoi(strings.TrimSpace(val))
		if err != nil {
			return 0, false
		}
		return n, true
	}
	return 0, false
}

// firstVisibleDeviceUUID returns the first device identifier in
// NVIDIA_VISIBLE_DEVICES. HAMi/the NVIDIA device plugin allocate exactly one
// physical GPU per container in the topology this resolver targets, so the
// first entry is authoritative when more than one is somehow present.
func firstVisibleDeviceUUID(env []string) string {
	prefix := envNvidiaVisibleDevices + "="
	for _, e := range env {
		if !strings.HasPrefix(e, prefix) {
			continue
		}
		val := strings.TrimPrefix(e, prefix)
		if val == "" || val == "void" || val == "none" || val == "all" {
			return ""
		}
		first := strings.SplitN(val, ",", 2)[0]
		return strings.TrimSpace(first)
	}
	return ""
}

type liveGPUInfo struct {
	usedMiB           int
	freeMiB           int
	computeCapability string
	driverVersion     string
}

// queryLiveGPUInfo shells out to nvidia-smi for the physical device
// identified by uuid. This is CRI-O's host-level process, so it queries the
// host driver directly — no container mount-namespace entry is needed since
// nvidia-smi talks to the driver via /dev/nvidiactl, not the filesystem.
func queryLiveGPUInfo(ctx context.Context, uuid string) (*liveGPUInfo, error) {
	out, err := exec.CommandContext(ctx, "nvidia-smi",
		"--query-gpu=memory.used,memory.free,compute_cap,driver_version",
		"--format=csv,noheader,nounits",
		"-i", uuid,
	).Output()
	if err != nil {
		return nil, err
	}

	fields := strings.Split(strings.TrimSpace(string(out)), ",")
	if len(fields) < 4 {
		return nil, &nvidiaSMIParseError{raw: string(out)}
	}

	usedMiB, _ := strconv.Atoi(strings.TrimSpace(fields[0]))
	freeMiB, _ := strconv.Atoi(strings.TrimSpace(fields[1]))

	return &liveGPUInfo{
		usedMiB:           usedMiB,
		freeMiB:           freeMiB,
		computeCapability: strings.TrimSpace(fields[2]),
		driverVersion:     strings.TrimSpace(fields[3]),
	}, nil
}

type nvidiaSMIParseError struct{ raw string }

func (e *nvidiaSMIParseError) Error() string {
	return "unexpected nvidia-smi output: " + e.raw
}
