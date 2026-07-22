package lib

import (
	"archive/tar"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"

	metadata "github.com/checkpoint-restore/checkpointctl/lib"
	"github.com/opencontainers/runtime-tools/generate"

	"github.com/cri-o/cri-o/internal/lib/sandbox"
	"github.com/cri-o/cri-o/internal/lib/vgpuresolver"
	"github.com/cri-o/cri-o/internal/log"
	"github.com/cri-o/cri-o/internal/oci"
)

const (
	// gpuCapabilityFile is the checkpoint-archive sidecar recording the GPU
	// Capability Descriptor of the container's vGPU allocation at dump time.
	// Kept out of CRIU's own binary image format deliberately — see
	// docs/requirements/checkpoint-restore-gpu-decoupling-design.md §3.1.
	gpuCapabilityFile = "gpu-capability.json"

	// gpuRemapFile records the old->new GPU identity/size translation
	// computed at restore time, for auditing and for the post-restore
	// action script to cross-check against the device it actually bound.
	gpuRemapFile = "gpu-remap.json"

	// gpuCapabilityConfFile is a plain key=value sidecar (not JSON) written
	// alongside gpu-remap.json at restore time, read by CRIU's CUDA plugin
	// guard hook (cuda_plugin_verify_gpu_capacity, plugins/cuda/cuda_plugin.c)
	// — CRIU's C tree intentionally has no JSON dependency.
	gpuCapabilityConfFile = "gpu-capability.conf"
)

// gpuRemapMetadata is the on-disk shape of gpuRemapFile.
type gpuRemapMetadata struct {
	OldUUID    string `json:"old_uuid"`
	NewUUID    string `json:"new_uuid"`
	OldVRAMMiB int    `json:"old_vram_mb"`
	NewVRAMMiB int    `json:"new_vram_mb"`
	OldCores   int    `json:"old_cores"`
	NewCores   int    `json:"new_cores"`
	Resolver   string `json:"resolver"`
}

// gpuCapabilityCompatibilityError is a distinct error type so callers
// (server/container_restore.go) can surface a clear, actionable message
// instead of letting the failure resurface deep inside CRIU/cuda-checkpoint.
type gpuCapabilityCompatibilityError struct {
	reason string
}

func (e *gpuCapabilityCompatibilityError) Error() string {
	return "GPU capability incompatible with restore target: " + e.reason
}

// writeGPUCapabilityMetadata captures the container's live vGPU allocation
// (via the sandbox's vGPU resolver — HAMi today) and writes it to
// gpuCapabilityFile in ctr.Dir(), returning its path so the caller can add
// it to the exported tar's file list. Returns ("", nil) — not an error —
// when the pod isn't managed by any registered resolver, or the resolver
// finds no allocation env vars (a CPU-only container). A live-query failure
// (e.g. nvidia-smi unavailable) is logged and also treated as "nothing to
// write," since checkpoint metadata capture must never block the checkpoint
// itself — a checkpoint predating this capability is handled today (as a
// "legacy" checkpoint) by the restore-time decision layer.
func (c *ContainerServer) writeGPUCapabilityMetadata(ctx context.Context, ctr *oci.Container, specgen *generate.Generator) (string, error) {
	sb := c.GetSandbox(ctr.Sandbox())
	if sb == nil {
		return "", nil
	}

	resolver := vgpuresolver.For(sb.Labels())
	if resolver == nil {
		return "", nil
	}

	if specgen.Config.Process == nil {
		return "", nil
	}

	cap, ok, err := resolver.ResolveAllocation(ctx, specgen.Config.Process.Env)
	if err != nil {
		log.Warnf(ctx, "GPU capability capture failed for container %s (resolver=%s): %v — checkpoint will proceed without GPU sizing metadata", ctr.ID(), resolver.Name(), err)
		return "", nil
	}
	if !ok {
		return "", nil
	}

	path, err := metadata.WriteJSONFile(cap, ctr.Dir(), gpuCapabilityFile)
	if err != nil {
		return "", fmt.Errorf("writing %s: %w", gpuCapabilityFile, err)
	}

	log.Infof(ctx, "Captured GPU capability for container %s: vram_mb=%d cores=%d uuid=%s high_water_mark_mb=%d",
		ctr.ID(), cap.VRAMMiB, cap.ComputeCores, cap.GPUUUID, cap.HighWaterMarkMiB)

	return path, nil
}

// readGPUCapabilityFromImage reads gpuCapabilityFile from an already-mounted
// checkpoint OCI image. Returns (nil, nil) — not an error — when the file is
// absent, since that means the checkpoint predates GCD capture or never had
// a GPU allocation; callers must treat that as "legacy, no compatibility
// check possible" rather than a failure.
func readGPUCapabilityFromImage(ctx context.Context, imageMountPoint string) (*vgpuresolver.GPUCapability, error) {
	path := filepath.Join(imageMountPoint, gpuCapabilityFile)
	if _, err := os.Stat(path); err != nil {
		return nil, nil
	}

	var cap vgpuresolver.GPUCapability
	if _, err := metadata.ReadJSONFile(&cap, imageMountPoint, gpuCapabilityFile); err != nil {
		log.Warnf(ctx, "Failed to read %s for GPU capability check: %v", gpuCapabilityFile, err)
		return nil, nil
	}
	return &cap, nil
}

// readGPUCapabilityFromArchive extracts gpuCapabilityFile from a checkpoint
// tar archive without disturbing the main restore extraction. Same
// (nil, nil)-on-absent contract as readGPUCapabilityFromImage.
func readGPUCapabilityFromArchive(ctx context.Context, archivePath string) (*vgpuresolver.GPUCapability, error) {
	raw, err := extractSingleFileFromTar(archivePath, gpuCapabilityFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		log.Warnf(ctx, "Failed to extract %s from checkpoint archive %s: %v", gpuCapabilityFile, archivePath, err)
		return nil, nil
	}
	if raw == nil {
		return nil, nil
	}

	var cap vgpuresolver.GPUCapability
	if err := json.Unmarshal(raw, &cap); err != nil {
		log.Warnf(ctx, "Failed to parse %s from checkpoint archive %s: %v", gpuCapabilityFile, archivePath, err)
		return nil, nil
	}
	return &cap, nil
}

// extractSingleFileFromTar reads exactly one named file (matched by base
// name) out of an uncompressed tar archive, without extracting the rest to
// disk. Returns (nil, os.ErrNotExist)-compatible error when the archive
// contains no matching entry.
func extractSingleFileFromTar(tarPath, targetName string) ([]byte, error) {
	f, err := os.Open(tarPath)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	tr := tar.NewReader(f)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			return nil, os.ErrNotExist
		}
		if err != nil {
			return nil, err
		}
		if hdr.Typeflag != tar.TypeReg {
			continue
		}
		if filepath.Base(hdr.Name) != targetName {
			continue
		}
		return io.ReadAll(tr)
	}
}

// resolveNewGPUAllocation builds the GPU Capability Descriptor for the
// restore target — the pod/container that was just created by Kubernetes —
// from its own live CDI/env allocation, exactly as it would be resolved at
// checkpoint time. Returns (nil, nil, false, nil) when the pod isn't managed
// by any registered resolver.
func resolveNewGPUAllocation(ctx context.Context, sb *sandbox.Sandbox, ctrSpec *generate.Generator) (*vgpuresolver.GPUCapability, vgpuresolver.Resolver, bool, error) {
	if sb == nil || ctrSpec == nil || ctrSpec.Config == nil || ctrSpec.Config.Process == nil {
		return nil, nil, false, nil
	}

	resolver := vgpuresolver.For(sb.Labels())
	if resolver == nil {
		return nil, nil, false, nil
	}

	cap, ok, err := resolver.ResolveAllocation(ctx, ctrSpec.Config.Process.Env)
	if err != nil {
		return nil, resolver, false, err
	}
	if !ok {
		return nil, resolver, false, nil
	}
	return cap, resolver, true, nil
}

// validateGPUCapabilityCompatibility is the fail-fast pre-flight check run
// before CRIU/runc is ever invoked. It compares the checkpoint's recorded
// GCD against the new allocation's GCD and returns a
// gpuCapabilityCompatibilityError with an actionable reason on any
// incompatibility, instead of letting a doomed restore proceed into a deep,
// cryptic CRIU/cuda-checkpoint failure.
//
// Either argument may be nil: a nil old means the checkpoint predates GCD
// capture (nothing to validate against — proceed); a nil new means this
// pod holds no GPU allocation at all under the resolver that owned the
// checkpoint, which is itself an incompatibility only when old is non-nil
// (a GPU checkpoint restoring onto a non-GPU allocation can never work).
func validateGPUCapabilityCompatibility(old, new *vgpuresolver.GPUCapability) error {
	if old == nil {
		return nil
	}
	if new == nil {
		return &gpuCapabilityCompatibilityError{
			reason: fmt.Sprintf("checkpoint requires a GPU allocation (%d MiB high-water-mark) but the restore target has none", old.HighWaterMarkMiB),
		}
	}
	if old.HighWaterMarkMiB > 0 && new.VRAMMiB > 0 && new.VRAMMiB < old.HighWaterMarkMiB {
		return &gpuCapabilityCompatibilityError{
			reason: fmt.Sprintf("restore target's GPU allocation (%d MiB) is smaller than the checkpoint's peak VRAM usage (%d MiB)", new.VRAMMiB, old.HighWaterMarkMiB),
		}
	}
	if old.ComputeCapability != "" && new.ComputeCapability != "" && old.ComputeCapability != new.ComputeCapability {
		return &gpuCapabilityCompatibilityError{
			reason: fmt.Sprintf("checkpoint was taken on a GPU with compute capability %s, restore target has %s — CUDA context restore across incompatible architectures is not supported by cuda-checkpoint", old.ComputeCapability, new.ComputeCapability),
		}
	}
	return nil
}

// writeGPURemapFiles writes gpuRemapFile (JSON, for CRI-O's own audit trail
// and the shell action-script) and gpuCapabilityConfFile (plain key=value,
// for CRIU's C plugin) into checkpointDir. old may be nil (legacy
// checkpoint, nothing to remap); when it is, both files are skipped and
// ("", "", nil) is returned — there is nothing for the CRIU-side guard to
// check, and it must not fail closed on data that was never captured.
func writeGPURemapFiles(checkpointDir string, old, new *vgpuresolver.GPUCapability, resolverName string) (remapPath, confPath string, err error) {
	if old == nil || new == nil {
		return "", "", nil
	}

	remap := gpuRemapMetadata{
		OldUUID:    old.GPUUUID,
		NewUUID:    new.GPUUUID,
		OldVRAMMiB: old.VRAMMiB,
		NewVRAMMiB: new.VRAMMiB,
		OldCores:   old.ComputeCores,
		NewCores:   new.ComputeCores,
		Resolver:   resolverName,
	}

	remapJSON, err := json.MarshalIndent(remap, "", "  ")
	if err != nil {
		return "", "", fmt.Errorf("marshal %s: %w", gpuRemapFile, err)
	}
	remapPath = filepath.Join(checkpointDir, gpuRemapFile)
	if err := os.WriteFile(remapPath, remapJSON, 0o600); err != nil {
		return "", "", fmt.Errorf("write %s: %w", remapPath, err)
	}

	confContent := fmt.Sprintf(
		"old_uuid=%s\nnew_uuid=%s\nold_vram_mb=%d\nnew_vram_mb=%d\nhigh_water_mark_mb=%d\n",
		old.GPUUUID, new.GPUUUID, old.VRAMMiB, new.VRAMMiB, old.HighWaterMarkMiB,
	)
	confPath = filepath.Join(checkpointDir, gpuCapabilityConfFile)
	if err := os.WriteFile(confPath, []byte(confContent), 0o600); err != nil {
		os.Remove(remapPath)
		return "", "", fmt.Errorf("write %s: %w", confPath, err)
	}

	return remapPath, confPath, nil
}

// cleanupGPURemapFiles removes the sidecar files written by
// writeGPURemapFiles. Safe to call with empty paths (no-op).
func cleanupGPURemapFiles(ctx context.Context, remapPath, confPath string) {
	for _, p := range []string{remapPath, confPath} {
		if p == "" {
			continue
		}
		if err := os.Remove(p); err != nil && !os.IsNotExist(err) {
			log.Warnf(ctx, "Failed to clean up GPU remap file %s: %v", p, err)
		}
	}
}
