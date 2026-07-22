// Package vgpuresolver abstracts the vGPU-scheduler-specific parts of GPU
// checkpoint/restore behind a small interface, so internal/lib/restore.go
// and internal/lib/checkpoint.go never hardcode a scheduler's env var names,
// label keys, or claim-path conventions. HAMi is the only implementation
// today; a future NVIDIA DRA or MIG backend registers its own Resolver
// without any change to the restore/checkpoint call sites.
package vgpuresolver

import (
	"context"

	rspec "github.com/opencontainers/runtime-spec/specs-go"
)

// GPUCapability is the portable "GPU Capability Descriptor" — what a
// checkpoint needs to record about the GPU slice it ran against, and what a
// candidate restore target must be compared against. It intentionally
// carries no scheduler-specific fields (no HAMi claim paths, no DRA
// ResourceClaim names) — those stay inside each Resolver implementation.
type GPUCapability struct {
	// VRAMMiB is the vGPU memory allocation granted to the container
	// (nvidia.com/gpumem-equivalent), in MiB.
	VRAMMiB int `json:"vram_mb"`
	// ComputeCores is the vGPU compute-share allocation, on the resolver's
	// own normalized scale (HAMi: 0-100; nvidia.com/gpucores-equivalent).
	ComputeCores int `json:"compute_cores"`
	// GPUCount is how many physical GPUs this allocation spans. Always 1 in
	// the current HAMi resolver — HAMi does not split a single container
	// across multiple physical GPUs.
	GPUCount int `json:"gpu_count"`
	// ComputeCapability is the CUDA compute capability of the physical GPU
	// backing this allocation (e.g. "8.6"), live-queried, never hardcoded.
	ComputeCapability string `json:"compute_capability,omitempty"`
	// DriverVersion is the host NVIDIA driver version, live-queried.
	DriverVersion string `json:"driver_version,omitempty"`
	// HighWaterMarkMiB is the actual peak VRAM resident in the GPU context at
	// the moment this descriptor was captured (checkpoint time) or, for a
	// restore target, the currently free VRAM available to it. Zero means
	// "not measured" (e.g. a resolve failure), never a placeholder value —
	// callers must treat zero as "unknown," not "no usage."
	HighWaterMarkMiB int `json:"high_water_mark_mb,omitempty"`
	// GPUUUID is the physical GPU's UUID at capture time, live-queried.
	GPUUUID string `json:"gpu_uuid,omitempty"`
}

// Resolver is implemented once per vGPU scheduler backend.
type Resolver interface {
	// Name identifies the resolver for logging (e.g. "hami").
	Name() string

	// IsManagedPod reports whether this resolver owns pods carrying the
	// given sandbox labels.
	IsManagedPod(labels map[string]string) bool

	// ClaimMountSources returns the subset of mount sources in spec that
	// this resolver recognizes as its own per-container vGPU claim
	// bind-mounts (e.g. HAMi's UUID-named directories under
	// /usr/local/vgpu/claims/).
	ClaimMountSources(mounts []rspec.Mount) []string

	// MatchClaimSources pairs each entry in oldSources (from the checkpoint)
	// with an entry in newSources (from the freshly generated restore spec),
	// returning an old->new map. A source with no match is omitted from the
	// map, not defaulted to an arbitrary pairing — callers must treat a
	// missing key as "this claim could not be remapped."
	MatchClaimSources(oldSources, newSources []string) map[string]string

	// ResolveAllocation reads the live GPU allocation a container process
	// actually holds, from its environment and a live query against the
	// bound device. ok is false when env carries no allocation for this
	// resolver (e.g. a CPU-only container) — that is not an error, callers
	// should skip GPU-capability handling entirely in that case.
	ResolveAllocation(ctx context.Context, env []string) (cap *GPUCapability, ok bool, err error)
}

var registry []Resolver

// Register adds a Resolver implementation to the global registry. Called
// from each backend's init().
func Register(r Resolver) {
	registry = append(registry, r)
}

// For returns the first registered Resolver that claims the given sandbox
// labels, or nil if none does (the pod is not managed by any known vGPU
// scheduler — checkpoint/restore proceeds with no GPU-capability handling).
func For(labels map[string]string) Resolver {
	for _, r := range registry {
		if r.IsManagedPod(labels) {
			return r
		}
	}
	return nil
}
