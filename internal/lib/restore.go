package lib

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	metadata "github.com/checkpoint-restore/checkpointctl/lib"
	"github.com/checkpoint-restore/go-criu/v7/stats"
	rspec "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/opencontainers/runtime-tools/generate"
	"github.com/sirupsen/logrus"
	"go.podman.io/common/pkg/crutils"
	"go.podman.io/storage/pkg/archive"

	"github.com/cri-o/cri-o/internal/annotations"
	"github.com/cri-o/cri-o/internal/lib/vgpuresolver"
	"github.com/cri-o/cri-o/internal/log"
	"github.com/cri-o/cri-o/internal/oci"
	"tags.cncf.io/container-device-interface/pkg/cdi"
)

// criuDeviceRestorerScript is the path to the CRIU action script that
// remounts devices into the container after a CRIU restore.
// This is set at build time via -ldflags to match the install PREFIX.
// Defaults to /usr/libexec/crio/criu-device-restorer.sh for production builds.
var criuDeviceRestorerScript = "/usr/libexec/crio/criu-device-restorer.sh"

const (

	// deviceMappingFile is the name of the JSON file written into the CRIU
	// image directory containing host-to-container device mappings.
	deviceMappingFile = "device-mapping.json"

	// criuConfigFile is the name of the CRIU configuration file written
	// alongside the checkpoint that tells CRIU to invoke the action script.
	criuConfigFile = "criu-restore.conf"
)

// vGPU claim-path remapping (the HAMi UUID-directory problem) and GPU
// Capability Descriptor tracking are handled generically here via
// internal/lib/vgpuresolver — see that package and gpu_capability.go for the
// scheduler-specific logic (HAMi today). Nothing in this file hardcodes a
// resolver's label keys, env var names, or claim-path conventions.

// ContainerRestore restores a checkpointed container.
func (c *ContainerServer) ContainerRestore(
	ctx context.Context,
	config *metadata.ContainerConfig,
	opts *ContainerCheckpointOptions,
) (string, error) {
	var ctr *oci.Container

	var err error

	ctr, err = c.LookupContainer(ctx, config.ID)
	if err != nil {
		return "", fmt.Errorf("failed to find container %s: %w", config.ID, err)
	}

	cStatus := ctr.State()
	if cStatus.Status == oci.ContainerStateRunning {
		return "", fmt.Errorf("cannot restore running container %s", ctr.ID())
	}

	// Get config.json
	// This file is generated twice by earlier code. Once in BundlePath() and
	// once in Dir(). This code takes the version from Dir(), modifies it and
	// overwrites both versions (Dir() and BundlePath())
	ctrSpec, err := generate.NewFromFile(filepath.Join(ctr.Dir(), "config.json"))
	if err != nil {
		return "", err
	}
	// During checkpointing the container is unmounted. This mounts the container again.
	mountPoint, err := c.StorageImageServer().GetStore().Mount(ctr.ID(), ctrSpec.Config.Linux.MountLabel)
	if err != nil {
		log.Debugf(ctx, "Failed to mount container %q: %v", ctr.ID(), err)

		return "", err
	}

	log.Debugf(ctx, "Container mountpoint %v", mountPoint)
	log.Debugf(ctx, "Sandbox %v", ctr.Sandbox())
	log.Debugf(ctx, "Specgen.Config.Annotations[io.kubernetes.cri-o.SandboxID] %v", ctrSpec.Config.Annotations["io.kubernetes.cri-o.SandboxID"])

	sb, err := c.LookupSandbox(ctr.Sandbox())
	if err != nil {
		return "", err
	}

	// claimSources captures the original vGPU claim bind-mount source paths
	// recorded in the checkpoint (HAMi today, via vgpuresolver.Resolver).
	// They are needed to emit CRIU ext-mount-map entries because HAMi
	// assigns a new UUID-named directory to each freshly created pod, so
	// the old path is gone by the time we restore.
	var claimSources []string
	// oldGPUCapability is the checkpoint's recorded GPU Capability
	// Descriptor (nil for a checkpoint that predates GCD capture, or one
	// that never held a GPU allocation). newGPUCapability/gpuResolver are
	// resolved from the restore target's own live allocation once we know
	// there is checkpoint data to compare against — both stay nil when this
	// container is not being restored from an archive/image at all.
	var oldGPUCapability *vgpuresolver.GPUCapability
	var newGPUCapability *vgpuresolver.GPUCapability
	var gpuResolver vgpuresolver.Resolver

	if ctr.RestoreArchivePath() != "" || ctr.RestoreStorageImageID() != nil {
		if ctr.RestoreStorageImageID() != nil {
			log.Debugf(ctx, "Restoring from %v", ctr.RestoreStorageImageID())
			// This is not out-of-process, but it is at least out of the CRI-O codebase; containers/storage uses raw strings.
			imageMountPoint, err := c.StorageImageServer().GetStore().MountImage(ctr.RestoreStorageImageID().IDStringForOutOfProcessConsumptionOnly(), nil, "")
			if err != nil {
				return "", err
			}

			logrus.Debugf("Checkpoint image mounted at %v", imageMountPoint)

			defer func() {
				// This is not out-of-process, but it is at least out of the CRI-O codebase; containers/storage uses raw strings.
				_, err := c.StorageImageServer().GetStore().UnmountImage(ctr.RestoreStorageImageID().IDStringForOutOfProcessConsumptionOnly(), true)
				if err != nil {
					log.Errorf(ctx, "Failed to unmount checkpoint image: %q", err)
				}
			}()

			// Import all checkpoint files except ConfigDumpFile and SpecDumpFile. We
			// generate new container config files to enable to specifying a new
			// container name.
			checkpoint := []string{
				"artifacts",
				metadata.CheckpointDirectory,
				metadata.DevShmCheckpointTar,
				metadata.RootFsDiffTar,
				metadata.DeletedFilesFile,
				metadata.PodOptionsFile,
				metadata.PodDumpFile,
				stats.StatsDump,
				"bind.mounts",
				annotations.LogPath,
			}
			for _, name := range checkpoint {
				src := filepath.Join(imageMountPoint, name)
				dst := filepath.Join(ctr.Dir(), name)

				if err := archive.NewDefaultArchiver().CopyWithTar(src, dst); err != nil {
					logrus.Debugf("Can't import '%s' from checkpoint image", name)
				}
			}

			// Pick up the original vGPU claim mount sources and GPU
			// Capability Descriptor before we drop access to the
			// checkpoint image.
			if resolver := vgpuresolver.For(sb.Labels()); resolver != nil {
				var dumpSpec rspec.Spec
				if _, err := metadata.ReadJSONFile(&dumpSpec, imageMountPoint, metadata.SpecDumpFile); err != nil {
					log.Warnf(ctx, "Failed to read spec.dump for vGPU claim remap: %v", err)
				} else {
					claimSources = resolver.ClaimMountSources(dumpSpec.Mounts)
				}
			}
			oldGPUCapability, _ = readGPUCapabilityFromImage(ctx, imageMountPoint)
		} else {
			if err := crutils.CRImportCheckpointWithoutConfig(ctr.Dir(), ctr.RestoreArchivePath()); err != nil {
				return "", err
			}

			if resolver := vgpuresolver.For(sb.Labels()); resolver != nil {
				claimSources = readClaimSourcesFromArchive(ctx, ctr.RestoreArchivePath(), resolver)
			}
			oldGPUCapability, _ = readGPUCapabilityFromArchive(ctx, ctr.RestoreArchivePath())
		}

		// Fail fast, before CRIU/runc is ever invoked, if the restore
		// target's GPU allocation cannot host what the checkpoint actually
		// needs. A nil newGPUCapability with a non-nil oldGPUCapability is
		// itself a hard incompatibility (checked inside
		// validateGPUCapabilityCompatibility); a resolve error on the new
		// side is logged and treated as "could not verify," not a silent
		// pass — the pre-flight check below still runs against nil.
		var resolveErr error
		newGPUCapability, gpuResolver, _, resolveErr = resolveNewGPUAllocation(ctx, sb, &ctrSpec)
		if resolveErr != nil {
			log.Warnf(ctx, "Failed to resolve restore target's GPU allocation: %v", resolveErr)
		}
		if err := validateGPUCapabilityCompatibility(oldGPUCapability, newGPUCapability); err != nil {
			return "", err
		}

		if err := c.restoreFileSystemChanges(ctr, mountPoint); err != nil {
			return "", err
		}

		_, err = os.Stat(filepath.Join(ctr.Dir(), annotations.LogPath))
		if err == nil {
			src, err := os.Open(filepath.Join(ctr.Dir(), annotations.LogPath))
			if err != nil {
				return "", fmt.Errorf("error opening log file %q: %w", annotations.LogPath, err)
			}

			defer src.Close()

			destLogPath := ctrSpec.Config.Annotations[annotations.LogPath]

			destLog, err := os.Create(destLogPath)
			if err != nil {
				return "", fmt.Errorf("error opening log file %q: %w", destLogPath, err)
			}

			defer destLog.Close()

			_, err = io.Copy(destLog, src)
			if err != nil {
				return "", fmt.Errorf("copying log file to %q failed: %w", destLogPath, err)
			}
		}

		_, err = os.Stat(filepath.Join(ctr.Dir(), "bind.mounts"))
		if err == nil {
			// If the file does not exist we assume it is an older checkpoint archive
			// without this type of file and we just ignore it. Possible failures are
			// caught in the next block.
			var externalBindMounts []ExternalBindMount

			_, err := metadata.ReadJSONFile(&externalBindMounts, ctr.Dir(), "bind.mounts")
			if err != nil {
				return "", err
			}

			for _, e := range externalBindMounts {
				if func() bool {
					for _, m := range ctrSpec.Config.Mounts {
						if (m.Destination == e.Destination) && (m.Source != e.Source) {
							// If the source differs this means that the external mount
							// source has already been fixed up earlier by the restore
							// code and no need to deal with it here.
							// Good example is the /etc/resolv.conf bind mount is now
							// pointing to the new /etc/resolv.conf of the new pod.
							return true
						}
					}

					return false
				}() {
					continue
				}

				_, err = os.Lstat(e.Source)
				if err != nil {
					// Even if this looks suspicious it is was CRI-O does during
					// container create. For each missing bind mount source CRI-O
					// creates a directory. For restore that is problematic as
					// CRIU will fail to bind mount a directory on a file.
					// Therefore during restore CRI-O does not create a directory
					// for each missing bind mount source. We track external bind
					// mounts in the checkpoint archive and can now recreate missing
					// files or directories.
					// This is especially useful if restoring a Kubernetes container
					// outside of Kubernetes.
					if e.FileType == "directory" {
						if err := os.MkdirAll(e.Source, os.FileMode(e.Permissions)); err != nil {
							return "", fmt.Errorf(
								"failed to recreate directory %q for container %s: %w",
								e.Source,
								ctr.ID(),
								err,
							)
						}
					} else {
						if err := os.MkdirAll(filepath.Dir(e.Source), 0o700); err != nil {
							return "", err
						}

						source, err := os.OpenFile(
							e.Source,
							os.O_RDONLY|os.O_CREATE,
							os.FileMode(e.Permissions),
						)
						if err != nil {
							return "", fmt.Errorf(
								"failed to recreate file %q for container %s: %w",
								e.Source,
								ctr.ID(),
								err,
							)
						}

						source.Close()
					}

					log.Debugf(ctx, "Created missing external bind mount %q %q\n", e.FileType, e.Source)
				}
			}
		}

		for _, m := range ctrSpec.Config.Mounts {
			// This checks if all bind mount sources exist.
			// We cannot create missing bind mount sources automatically
			// as the source and destination need to be of the same type.
			// CRIU will fail restoring if the external bind mount source
			// is a directory but the internal destination is a file.
			// As destinations can be in nested bind mounts, which are only
			// correctly setup by runc/crun during container restore, we
			// cannot figure out the file type of the destination.
			// At this point we will fail and tell the user to create
			// the missing bind mount source file/directory.
			// With the code to create directories or files as necessary
			// this should not happen anymore. Still keeping the code
			// for backwards compatibility.
			if m.Type != bindMount {
				continue
			}

			_, err := os.Lstat(m.Source)
			if err != nil {
				return "", fmt.Errorf(
					"the bind mount source %s is missing. %s",
					m.Source,
					"Please create the corresponding file or directory",
				)
			}
		}
	}

	// We need to adapt the to be restored container to the sandbox created for this container.

	// The container will be restored in another sandbox. Adapt to
	// namespaces of the new sandbox
	for i, n := range ctrSpec.Config.Linux.Namespaces {
		if n.Path == "" {
			// The namespace in the original container did not point to
			// an existing interface. Leave it as it is.
			// CRIU will restore the namespace
			continue
		}

		for _, np := range sb.NamespacePaths() {
			if string(np.Type()) == string(n.Type) {
				ctrSpec.Config.Linux.Namespaces[i].Path = np.Path()

				break
			}
		}
	}

	// Update Sandbox Name
	ctrSpec.AddAnnotation(annotations.SandboxName, sb.Name())
	// Update Sandbox ID
	ctrSpec.AddAnnotation(annotations.SandboxID, ctr.Sandbox())

	mData := fmt.Sprintf(
		"k8s_%s_%s_%s_%s0",
		ctr.Name(),
		sb.KubeName(),
		sb.Namespace(),
		sb.Metadata().GetUid(),
	)
	ctrSpec.AddAnnotation(annotations.Name, mData)

	ctr.SetSandbox(ctr.Sandbox())

	// nvidia-container-runtime intercepts "runc create" but not "runc restore",
	// so GPU CDI state (devices, cgroup rules, hooks) must be injected here.
	//
	// ctrSpec.Config.Process.Env at this point already contains ONLY the new
	// CRI CreateContainer request's environment — server/checkpoint_utils.go's
	// buildContainerConfig() sets Envs: createConfig.GetEnvs() unconditionally
	// and never reads env from the checkpoint's dumpSpec. This is the
	// "new-request-wins" precedence the GPU-decoupling design requires for
	// HAMi's vGPU limit env vars and NVIDIA_VISIBLE_DEVICES (see
	// docs/requirements/checkpoint-restore-gpu-decoupling-design.md §3.4) —
	// verified here rather than re-implemented, since re-deriving it from
	// dumpSpec would reintroduce exactly the staleness bug that section warns
	// against. Do not add any code path here that copies Process.Env entries
	// from dumpSpec/hamiOldVGPUPaths/oldGPUCapability.
	injectCDIDevicesForRestore(ctx, &ctrSpec)

	// Generate device mapping metadata, CRIU config for the action script,
	// and (when the checkpoint carried a GPU Capability Descriptor) the
	// GPU remap sidecars consumed by the extended action script and by
	// CRIU's CUDA plugin guard hook. This must happen before saving
	// config.json so the org.criu.config annotation is included.
	deviceMappingPath, criuConfigPath, gpuRemapPath, gpuCapConfPath, err := writeDeviceRestoreMetadata(
		ctx, ctr, &ctrSpec, claimSources, gpuResolver, oldGPUCapability, newGPUCapability,
	)
	if err != nil {
		return "", fmt.Errorf("failed to write device restore metadata: %w", err)
	}

	saveOptions := generate.ExportOptions{}
	if err := ctrSpec.SaveToFile(filepath.Join(ctr.Dir(), "config.json"), saveOptions); err != nil {
		return "", err
	}

	if err := ctrSpec.SaveToFile(filepath.Join(ctr.BundlePath(), "config.json"), saveOptions); err != nil {
		return "", err
	}

	restoreErr := c.runtime.RestoreContainer(
		ctx,
		ctr,
		sb.CgroupParent(),
		sb.MountLabel(),
	)

	// Clean up device mapping, CRIU config, and GPU remap files regardless
	// of success/failure.
	cleanupDeviceRestoreMetadata(ctx, deviceMappingPath, criuConfigPath)
	cleanupGPURemapFiles(ctx, gpuRemapPath, gpuCapConfPath)

	if restoreErr != nil {
		return "", fmt.Errorf("failed to restore container %s: %w", ctr.ID(), restoreErr)
	}

	if err := c.ContainerStateToDisk(ctx, ctr); err != nil {
		log.Warnf(ctx, "Unable to write containers %s state to disk: %v", ctr.ID(), err)
	}

	if !opts.Keep {
		// Delete all checkpoint related files. At this point, in theory, all files
		// should exist. Still ignoring errors for now as the container should be
		// restored and running. Not erroring out just because some cleanup operation
		// failed. Starting with the checkpoint directory
		err = os.RemoveAll(ctr.CheckpointPath())
		if err != nil {
			log.Debugf(ctx, "Non-fatal: removal of checkpoint directory (%s) failed: %v", ctr.CheckpointPath(), err)
		}

		cleanup := [...]string{
			metadata.RestoreLogFile,
			metadata.DumpLogFile,
			stats.StatsDump,
			stats.StatsRestore,
			metadata.NetworkStatusFile,
			metadata.RootFsDiffTar,
			metadata.DeletedFilesFile,
		}
		for _, del := range cleanup {
			var file string
			if del == metadata.RestoreLogFile || del == stats.StatsRestore {
				// Checkpointing uses runc and it is possible to tell runc
				// the location of the log file using '--work-path'.
				// Restore goes through conmon and conmon does (not yet?)
				// expose runc's '--work-path' which means that temporary
				// restore files are put into BundlePath().
				file = filepath.Join(ctr.BundlePath(), del)
			} else {
				file = filepath.Join(ctr.Dir(), del)
			}

			err = os.Remove(file)
			if err != nil {
				log.Debugf(ctx, "Non-fatal: removal of checkpoint file (%s) failed: %v", file, err)
			}
		}
	}

	return ctr.ID(), nil
}

func (c *ContainerServer) restoreFileSystemChanges(ctr *oci.Container, mountPoint string) error {
	if err := crutils.CRApplyRootFsDiffTar(ctr.Dir(), mountPoint); err != nil {
		return err
	}

	if err := crutils.CRRemoveDeletedFiles(ctr.ID(), ctr.Dir(), mountPoint); err != nil {
		return err
	}

	return nil
}

// deviceMapping represents a single host-to-container device mapping entry
// written to the device-mapping.json file for the CRIU action script.
type deviceMapping struct {
	// HostPath is the device path on the host (e.g., /dev/nvidia0).
	HostPath string `json:"host_path"`
	// ContainerPath is the expected device path inside the container.
	ContainerPath string `json:"container_path"`
	// Type is the device type (e.g., "c" for char, "b" for block).
	Type string `json:"type"`
	// Major is the device major number on the host.
	Major int64 `json:"major"`
	// Minor is the device minor number on the host.
	Minor int64 `json:"minor"`
	// FileMode is the device file permissions (e.g., 0666).
	FileMode os.FileMode `json:"file_mode,omitempty"`
}

// injectCDIDevicesForRestore merges CDI device names from NVIDIA_VISIBLE_DEVICES
// (and CDI annotations) into the OCI spec via cdi.InjectDevices.
func injectCDIDevicesForRestore(ctx context.Context, ctrSpec *generate.Generator) {
	if ctrSpec.Config.Process == nil {
		return
	}

	// Collect CDI device names from NVIDIA_VISIBLE_DEVICES env.
	// KAI scheduler and NVIDIA device plugin set this to CDI-format names
	// like "k8s.device-plugin.nvidia.com/gpu=GPU-7ee58073-..."
	var cdiNames []string
	for _, env := range ctrSpec.Config.Process.Env {
		if !strings.HasPrefix(env, "NVIDIA_VISIBLE_DEVICES=") {
			continue
		}
		val := strings.TrimPrefix(env, "NVIDIA_VISIBLE_DEVICES=")
		if val == "" || val == "void" || val == "none" || val == "all" {
			break
		}
		for _, dev := range strings.Split(val, ",") {
			dev = strings.TrimSpace(dev)
			// CDI qualified name: <vendor>/<class>=<name>
			if idx := strings.LastIndex(dev, "="); idx != -1 && strings.Contains(dev[:idx], "/") {
				cdiNames = append(cdiNames, dev)
			}
		}
		break
	}

	// Also check CDI device annotations (older DRA drivers use these)
	if annots := ctrSpec.Config.Annotations; annots != nil {
		_, annotated, err := cdi.ParseAnnotations(annots)
		if err == nil {
			for _, name := range annotated {
				cdiNames = append(cdiNames, name)
			}
		}
	}

	if len(cdiNames) == 0 {
		return
	}

	// Deduplicate
	seen := make(map[string]bool)
	var unique []string
	for _, name := range cdiNames {
		if !seen[name] {
			seen[name] = true
			unique = append(unique, name)
		}
	}
	cdiNames = unique

	if err := cdi.Refresh(); err != nil {
		log.Warnf(ctx, "CDI registry refresh had errors: %v", err)
	}

	unresolved, err := cdi.InjectDevices(ctrSpec.Config, cdiNames...)
	if err != nil {
		log.Warnf(ctx, "CDI device injection for restore failed: %v (unresolved: %v)", err, unresolved)
		return
	}

	log.Infof(ctx, "CDI devices injected for restore: %v", cdiNames)
}

// writeDeviceRestoreMetadata generates device-mapping.json and the CRIU
// config file inside the checkpoint directory, injects the org.criu.config
// annotation into the OCI spec so runc passes the config to CRIU, and (when
// a resolver and GPU Capability Descriptors are present) writes the GPU
// remap sidecars consumed by the extended action script and by CRIU's CUDA
// plugin guard hook.
//
// claimSources, when non-empty, lists the original vGPU claim bind-mount
// sources recorded in the checkpoint (see vgpuresolver.Resolver); resolver
// pairs them with the current claim sources found in ctrSpec and the
// resulting CRIU ext-mount-map entries let mounts marked external in the
// checkpoint be re-bound from the new host paths. oldCap/newCap, when both
// non-nil, drive the GPU-remap sidecar files.
//
// Returns paths to the created files (empty strings for any file that
// wasn't needed).
func writeDeviceRestoreMetadata(
	ctx context.Context,
	ctr *oci.Container,
	ctrSpec *generate.Generator,
	claimSources []string,
	resolver vgpuresolver.Resolver,
	oldCap, newCap *vgpuresolver.GPUCapability,
) (mappingPath, configPath, remapPath, capConfPath string, err error) {
	extMountLines := buildExtMountMapLines(ctx, resolver, claimSources, ctrSpec)

	// Verify the action script exists on disk.
	if _, statErr := os.Stat(criuDeviceRestorerScript); statErr != nil {
		log.Warnf(ctx, "CRIU device restorer script not found at %s, skipping device restore metadata: %v",
			criuDeviceRestorerScript, statErr)
		// Even without the device action script we still want CRIU to see
		// the vGPU claim ext-mount-map entries, otherwise restore will fail
		// with "No mapping for ... mountpoint".
		if extMountLines == "" {
			return "", "", "", "", nil
		}
	}

	// Build device mapping entries from the OCI spec.
	var mappings []deviceMapping
	if ctrSpec.Config.Linux != nil {
		for _, d := range ctrSpec.Config.Linux.Devices {
			var mode os.FileMode
			if d.FileMode != nil {
				mode = *d.FileMode
			}
			mappings = append(mappings, deviceMapping{
				HostPath:      d.Path,
				ContainerPath: d.Path,
				Type:          d.Type,
				Major:         d.Major,
				Minor:         d.Minor,
				FileMode:      mode,
			})
		}
	}

	checkpointDir := ctr.CheckpointPath()

	// GPU remap sidecars are independent of the device-mapping/action-script
	// machinery above — they exist whenever both capability descriptors are
	// present, regardless of whether there are any /dev entries to remap.
	remapPath, capConfPath, err = writeGPURemapFiles(checkpointDir, oldCap, newCap, resolverName(resolver))
	if err != nil {
		return "", "", "", "", fmt.Errorf("write GPU remap metadata: %w", err)
	}

	if len(mappings) == 0 && extMountLines == "" {
		log.Debugf(ctx, "No devices to map for container %s, skipping device restore metadata", ctr.ID())
		return "", "", remapPath, capConfPath, nil
	}

	if len(mappings) > 0 {
		mappingJSON, marshalErr := json.MarshalIndent(mappings, "", "  ")
		if marshalErr != nil {
			cleanupGPURemapFiles(ctx, remapPath, capConfPath)
			return "", "", "", "", fmt.Errorf("marshal device mappings: %w", marshalErr)
		}

		mappingPath = filepath.Join(checkpointDir, deviceMappingFile)
		if writeErr := os.WriteFile(mappingPath, mappingJSON, 0o600); writeErr != nil {
			cleanupGPURemapFiles(ctx, remapPath, capConfPath)
			return "", "", "", "", fmt.Errorf("write %s: %w", mappingPath, writeErr)
		}

		log.Infof(ctx, "Wrote device mapping (%d devices) to %s", len(mappings), mappingPath)
	}

	// Write a CRIU config file that enables the action script and sets
	// restore options. CRIU reads this via --config and treats each line
	// as a command-line option (without the leading "--").
	//
	// Options:
	//   action-script:           invoke the device restorer script at post-restore
	//   tcp-close:               close stale TCP connections instead of re-establishing them
	//   skip-in-flight:          skip in-flight TCP data during restore (avoids retransmit errors)
	//   log-file:                write CRIU-internal log to /tmp/criu.log for debugging
	//   ghost-limit:             raise ghost file size limit to 100 MiB (needed for large tmpfs/device files)
	//   enable-external-masters: allow external master links in mount tree (common with bind mounts)
	//   ext-mount-map:           remap external mounts captured at checkpoint time
	//                            (used for vGPU claim directories whose
	//                            identity changes on every pod creation)
	var criuConfigContent string
	if _, err := os.Stat(criuDeviceRestorerScript); err == nil {
		criuConfigContent = fmt.Sprintf(
			"action-script %s\ntcp-close\nskip-in-flight\nlog-file /tmp/criu.log\nghost-limit 104857600\nenable-external-masters\n",
			criuDeviceRestorerScript,
		)
	} else {
		criuConfigContent = "tcp-close\nskip-in-flight\nlog-file /tmp/criu.log\nghost-limit 104857600\nenable-external-masters\n"
	}

	criuConfigContent += extMountLines

	configPath = filepath.Join(checkpointDir, criuConfigFile)
	if err := os.WriteFile(configPath, []byte(criuConfigContent), 0o600); err != nil {
		// Clean up files we already wrote.
		if mappingPath != "" {
			os.Remove(mappingPath)
		}
		cleanupGPURemapFiles(ctx, remapPath, capConfPath)
		return "", "", "", "", fmt.Errorf("write %s: %w", configPath, err)
	}

	log.Infof(ctx, "Wrote CRIU config to %s", configPath)

	// Inject the org.criu.config annotation so the OCI runtime passes
	// the CRIU config file path to CRIU during restore.
	ctrSpec.AddAnnotation(annotations.CRIUConfigAnnotation, configPath)

	return mappingPath, configPath, remapPath, capConfPath, nil
}

// resolverName returns resolver.Name(), or "" for a nil resolver — kept as a
// tiny helper so call sites don't need a nil check inline.
func resolverName(resolver vgpuresolver.Resolver) string {
	if resolver == nil {
		return ""
	}
	return resolver.Name()
}

// readClaimSourcesFromArchive extracts spec.dump from a checkpoint tar
// archive (without disturbing the main restore extraction, which omits
// spec.dump) and returns the resolver's claim mount sources from it.
func readClaimSourcesFromArchive(ctx context.Context, archivePath string, resolver vgpuresolver.Resolver) []string {
	tmpDir, err := os.MkdirTemp("", "crio-vgpu-spec-")
	if err != nil {
		log.Warnf(ctx, "Failed to create temp dir for vGPU claim spec.dump: %v", err)
		return nil
	}
	defer os.RemoveAll(tmpDir)

	if err := crutils.CRImportCheckpointConfigOnly(tmpDir, archivePath); err != nil {
		log.Warnf(ctx, "Failed to extract spec.dump from checkpoint archive %s: %v", archivePath, err)
		return nil
	}

	var dumpSpec rspec.Spec
	if _, err := metadata.ReadJSONFile(&dumpSpec, tmpDir, metadata.SpecDumpFile); err != nil {
		log.Warnf(ctx, "Failed to read spec.dump for vGPU claim remap: %v", err)
		return nil
	}
	return resolver.ClaimMountSources(dumpSpec.Mounts)
}

// buildExtMountMapLines pairs each old vGPU claim source path (from the
// checkpoint) with the current host source path found in the OCI spec,
// via resolver.MatchClaimSources, and returns one
// "ext-mount-map <old>:<new>\n" line per matched pair, in the same order as
// oldSources so output is deterministic despite map iteration.
//
// Returns an empty string when there is nothing to remap.
func buildExtMountMapLines(ctx context.Context, resolver vgpuresolver.Resolver, oldSources []string, ctrSpec *generate.Generator) string {
	if resolver == nil || len(oldSources) == 0 {
		return ""
	}
	if ctrSpec == nil || ctrSpec.Config == nil {
		return ""
	}

	newSources := resolver.ClaimMountSources(ctrSpec.Config.Mounts)
	if len(newSources) == 0 {
		log.Warnf(ctx, "%s pod has %d checkpointed vGPU claim mount(s) (%v) but no new claim mounts in the restored spec; skipping ext-mount-map",
			resolver.Name(), len(oldSources), oldSources)
		return ""
	}

	matches := resolver.MatchClaimSources(oldSources, newSources)

	var b strings.Builder
	for _, oldPath := range oldSources {
		newPath, ok := matches[oldPath]
		if !ok {
			log.Warnf(ctx, "No matching new %s vGPU claim mount for checkpointed path %s", resolver.Name(), oldPath)
			continue
		}
		fmt.Fprintf(&b, "ext-mount-map %s:%s\n", oldPath, newPath)
		log.Infof(ctx, "%s vGPU claim ext-mount-map: %s -> %s", resolver.Name(), oldPath, newPath)
	}
	return b.String()
}

// cleanupDeviceRestoreMetadata removes the device-mapping.json and CRIU config
// files that were created for the restore operation.
func cleanupDeviceRestoreMetadata(ctx context.Context, mappingPath, configPath string) {
	if mappingPath != "" {
		if err := os.Remove(mappingPath); err != nil && !os.IsNotExist(err) {
			log.Warnf(ctx, "Failed to clean up device mapping file %s: %v", mappingPath, err)
		} else {
			log.Debugf(ctx, "Cleaned up device mapping file %s", mappingPath)
		}
	}

	if configPath != "" {
		if err := os.Remove(configPath); err != nil && !os.IsNotExist(err) {
			log.Warnf(ctx, "Failed to clean up CRIU config file %s: %v", configPath, err)
		} else {
			log.Debugf(ctx, "Cleaned up CRIU config file %s", configPath)
		}
	}
}
