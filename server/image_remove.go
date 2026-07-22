package server

import (
	"context"
	"errors"
	"fmt"

	storagetypes "go.podman.io/storage"
	types "k8s.io/cri-api/pkg/apis/runtime/v1"

	"github.com/cri-o/cri-o/internal/log"
	"github.com/cri-o/cri-o/internal/ociartifact"
	"github.com/cri-o/cri-o/internal/storage"
)

// RemoveImage removes the image.
func (s *Server) RemoveImage(ctx context.Context, req *types.RemoveImageRequest) (*types.RemoveImageResponse, error) {
	ctx, span := log.StartSpan(ctx)
	defer span.End()

	imageRef := ""
	img := req.GetImage()

	if img != nil {
		imageRef = img.GetImage()
	}

	if imageRef == "" {
		return nil, errors.New("no image specified")
	}

	if err := s.removeImage(ctx, imageRef); err != nil {
		return nil, err
	}

	return &types.RemoveImageResponse{}, nil
}

func (s *Server) removeImage(ctx context.Context, imageRef string) (untagErr error) {
	ctx, span := log.StartSpan(ctx)
	defer span.End()

	if id := s.ContainerServer.StorageImageServer().HeuristicallyTryResolvingStringAsIDPrefix(imageRef); id != nil {
		if err := s.volumeInUse(id.IDStringForOutOfProcessConsumptionOnly()); err != nil {
			return err
		}

		if err := s.ContainerServer.StorageImageServer().DeleteImage(s.config.SystemContext, *id); err != nil {
			if errors.Is(err, storagetypes.ErrImageUnknown) {
				// imageRef matched the FORMAT of a full image ID but no
				// image actually has that ID — it may really be a manifest
				// digest's hex portion instead (same shape, different
				// value; see FindLocallyStoredImageMatchingDigest's doc
				// comment). Try that before treating this as "already
				// removed."
				if digestID, derr := s.ContainerServer.StorageImageServer().FindLocallyStoredImageMatchingDigest(imageRef); derr == nil && digestID != nil {
					if err := s.ContainerServer.StorageImageServer().DeleteImage(s.config.SystemContext, *digestID); err != nil && !errors.Is(err, storagetypes.ErrImageUnknown) {
						return fmt.Errorf("delete image: %w", err)
					}
				}

				// The RemoveImage RPC is idempotent, and must not return an
				// error if the image has already been removed. Ref:
				// https://github.com/kubernetes/cri-api/blob/c20fa40/pkg/apis/runtime/v1/api.proto#L156-L157
				return nil
			}

			return fmt.Errorf("delete image: %w", err)
		}

		return nil
	}

	var (
		deleted   bool
		statusErr error
	)

	potentialMatches, err := s.ContainerServer.StorageImageServer().CandidatesForPotentiallyShortImageName(s.config.SystemContext, imageRef)
	if err != nil {
		return err
	}

	for _, name := range potentialMatches {
		var status *storage.ImageResult

		status, statusErr = s.ContainerServer.StorageImageServer().ImageStatusByName(s.config.SystemContext, name)
		if statusErr != nil {
			log.Warnf(ctx, "Error getting image status %s: %v", name, statusErr)

			continue
		}

		if err := s.volumeInUse(status.ID.IDStringForOutOfProcessConsumptionOnly()); err != nil {
			return err
		}

		untagErr = s.ContainerServer.StorageImageServer().UntagImage(s.config.SystemContext, name)
		if untagErr != nil {
			log.Debugf(ctx, "Error deleting image %s: %v", name, untagErr)

			continue
		}

		deleted = true

		break
	}

	// None of the registries.conf-driven candidates matched an image to
	// untag, but the image may still be present in local storage under a
	// name outside that candidate set — see
	// FindLocallyStoredImageMatchingName's doc comment. Delete by ID
	// directly (mirroring the ID-prefix branch above) rather than via
	// UntagImage, since the fallback resolved an ID, not a candidate name.
	if !deleted {
		if id, ferr := s.ContainerServer.StorageImageServer().FindLocallyStoredImageMatchingName(imageRef); ferr == nil && id != nil {
			if err := s.volumeInUse(id.IDStringForOutOfProcessConsumptionOnly()); err != nil {
				return err
			}

			if err := s.ContainerServer.StorageImageServer().DeleteImage(s.config.SystemContext, *id); err != nil && !errors.Is(err, storagetypes.ErrImageUnknown) {
				return fmt.Errorf("delete image: %w", err)
			}

			deleted = true
			untagErr = nil
		}
	}

	if !deleted && untagErr != nil {
		return untagErr
	}

	artifact, err := s.ArtifactStore().Status(ctx, imageRef)
	if errors.Is(err, ociartifact.ErrNotFound) {
		return nil
	}

	if err != nil {
		return fmt.Errorf("failed to get artifact: %w", err)
	}

	if err := s.volumeInUse(artifact.Digest().Encoded()); err != nil {
		return err
	}

	if err := s.ArtifactStore().Remove(ctx, imageRef); err != nil && !errors.Is(err, ociartifact.ErrNotFound) {
		log.Errorf(ctx, "Unable to remove artifact: %v", err)
	}

	if errors.Is(statusErr, storagetypes.ErrNotAnImage) {
		// The RemoveImage RPC is idempotent, and must not return an
		// error if the image has already been removed. Ref:
		// https://github.com/kubernetes/cri-api/blob/c20fa40/pkg/apis/runtime/v1/api.proto#L156-L157
		return nil
	}

	return nil
}

// volumeInUse returns nil if it's not in use.
// It doesn't check if it's used as a container image because the check is done in storage pkg instead.
func (s *Server) volumeInUse(digest string) error {
	containerList, err := s.ContainerServer.ListContainers()
	if err != nil {
		return fmt.Errorf("error listing containers: %w", err)
	}

	for _, container := range containerList {
		for _, volume := range container.Volumes() {
			if volume.Image.GetImage() == digest {
				return fmt.Errorf("the image is in use by %s", container.ID())
			}
		}
	}

	return nil
}
