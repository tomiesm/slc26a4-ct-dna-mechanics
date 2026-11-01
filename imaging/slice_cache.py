"""Resize the cropped coronal slices and store one array per patient in config.SLICE_DIR.

Each slice is scaled to [0, 1], resized to config.SLICE_HEIGHT x config.SLICE_WIDTH by bilinear interpolation
(align_corners=False), multiplied by 255 and truncated to uint8. A patient's slices are saved as
<patient>.npy with shape (slices, SLICE_HEIGHT, SLICE_WIDTH); patients that already have a file are skipped.
The CNN scripts read these files and call prepare_cache for the patients they use.

Usage: python -m imaging.slice_cache
"""

import numpy as np
import torch
import torch.nn.functional as F

import config


def prepare_cache(patient_ids):
    """Write the resized slices of every patient in patient_ids that has no file in config.SLICE_DIR yet."""
    config.ensure_dirs()
    done = 0
    skipped = 0
    for pid in patient_ids:
        out_path = config.SLICE_DIR / f"{pid}.npy"
        if out_path.exists():
            skipped += 1
            continue
        data = np.load(config.CORONAL_CROPS_DIR / f"{pid}.npz")
        slices = data["slices"]  # (N, H, W) uint8

        # Resize directly to the target size (no padding)
        t = torch.from_numpy(slices).float().unsqueeze(1) / 255.0  # (N,1,H,W)
        t = F.interpolate(t, size=(config.SLICE_HEIGHT, config.SLICE_WIDTH),
                          mode="bilinear", align_corners=False)

        resized = (t.squeeze(1).numpy() * 255).astype(np.uint8)  # (N,H,W)
        np.save(out_path, resized)
        done += 1
    print(f"    Slice cache: {done} created, {skipped} already cached "
          f"in {config.SLICE_DIR}")


def main():
    patient_ids = sorted(p.stem for p in config.CORONAL_CROPS_DIR.glob("*.npz"))
    print(f"Patients with a coronal crop: {len(patient_ids)}")
    print(f"Resize target: ({config.SLICE_HEIGHT}, {config.SLICE_WIDTH})")

    prepare_cache(patient_ids)
    print("\nSlice cache complete.")


if __name__ == "__main__":
    main()
