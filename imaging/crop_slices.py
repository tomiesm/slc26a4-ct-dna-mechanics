"""Crop the coronal CT series of each patient in config.COHORT_LIST to one bounding box around the bone anatomy.

The scan used is the patient's earliest CT before cochlear implantation in config.CT_MANIFEST; patients
with no pre-implant entry there are skipped. The coronal series of that scan is cropped (the first one
listed in config.PNG_INVENTORY when the scan has several). Only when that scan has no coronal series is
the patient's first scan folder by name that has a coronal series used instead. Scan folders whose name
contains "CI" (post-implant CTs) are never used. Patients with fewer than three PNG slices, or with no
anatomy found in any slice, are skipped.

In every slice, pixels above BONE_THRESHOLD outside the corners, side margins and top and bottom bands
(where DICOM overlay text and markers appear) are closed morphologically, and connected components of at
least MIN_AREA_FRAC of the image give that slice's anatomy box. The union of these boxes over all slices,
padded by BBOX_PADDING pixels, is the patient's crop, so all slices of a patient are cropped identically.
Overlay regions that fall inside the crop are set to zero.

Writes one <patient>.npz (key "slices", uint8, shape (slices, height, width)) per patient and
crop_stats.csv into config.CORONAL_CROPS_DIR. That folder holds the crops supplied with the input data,
so running this script remakes the supplied crops and overwrites them.

Usage: python -m imaging.crop_slices
"""

import os

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

import config
from imaging.png_inventory import parse_slice_number

# ── Bounding-box parameters ──────────────────────────────────────────────
BONE_THRESHOLD = 80
MIN_AREA_FRAC = 0.005
CORNER_H = 130
CORNER_W = 220
MARKER_W = 25
BBOX_PADDING = 15


def get_anatomy_bbox(img_array):
    """Get bounding box of bone anatomy, excluding DICOM overlay text."""
    if img_array.ndim == 3:
        gray = img_array.mean(axis=2)
    else:
        gray = img_array.astype(float)

    h, w = gray.shape

    masked = gray.copy()
    masked[:CORNER_H, :CORNER_W] = 0
    masked[:CORNER_H, w - CORNER_W:] = 0
    masked[h - CORNER_H:, :CORNER_W] = 0
    masked[h - CORNER_H:, w - CORNER_W:] = 0
    masked[:, :MARKER_W] = 0
    masked[:, w - MARKER_W:] = 0
    masked[:20, :] = 0
    masked[h - 20:, :] = 0

    binary = masked > BONE_THRESHOLD
    min_area = int(binary.size * MIN_AREA_FRAC)

    struct = ndimage.generate_binary_structure(2, 2)
    binary = ndimage.binary_closing(binary, structure=struct, iterations=5)

    labeled, n_comp = ndimage.label(binary)
    if n_comp == 0:
        return None

    comp_sizes = ndimage.sum(binary, labeled, range(1, n_comp + 1))
    kept = [i + 1 for i, s in enumerate(comp_sizes) if s >= min_area]
    if not kept:
        return None

    kept_mask = np.isin(labeled, kept)
    rows = np.any(kept_mask, axis=1)
    cols = np.any(kept_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return (int(cmin), int(rmin), int(cmax + 1), int(rmax + 1))


def compute_union_bbox_all_slices(cor_dir, pngs, img_shape):
    """Union of the anatomy boxes of all slices, padded by BBOX_PADDING, and the number of slices with a box."""
    bboxes = []
    for png_name in pngs:
        img = Image.open(cor_dir / png_name)
        arr = np.array(img)
        bb = get_anatomy_bbox(arr)
        if bb is not None:
            bboxes.append(bb)

    if not bboxes:
        return None

    h, w = img_shape
    x0 = max(0, min(b[0] for b in bboxes) - BBOX_PADDING)
    y0 = max(0, min(b[1] for b in bboxes) - BBOX_PADDING)
    x1 = min(w, max(b[2] for b in bboxes) + BBOX_PADDING)
    y1 = min(h, max(b[3] for b in bboxes) + BBOX_PADDING)

    return (x0, y0, x1, y1), len(bboxes)


def mask_dicom_text(cropped_slice, bbox, orig_shape):
    """Set to zero the parts of the crop that lie in the overlay regions (corners, side margins, top and
    bottom bands) of the full image."""
    x0, y0, x1, y1 = bbox
    orig_h, orig_w = orig_shape

    text_regions = [
        (0, 0, CORNER_W, CORNER_H),
        (orig_w - CORNER_W, 0, orig_w, CORNER_H),
        (0, orig_h - CORNER_H, CORNER_W, orig_h),
        (orig_w - CORNER_W, orig_h - CORNER_H, orig_w, orig_h),
        (0, 0, MARKER_W, orig_h),
        (orig_w - MARKER_W, 0, orig_w, orig_h),
        (0, 0, orig_w, 20),
        (0, orig_h - 20, orig_w, orig_h),
    ]

    for tx0, ty0, tx1, ty1 in text_regions:
        cx0 = max(0, tx0 - x0)
        cy0 = max(0, ty0 - y0)
        cx1 = min(x1 - x0, tx1 - x0)
        cy1 = min(y1 - y0, ty1 - y0)
        if cx0 < cx1 and cy0 < cy1:
            cropped_slice[cy0:cy1, cx0:cx1] = 0

    return cropped_slice


def load_and_sort_pngs(coronal_dir):
    pngs = sorted(
        [f for f in os.listdir(coronal_dir) if f.lower().endswith(".png")],
        key=lambda f: parse_slice_number(f) or 0
    )
    return pngs


def find_coronal_series(patient_ids):
    """Coronal series folder of each patient, and the patients without one (with the reason)."""
    inv = pd.read_csv(config.PNG_INVENTORY)
    has_cor = inv[inv["coronal_subfolder"].notna()].copy()
    has_cor = has_cor[~has_cor["scan_folder"].str.contains("CI", case=True, na=False)]

    manifest = pd.read_csv(config.CT_MANIFEST)
    pre_ci = manifest[manifest["ci_status"] == "pre-CI"].copy()
    pre_ci["ct_date"] = pd.to_datetime(pre_ci["ct_date"])
    pre_ci = pre_ci.sort_values("ct_date").groupby("patient_id").first().reset_index()
    pre_ci["scan_folder"] = pre_ci["filename"].str.replace(".npz", "", regex=False)
    scan_map = dict(zip(pre_ci["patient_id"], pre_ci["scan_folder"]))

    patient_coronal = {}
    missing = []

    for pid in sorted(patient_ids):
        target_scan = scan_map.get(pid)
        if target_scan is None:
            missing.append((pid, "no pre-implant entry in the CT manifest"))
            continue

        match = has_cor[
            (has_cor["patient_id"] == pid) &
            (has_cor["scan_folder"] == target_scan)
        ]

        if len(match) == 0:
            match = has_cor[has_cor["patient_id"] == pid]
            if len(match) == 0:
                missing.append((pid, "no coronal folder"))
                continue
            match = match.sort_values("scan_folder", kind="stable").iloc[[0]]

        row = match.iloc[0]
        cor_path = config.CT_PNG_DIR / row["scan_folder"] / row["coronal_subfolder"]
        if cor_path.is_dir():
            patient_coronal[pid] = cor_path
        else:
            missing.append((pid, f"dir not found: {cor_path}"))

    return patient_coronal, missing


def main():
    config.CORONAL_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    patient_ids = set(pd.read_csv(config.COHORT_LIST)["patient_id"].astype(str))
    print(f"Patients in the cohort list: {len(patient_ids)}")

    patient_coronal, missing = find_coronal_series(patient_ids)
    print(f"Patients with a coronal series: {len(patient_coronal)}")
    if missing:
        print(f"Missing ({len(missing)}):")
        for pid, reason in missing:
            print(f"  {pid}: {reason}")

    print(f"\nCropping {len(patient_coronal)} patients (bounding box from all slices)...")

    stats = []
    failed = []

    for pid in tqdm(sorted(patient_coronal.keys()), desc="Cropping"):
        cor_dir = patient_coronal[pid]
        pngs = load_and_sort_pngs(cor_dir)

        if len(pngs) < 3:
            failed.append((pid, f"only {len(pngs)} PNGs"))
            continue

        # Image shape from the first slice
        first_img = Image.open(cor_dir / pngs[0])
        img_shape = np.array(first_img).shape[:2]

        result = compute_union_bbox_all_slices(cor_dir, pngs, img_shape)
        if result is None:
            failed.append((pid, "no anatomy detected in any slice"))
            continue

        bbox, n_bbox_slices = result
        x0, y0, x1, y1 = bbox

        # Crop all slices with the same box
        cropped_slices = []
        for png_name in pngs:
            img = Image.open(cor_dir / png_name)
            arr = np.array(img)
            if arr.ndim == 3:
                gray = np.mean(arr, axis=2).astype(np.uint8)
            else:
                gray = arr.astype(np.uint8)
            cropped = gray[y0:y1, x0:x1]
            cropped = mask_dicom_text(cropped, bbox, img_shape)
            cropped_slices.append(cropped)

        volume = np.stack(cropped_slices, axis=0)

        out_path = config.CORONAL_CROPS_DIR / f"{pid}.npz"
        np.savez_compressed(out_path, slices=volume)

        stats.append({
            "patient_id": pid,
            "n_slices": volume.shape[0],
            "crop_h": volume.shape[1],
            "crop_w": volume.shape[2],
            "bbox_x0": x0, "bbox_y0": y0,
            "bbox_x1": x1, "bbox_y1": y1,
            "orig_h": img_shape[0],
            "orig_w": img_shape[1],
            "n_bbox_slices": n_bbox_slices,
            "file_size_mb": out_path.stat().st_size / 1e6,
        })

    df = pd.DataFrame(stats)
    print(f"\nPatients cropped: {len(df)}")
    print(f"Failed: {len(failed)}")
    for pid, reason in failed:
        print(f"  {pid}: {reason}")

    if len(df) > 0:
        print(f"Slice counts: min={df['n_slices'].min()}, max={df['n_slices'].max()}, "
              f"median={df['n_slices'].median():.0f}, mean={df['n_slices'].mean():.0f}")
        print(f"Crop height:  min={df['crop_h'].min()}, max={df['crop_h'].max()}, "
              f"mean={df['crop_h'].mean():.0f}")
        print(f"Crop width:   min={df['crop_w'].min()}, max={df['crop_w'].max()}, "
              f"mean={df['crop_w'].mean():.0f}")
        print(f"Slices with anatomy: min={df['n_bbox_slices'].min()}, "
              f"mean={df['n_bbox_slices'].mean():.0f}, max={df['n_bbox_slices'].max()}")
        print(f"File sizes:   min={df['file_size_mb'].min():.1f} MB, "
              f"max={df['file_size_mb'].max():.1f} MB, "
              f"total={df['file_size_mb'].sum():.1f} MB")

        stats_path = config.CORONAL_CROPS_DIR / "crop_stats.csv"
        df.to_csv(stats_path, index=False)
        print(f"\nSaved {stats_path} and the crops in {config.CORONAL_CROPS_DIR}")


if __name__ == "__main__":
    main()
