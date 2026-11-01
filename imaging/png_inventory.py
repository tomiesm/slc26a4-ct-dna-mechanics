"""List the exported coronal CT series of every scan folder in config.CT_PNG_DIR and write config.PNG_INVENTORY.

Scan folders are named <patient>_<YYYYMMDD>[ <note>]; every subfolder whose name contains "coronal" (any case)
is one coronal series and gets one row with its PNG count, and the first and last slice number and the file
sizes of the first, middle and last slice among the PNGs whose name holds a slice number "(n)". A scan folder
without a coronal subfolder gets one row with n_pngs 0 and empty series fields.

Usage: python -m imaging.png_inventory
"""

import os
import re

import pandas as pd

import config


def parse_slice_number(filename):
    """Slice number from a file name such as 'image (247).png'."""
    m = re.search(r'\((\d+)\)', filename)
    return int(m.group(1)) if m else None


def parse_patient_id(folder_name):
    """Patient identifier from a scan folder name '<patient>_<YYYYMMDD>[ <note>]'."""
    # Everything before the first underscore followed by a date
    m = re.match(r'^(.+?)_\d{8}', folder_name)
    return m.group(1) if m else folder_name


def main():
    config.ensure_dirs()

    inventory = []
    scan_folders = sorted(os.listdir(config.CT_PNG_DIR))

    for scan_folder in scan_folders:
        scan_path = config.CT_PNG_DIR / scan_folder
        if not scan_path.is_dir():
            continue

        patient_id = parse_patient_id(scan_folder)

        # Find coronal subfolders
        coronal_dirs = []
        for item in sorted(os.listdir(scan_path)):
            item_path = scan_path / item
            if item_path.is_dir() and "coronal" in item.lower():
                coronal_dirs.append(item)

        if not coronal_dirs:
            inventory.append({
                "scan_folder": scan_folder,
                "patient_id": patient_id,
                "coronal_subfolder": None,
                "n_pngs": 0,
                "first_slice": None,
                "last_slice": None,
                "file_size_first": None,
                "file_size_mid": None,
                "file_size_last": None,
            })
            continue

        for cor_dir in coronal_dirs:
            cor_path = scan_path / cor_dir
            pngs = [f for f in os.listdir(cor_path) if f.lower().endswith(".png")]
            slice_nums = []
            for p in pngs:
                sn = parse_slice_number(p)
                if sn is not None:
                    slice_nums.append((sn, p))
            slice_nums.sort()

            n_pngs = len(pngs)
            first_slice = slice_nums[0][0] if slice_nums else None
            last_slice = slice_nums[-1][0] if slice_nums else None

            # File sizes for first, middle, last
            fs_first = fs_mid = fs_last = None
            if slice_nums:
                fs_first = (cor_path / slice_nums[0][1]).stat().st_size
                mid_idx = len(slice_nums) // 2
                fs_mid = (cor_path / slice_nums[mid_idx][1]).stat().st_size
                fs_last = (cor_path / slice_nums[-1][1]).stat().st_size

            inventory.append({
                "scan_folder": scan_folder,
                "patient_id": patient_id,
                "coronal_subfolder": cor_dir,
                "n_pngs": n_pngs,
                "first_slice": first_slice,
                "last_slice": last_slice,
                "file_size_first": fs_first,
                "file_size_mid": fs_mid,
                "file_size_last": fs_last,
            })

    inv_df = pd.DataFrame(inventory)
    inv_df.to_csv(config.PNG_INVENTORY, index=False)

    has_coronal = inv_df[inv_df["coronal_subfolder"].notna()]
    no_coronal = inv_df[inv_df["coronal_subfolder"].isna()]
    patients_with = set(has_coronal["patient_id"].unique())
    # A patient can have one scan with a coronal series and another without
    patients_without_any = set(no_coronal["patient_id"].unique()) - patients_with

    print(f"Scan folders: {len(scan_folders)}")
    print(f"Coronal series: {len(has_coronal)}")
    print(f"Scan folders without a coronal series: {len(no_coronal)}")
    print(f"Patients with a coronal series: {len(patients_with)}")
    print(f"Patients without any coronal series: {len(patients_without_any)}")
    print(f"Saved {config.PNG_INVENTORY}")


if __name__ == "__main__":
    main()
