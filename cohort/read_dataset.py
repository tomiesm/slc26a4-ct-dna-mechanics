"""Read the clinical dataset into one table of ear records and list the axial CT files.

The "malformation" and "normal" sheets hold the same fields, but from column 15 on the fields of the
normal sheet are shifted by three columns; both sheets are mapped to one set of column names. Each
axial CT file name gives the patient, the CT date and, when present, an implant note; the files are
linked to the records by patient and CT date.

Usage: python -m cohort.read_dataset
"""

import re
from datetime import datetime

import numpy as np
import openpyxl
import pandas as pd

import config

# ── Column mappings (1-indexed Excel columns → unified names) ──────────────
# Columns 1-14 are identical in both sheets.
_COMMON_COLS = {
    1: "op_date",
    2: "laterality",
    3: "patient_id",
    4: "birthday",
    5: "gender",
    6: "ct_date",
    7: "cochlea",
    8: "post_ci",
    9: "vestibule",
    10: "scc",
    11: "eva",
    12: "iac",
    13: "aperture",
    14: "genetic_mutation",
}

# Columns that differ between sheets due to the 3-column offset
_MALFORMATION_EXTRA = {
    15: "underlying",
    16: "ci_op_finding",
    17: "ref_ct_date",
    18: "iem_flag",
    19: "hjb",
    20: "cn",
    21: "fn_old",
    22: "fn_meatal",
    23: "fn_ls",
    24: "fn_ts",
    25: "fn_ms",
    26: "fn_dehiscence",
    27: "fn_anomaly",
    28: "mri_report",
    29: "ct_report",
}

_NORMAL_EXTRA = {
    # 15-17 are empty in the normal sheet (gap columns)
    18: "underlying",
    19: "ci_op_finding",
    20: "ref_ct_date",
    21: "iem_flag",
    22: "hjb",
    23: "cn",
    24: "fn_old",
    25: "fn_meatal",
    26: "fn_ls",
    27: "fn_ts",
    28: "fn_ms",
    29: "fn_dehiscence",
    30: "fn_anomaly",
    31: "mri_report",
    32: "ct_report",
}


def _parse_date_value(val):
    """Parse a cell value that should be a date; return NaT when it is not one."""
    if val is None:
        return pd.NaT
    if isinstance(val, datetime):
        if val.year == 9999:
            return pd.NaT  # sentinel for "no surgery"
        return pd.Timestamp(val)
    s = str(val).strip()
    if s.lower() in ("pending", "no ct", "motion", ""):
        return pd.NaT
    if "9999" in s:
        return pd.NaT
    # Try common formats
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%Y%m%d"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year == 9999:
                return pd.NaT
            return pd.Timestamp(dt)
        except ValueError:
            continue
    return pd.NaT


def _detect_had_surgery(val):
    """Determine whether the op_date indicates surgery occurred."""
    if val is None:
        return False
    if isinstance(val, datetime):
        return val.year != 9999
    s = str(val).strip().lower()
    if s in ("pending", "no ct", "motion", ""):
        return False
    if "9999" in s:
        return False
    # If it parsed to a real date, surgery happened
    return _parse_date_value(val) is not pd.NaT


def _safe_int(val):
    """Convert a cell value to int, or NaN."""
    if val is None:
        return np.nan
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return np.nan
        return int(val)
    s = str(val).strip().lower()
    if s in ("", "na", "none", "pending"):
        return np.nan
    if s == "mondini":
        return 1  # IEM positive
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return np.nan


def _safe_str(val):
    """Convert a cell value to string, or None."""
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None


def read_sheet(wb, sheet_name: str) -> pd.DataFrame:
    """Read a sheet row-by-row and return a DataFrame with the unified schema."""
    ws = wb[sheet_name]
    is_malformation = sheet_name == "malformation"
    extra_cols = _MALFORMATION_EXTRA if is_malformation else _NORMAL_EXTRA
    col_map = {**_COMMON_COLS, **extra_cols}

    rows = []
    for r in range(3, ws.max_row + 1):
        # Skip completely empty rows
        pid = ws.cell(row=r, column=3).value
        if pid is None:
            continue

        row_dict = {}
        for excel_col, unified_name in col_map.items():
            val = ws.cell(row=r, column=excel_col).value
            row_dict[unified_name] = val

        rows.append(row_dict)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ── Type conversions ──────────────────────────────────────────────────
    # Detect had_surgery BEFORE parsing op_date to NaT
    df["had_surgery"] = df["op_date"].apply(_detect_had_surgery)

    # Parse dates
    for col in ["op_date", "ct_date", "birthday", "ref_ct_date"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_date_value)

    # Patient ID: always string, stripped
    df["patient_id"] = df["patient_id"].astype(str).str.strip()

    # An ID written as "<label> (=<id>)" is reduced to <id>.
    _alias_re = re.compile(r".*\(=\s*([A-Za-z0-9_]+)\s*\)\s*$")
    def _normalize_pid(pid: str) -> str:
        m = _alias_re.match(pid)
        return m.group(1) if m else pid
    df["patient_id"] = df["patient_id"].apply(_normalize_pid)

    # Laterality: always string
    df["laterality"] = df["laterality"].astype(str).str.strip().str.upper()

    # Gender: always string
    df["gender"] = df["gender"].astype(str).str.strip().str.upper()

    # Integer anatomical codes
    for col in ["cochlea", "post_ci", "vestibule", "scc", "eva", "iac", "aperture",
                "iem_flag", "hjb", "cn", "fn_meatal", "fn_ts", "fn_ms", "fn_anomaly"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_int)

    # fn_old: integer code (values 0-4)
    if "fn_old" in df.columns:
        df["fn_old"] = df["fn_old"].apply(_safe_int)

    # fn_ls: keep as string (has "2a", "2b", "2c" values)
    if "fn_ls" in df.columns:
        df["fn_ls"] = df["fn_ls"].apply(_safe_str)

    # fn_dehiscence: keep as string ("0", "1", "2", "3", "NA")
    if "fn_dehiscence" in df.columns:
        df["fn_dehiscence"] = df["fn_dehiscence"].apply(_safe_str)

    # Free-text columns: keep as string
    for col in ["genetic_mutation", "underlying", "ci_op_finding",
                "mri_report", "ct_report"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_str)

    # Add sheet source
    df["sheet_source"] = sheet_name

    return df


def build_npz_manifest() -> pd.DataFrame:
    """Parse all axial CT file names into a table."""
    records = []
    for f in sorted(config.CT_AXIAL_DIR.iterdir()):
        if not f.name.endswith(".npz"):
            continue
        name = f.stem  # filename without .npz
        # Split on first underscore to get patient_id and rest
        parts = name.split("_", 1)
        if len(parts) != 2:
            print(f"  WARNING: unexpected filename format: {f.name}")
            continue

        patient_id = parts[0]
        rest = parts[1]

        # Extract date (first 8 digits) and CI info
        date_match = re.match(r"(\d{8})(.*)", rest)
        if not date_match:
            print(f"  WARNING: no date found in filename: {f.name}")
            continue

        ct_date_str = date_match.group(1)
        ci_part = date_match.group(2).strip()

        if ci_part:
            ci_status = ci_part  # "right CI", "left CI", "bilateral CI"
        else:
            ci_status = "pre-CI"

        records.append({
            "filename": f.name,
            "patient_id": patient_id,
            "ct_date_str": ct_date_str,
            "ct_date": pd.Timestamp(datetime.strptime(ct_date_str, "%Y%m%d")),
            "ci_status": ci_status,
        })

    return pd.DataFrame(records)


def link_npz_to_master(npz_manifest: pd.DataFrame,
                       master_df: pd.DataFrame) -> pd.DataFrame:
    """Add has_npz and npz_filename columns to master_df."""
    # Build a set of (patient_id, ct_date) tuples from the CT file table
    npz_keys = set()
    npz_fname_map = {}
    for _, row in npz_manifest.iterrows():
        key = (row["patient_id"], row["ct_date"])
        npz_keys.add(key)
        npz_fname_map[key] = row["filename"]

    master_df["has_npz"] = False
    master_df["npz_filename"] = None

    for idx, row in master_df.iterrows():
        key = (row["patient_id"], row["ct_date"])
        if key in npz_keys:
            master_df.at[idx, "has_npz"] = True
            master_df.at[idx, "npz_filename"] = npz_fname_map.get(key)

    return master_df


def validate_master_df(master_df: pd.DataFrame, npz_manifest: pd.DataFrame):
    """Run validation checks on the merged record table."""
    print("\n── Validation ──")

    # 1. Row count
    n_mal = (master_df["sheet_source"] == "malformation").sum()
    n_norm = (master_df["sheet_source"] == "normal").sum()
    print(f"  Total rows: {len(master_df)} (malformation={n_mal}, normal={n_norm})")
    assert len(master_df) > 900, f"Expected ~938 rows, got {len(master_df)}"

    # 2. Laterality
    lat_vals = master_df["laterality"].unique()
    assert set(lat_vals) <= {"R", "L"}, f"Unexpected laterality values: {lat_vals}"
    print(f"  Laterality values: {sorted(lat_vals)} ✓")

    # 3. R/L pairing — every (patient_id, ct_date) should have exactly 2 rows
    pair_counts = master_df.groupby(["patient_id", "ct_date"]).size()
    bad_pairs = pair_counts[pair_counts != 2]
    if len(bad_pairs) > 0:
        print(f"  WARNING: {len(bad_pairs)} patient-date groups don't have exactly 2 rows:")
        print(f"    {bad_pairs.head(10)}")
    else:
        print(f"  All {len(pair_counts)} patient-date groups have R/L pairs ✓")

    # 4. CT file linkage — every file belongs to a patient in the dataset
    npz_patients = set(npz_manifest["patient_id"].unique())
    master_patients = set(master_df["patient_id"].unique())
    orphan_npz = npz_patients - master_patients
    if orphan_npz:
        print(f"  WARNING: {len(orphan_npz)} CT-file patients not in the dataset: {orphan_npz}")
    else:
        print(f"  All {len(npz_patients)} CT-file patients found in the dataset ✓")

    # 5. CT date matching
    npz_matched = master_df["has_npz"].sum()
    print(f"  Rows with a CT file: {npz_matched} / {len(master_df)}")

    # Count unique patients with a CT file
    patients_with_npz = master_df[master_df["has_npz"]]["patient_id"].nunique()
    print(f"  Unique patients with a CT file: {patients_with_npz}")

    # 6. No overlap between sheets
    mal_ids = set(master_df[master_df["sheet_source"] == "malformation"]["patient_id"])
    norm_ids = set(master_df[master_df["sheet_source"] == "normal"]["patient_id"])
    overlap = mal_ids & norm_ids
    assert len(overlap) == 0, f"Patient IDs in both sheets: {overlap}"
    print(f"  No patient ID overlap between sheets ✓")

    print("── Validation complete ──\n")


def main():
    config.ensure_dirs()
    print("=" * 60)
    print("Clinical dataset and axial CT files")
    print("=" * 60)

    # ── Read Excel ─────────────────────────────────────────────────────
    print("\nReading the clinical dataset...")
    wb = openpyxl.load_workbook(str(config.CLINICAL_DATASET), data_only=True)

    print("  Reading malformation sheet...")
    df_mal = read_sheet(wb, "malformation")
    print(f"    → {len(df_mal)} rows, {df_mal['patient_id'].nunique()} patients")

    print("  Reading normal sheet...")
    df_norm = read_sheet(wb, "normal")
    print(f"    → {len(df_norm)} rows, {df_norm['patient_id'].nunique()} patients")

    # ── Merge ──────────────────────────────────────────────────────────
    print("\nMerging the two sheets...")
    # Ensure both DataFrames have the same columns
    all_cols = sorted(set(df_mal.columns) | set(df_norm.columns))
    for col in all_cols:
        if col not in df_mal.columns:
            df_mal[col] = None
        if col not in df_norm.columns:
            df_norm[col] = None

    master_df = pd.concat([df_mal[all_cols], df_norm[all_cols]], ignore_index=True)
    print(f"  Records: {len(master_df)} rows, {master_df['patient_id'].nunique()} patients")

    # ── CT file table ──────────────────────────────────────────────────
    print("\nListing axial CT files...")
    npz_manifest = build_npz_manifest()
    print(f"  {len(npz_manifest)} CT files parsed")
    print(f"  {npz_manifest['patient_id'].nunique()} unique patients")
    print(f"  CI status distribution:")
    for status, count in npz_manifest["ci_status"].value_counts().items():
        print(f"    {status}: {count}")

    # ── Link CT files to records ───────────────────────────────────────
    print("\nLinking CT files to the records...")
    master_df = link_npz_to_master(npz_manifest, master_df)

    # ── Computed columns ───────────────────────────────────────────────
    # has_any_abnormality: True if any anatomical structure is non-zero
    abnormality_cols = ["cochlea", "vestibule", "scc", "eva", "iac", "aperture"]
    master_df["has_any_abnormality"] = master_df[abnormality_cols].apply(
        lambda row: any(pd.notna(v) and v != 0 for v in row), axis=1
    )
    n_abn_rows = master_df["has_any_abnormality"].sum()
    print(f"\n  has_any_abnormality: {n_abn_rows} rows ({100 * n_abn_rows / len(master_df):.1f}%)")

    # ── Validate ───────────────────────────────────────────────────────
    validate_master_df(master_df, npz_manifest)

    # ── Save ───────────────────────────────────────────────────────────
    master_df.to_parquet(config.RECORDS, index=False)
    npz_manifest.to_csv(config.CT_MANIFEST, index=False)
    print(f"Saved {config.RECORDS} ({len(master_df)} rows)")
    print(f"Saved {config.CT_MANIFEST} ({len(npz_manifest)} rows)")

    return master_df, npz_manifest


if __name__ == "__main__":
    main()
