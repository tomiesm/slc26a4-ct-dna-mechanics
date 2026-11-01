"""Build one row per patient record with the bilateral grades.

The grades of each ear come from the reference CT (the first recorded reference date when it has at
least two ear rows, otherwise the earliest CT).

Usage: python -m cohort.patient_table
"""

import numpy as np
import pandas as pd

import config

# The 7 graded structures, each with a right (_R) and left (_L) grade
GRADED_STRUCTURES = ["cochlea", "vestibule", "scc", "eva", "iac", "aperture", "cn"]


def _get_reference_rows(patient_group: pd.DataFrame) -> pd.DataFrame:
    """Return the reference annotation rows of a patient.

    Return the first two rows of the first recorded reference CT date when that date has at least two
    rows; otherwise the first two rows (or all rows, if fewer) of the earliest CT date.
    """
    df = patient_group.copy()

    # If ref_ct_date is available and matches some rows
    if df["ref_ct_date"].notna().any():
        ref_date = df["ref_ct_date"].dropna().iloc[0]
        ref_rows = df[df["ct_date"] == ref_date]
        if len(ref_rows) >= 2:
            return ref_rows.head(2)

    # Fall back to earliest ct_date
    earliest = df["ct_date"].min()
    earliest_rows = df[df["ct_date"] == earliest]
    if len(earliest_rows) >= 2:
        return earliest_rows.head(2)

    # If we still don't have 2 rows, return what we have
    return earliest_rows


def build_patient_df(master_df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-patient summary DataFrame."""
    records = []

    for pid, group in master_df.groupby("patient_id"):
        ref_rows = _get_reference_rows(group)

        # Get R and L rows
        r_rows = ref_rows[ref_rows["laterality"] == "R"]
        l_rows = ref_rows[ref_rows["laterality"] == "L"]

        rec = {"patient_id": pid}

        # Demographics (constant per patient)
        rec["sheet_source"] = group["sheet_source"].iloc[0]
        rec["gender"] = group["gender"].iloc[0]
        rec["birthday"] = group["birthday"].iloc[0]

        # ID prefix
        prefix = ""
        for ch in pid:
            if ch.isdigit():
                break
            prefix += ch
        rec["id_prefix"] = prefix

        # Age at first CT
        first_ct = group["ct_date"].min()
        if pd.notna(rec["birthday"]) and pd.notna(first_ct):
            rec["age_at_first_ct"] = (first_ct - rec["birthday"]).days / 365.25
        else:
            rec["age_at_first_ct"] = np.nan

        # Scan counts
        rec["n_ct_scans"] = group["ct_date"].nunique()
        rec["n_npz_files"] = group[group["has_npz"]]["ct_date"].nunique()
        rec["has_npz"] = group["has_npz"].any()

        # Gene labels (constant per patient)
        rec["gene_category"] = group["gene_category"].iloc[0]
        rec["gene_class_3"] = group["gene_class_3"].iloc[0]
        rec["gene_class_fine"] = group["gene_class_fine"].iloc[0]
        rec["genetic_mutation_raw"] = group["genetic_mutation"].iloc[0]
        rec["gene_tested"] = group["gene_tested"].iloc[0]

        # Surgery
        rec["had_surgery"] = group["had_surgery"].any()

        # has_any_abnormality (True if any anatomical code non-zero in either ear)
        rec["has_any_abnormality"] = group["has_any_abnormality"].any()

        # Syndrome flags
        for flag in ["charge_syndrome", "waardenburg", "digeorge", "goiter", "vacterl"]:
            rec[flag] = group[flag].any()

        # Underlying text (raw)
        underlying_vals = group["underlying"].dropna().unique()
        underlying_vals = [v for v in underlying_vals if v.lower() not in ("nil", "none")]
        rec["underlying_text"] = "; ".join(underlying_vals) if underlying_vals else None

        # ── Bilateral anatomical codes ──────────────────────────────────
        # For each structure, get R and L values from reference rows
        for struct in GRADED_STRUCTURES:
            r_val = r_rows[struct].iloc[0] if len(r_rows) > 0 else np.nan
            l_val = l_rows[struct].iloc[0] if len(l_rows) > 0 else np.nan
            rec[f"{struct}_R"] = r_val
            rec[f"{struct}_L"] = l_val

        # Other bilateral codes
        for extra in ["iem_flag", "hjb", "fn_anomaly"]:
            r_val = r_rows[extra].iloc[0] if len(r_rows) > 0 and extra in r_rows.columns else np.nan
            l_val = l_rows[extra].iloc[0] if len(l_rows) > 0 and extra in l_rows.columns else np.nan
            rec[f"{extra}_R"] = r_val
            rec[f"{extra}_L"] = l_val

        records.append(rec)

    patient_df = pd.DataFrame(records)
    return patient_df


def main():
    config.ensure_dirs()
    print("=" * 60)
    print("Patient table")
    print("=" * 60)

    # ── Load data ──────────────────────────────────────────────────────
    master_df = pd.read_parquet(config.RECORDS)
    print(f"\nLoaded records: {len(master_df)} rows, {master_df['patient_id'].nunique()} patients")

    # ── Build patient_df ───────────────────────────────────────────────
    print("\nBuilding patient-level DataFrame...")
    patient_df = build_patient_df(master_df)
    print(f"  {len(patient_df)} patients")
    print(f"  Sheets: {patient_df['sheet_source'].value_counts().to_dict()}")

    # ── Verify reference annotations constant per patient ──────────────
    print("\nVerifying reference annotations are constant per patient...")
    ref_cols = ["cochlea", "vestibule", "scc", "eva", "iac", "aperture", "cn",
                "iem_flag", "hjb", "fn_anomaly"]
    issues = 0
    for pid, group in master_df.groupby("patient_id"):
        for lat in ["R", "L"]:
            lat_rows = group[group["laterality"] == lat]
            if len(lat_rows) <= 1:
                continue
            for col in ref_cols:
                vals = lat_rows[col].dropna().unique()
                if len(vals) > 1:
                    issues += 1
                    if issues <= 5:
                        print(f"  WARNING: {pid} {lat} has varying {col}: {vals}")
    if issues == 0:
        print("  All reference annotations are constant per patient-laterality ✓")
    else:
        print(f"  {issues} inconsistencies found (may be due to pre/post-CI scans)")

    # ── Save ───────────────────────────────────────────────────────────
    patient_df.to_parquet(config.PATIENT_RECORDS, index=False)
    print(f"\nSaved {config.PATIENT_RECORDS} ({len(patient_df)} rows)")

    # ── Summary of has_any_abnormality ─────────────────────────────────
    print("\nVerifying has_any_abnormality column...")
    n_abn = patient_df["has_any_abnormality"].sum()
    n_norm_anatomy = len(patient_df) - n_abn
    print(f"  Patients with any abnormality: {n_abn}")
    print(f"  Patients with all-normal anatomy: {n_norm_anatomy}")

    # Cross-check: normal-sheet patients with abnormalities
    norm_patients = patient_df[patient_df["sheet_source"] == "normal"]
    norm_with_abn = norm_patients["has_any_abnormality"].sum()
    print(f"  Normal-sheet patients with abnormalities: {int(norm_with_abn)}/{len(norm_patients)}")

    # Cross-check: malformation-sheet patients with all-normal anatomy
    mal_patients_df = patient_df[patient_df["sheet_source"] == "malformation"]
    mal_all_normal = (~mal_patients_df["has_any_abnormality"]).sum()
    print(f"  Malformation-sheet patients with all-normal anatomy: {int(mal_all_normal)}/{len(mal_patients_df)}")

    # Verify gene_tested
    print(f"\n  gene_tested: {patient_df['gene_tested'].sum()}/{len(patient_df)} patients")
    print(f"  Not tested: {(~patient_df['gene_tested']).sum()} patients")

    return patient_df


if __name__ == "__main__":
    main()
