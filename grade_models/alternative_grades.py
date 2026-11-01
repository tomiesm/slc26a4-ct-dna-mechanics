"""Sensitivity of the grade model to the alternative grades of duplicate records.

Three patients had clearly different anatomy grades in their two merged records (the pairs in
DISCORDANT below).
The main analysis uses the grades of the retained record. This script refits the six-structure
grade model with the grades of the alias record instead, and with the three patients excluded,
using the same folds and the same nested logistic pipeline. The confidence intervals are
percentile bootstraps of the fixed held-out predictions.

Usage: python -m grade_models.alternative_grades
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from grade_models.analysis import (
    STRUCTS, dump_json, load_corrected_patient_data,
    nested_oof, onehot_block,
)

DISCORDANT = {"<PATIENT_ID_02>": "<PATIENT_ID_03>", "<PATIENT_ID_10>": "<PATIENT_ID_11>", "<PATIENT_ID_12>": "<PATIENT_ID_13>"}
ANATOMY_COLS = [f"{s}_{side}" for s in ("cochlea", "vestibule", "scc", "eva", "iac", "aperture", "cn")
                for side in ("R", "L")]
B_BOOT = 10000


def boot_ci(y, p, seed):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def anatomy_auc(d, y, folds, label, seed, groups):
    # Group-aware inner selection: cv_group_id keeps the possible family pair together within
    # every inner tuning split, as in the main nested pipeline.
    pred, _ = nested_oof(onehot_block(d, STRUCTS), y, folds, groups=groups)
    return {"label": label, "N": int(len(y)), "n_SLC26A4": int(y.sum()),
            "auroc": float(roc_auc_score(y, pred)), "auroc_percentile_ci": boot_ci(y, pred, seed)}, pred


def main():
    config.ensure_dirs()
    resolved, _ = load_corrected_patient_data()
    raw = pd.read_parquet(config.PATIENT_RECORDS).set_index("patient_id")
    fold_table = pd.read_csv(config.FOLDS)
    manifest = fold_table[["patient_id", "fold", "binary_label"]]

    # CV groups of the main nested pipeline, read from the fold table to avoid merge suffixes.
    grp_map = fold_table.set_index("patient_id")["cv_group_id"].astype(str)

    # retained record
    d_can = manifest.merge(resolved, on="patient_id", how="left", validate="one_to_one").set_index("patient_id")
    y = (d_can["binary_label"] == 0).astype(int).to_numpy()
    folds = d_can["fold"].astype(int).to_numpy()
    groups = grp_map.reindex(d_can.index).to_numpy()
    assert pd.Series(groups).notna().all(), "cv_group_id missing for a patient of the analysis cohort"
    assert pd.Series(groups).duplicated().any(), "expected at least one CV group with two patients"

    # alias record: swap the three patients' grades for those of the alias record
    alt = resolved.set_index("patient_id").copy()
    for canonical_id, alias_id in DISCORDANT.items():
        for col in ANATOMY_COLS:
            alt.loc[canonical_id, col] = raw.loc[alias_id, col]
    alt = alt.reset_index()
    d_alt = manifest.merge(alt, on="patient_id", how="left", validate="one_to_one").set_index("patient_id")

    # exclusion: drop the three patients
    keep = ~manifest["patient_id"].isin(DISCORDANT)
    m_excl = manifest[keep]
    d_excl = m_excl.merge(resolved, on="patient_id", how="left", validate="one_to_one").set_index("patient_id")
    y_excl = (d_excl["binary_label"] == 0).astype(int).to_numpy()
    folds_excl = d_excl["fold"].astype(int).to_numpy()
    groups_excl = grp_map.reindex(d_excl.index).to_numpy()

    # d_can and d_alt share the same patient_id index order, so `groups` aligns to both.
    res_can, _ = anatomy_auc(d_can, y, folds, "retained_record", 101, groups)
    res_alt, _ = anatomy_auc(d_alt, y, folds, "alias_record", 202, groups)
    res_excl, _ = anatomy_auc(d_excl, y_excl, folds_excl, "exclude_three_discordant", 303, groups_excl)

    out = {
        "group_aware_inner_selection": True,
        "cv_group_source": f"{config.FOLDS.name}:cv_group_id",
        "discordant_identities": DISCORDANT,
        "retained_record": res_can,
        "alias_record": res_alt,
        "exclude_discordant": res_excl,
        "delta_alias_minus_retained": round(res_alt["auroc"] - res_can["auroc"], 4),
        "description": (
            "AUROC of the six-structure grade model with the grades of the retained record, with the "
            "grades of the alias record, and with the three patients excluded"
        ),
    }
    dump_json(config.GRADE_DIR / "alternative_grades.json", out)
    print(f"retained-record AUROC     = {res_can['auroc']:.4f}  CI={res_can['auroc_percentile_ci']}")
    print(f"alias-record AUROC        = {res_alt['auroc']:.4f}  CI={res_alt['auroc_percentile_ci']}")
    print(f"three patients excluded   = {res_excl['auroc']:.4f}  (N={res_excl['N']})")
    print(f"delta (alias - retained)  = {out['delta_alias_minus_retained']:+.4f}")


if __name__ == "__main__":
    main()
