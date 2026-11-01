"""Sensitivity of the grade model to the encoding of the grades.

Checks whether the AUROC of the six-structure grade model depends on the bilateral one-hot
encoding of the grades. With the same group-aware nested folds and one-standard-error selection
of C, it compares four encodings:

  1. bilateral one-hot indicators (the encoding of the grade model);
  2. ordinal bilateral grades (right and left as numbers, 12 features);
  3. worst-ear grade per structure (6 features);
  4. worst-ear grade and absolute side difference per structure (12 features).

Reports AUROC with a percentile CI, log loss, Brier score, calibration-in-the-large and slope,
and the selected C of each fold.

Usage: python -m grade_models.encoding
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from grade_models.analysis import (
    STRUCTS, onehot_block, nested_oof, score_metrics, load_corrected_patient_data, derived_seed,
)

B_BOOT = 10000


def boot_ci(y, p, seed):
    rng = np.random.default_rng(seed)
    v = [roc_auc_score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B_BOOT))
         if np.unique(y[i]).size == 2]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    config.ensure_dirs()
    df, _ = load_corrected_patient_data()
    man = pd.read_csv(config.FOLDS)[["patient_id", "fold", "binary_label"]]
    d = man.merge(df, on="patient_id", how="left", validate="one_to_one").set_index("patient_id")
    y = (d["binary_label"] == 0).astype(int).to_numpy()
    folds = d["fold"].astype(int).to_numpy()
    groups = d["cv_group_id"].astype(str).to_numpy()

    def grade(side_struct):
        return d[side_struct].fillna(0).astype(float)

    onehot = onehot_block(d, STRUCTS)
    raw_ordinal = pd.DataFrame(
        {f"{s}_{side}": grade(f"{s}_{side}") for s in STRUCTS for side in ("R", "L")}, index=d.index)
    worst = pd.DataFrame(
        {s: np.maximum(grade(f"{s}_R"), grade(f"{s}_L")) for s in STRUCTS}, index=d.index)
    symmetry = pd.concat([
        pd.DataFrame({f"{s}_worst": np.maximum(grade(f"{s}_R"), grade(f"{s}_L")) for s in STRUCTS}, index=d.index),
        pd.DataFrame({f"{s}_sidediff": (grade(f"{s}_R") - grade(f"{s}_L")).abs() for s in STRUCTS}, index=d.index),
    ], axis=1)

    # The names also seed the bootstrap intervals.
    encodings = {"one_hot_bilateral_PRIMARY": onehot, "raw_ordinal_bilateral": raw_ordinal,
                 "worst_ear_per_structure": worst, "symmetry_aware_worst_plus_sidediff": symmetry}

    out = {"n": int(len(d)), "bootstrap_replicates": B_BOOT,
           "seed_scheme": "derived_seed (SHA-256 of the label plus SEED)",
           "encodings": {}}
    for name, X in encodings.items():
        pred, tuning = nested_oof(X, y, folds, groups=groups)
        m = score_metrics(y, pred, name)
        out["encodings"][name] = {
            "n_features": int(X.shape[1]),
            "auroc": m["auroc"], "auroc_percentile_ci": boot_ci(y, pred, derived_seed(f"encoding::{name}")),
            "log_loss": m["log_loss"], "brier": m["brier"],
            "calibration_in_the_large": m["calibration_in_the_large"], "calibration_slope": m["calibration_slope"],
            "selected_C_by_fold": {k: v["C"] for k, v in tuning.items()},
        }
    with open(config.GRADE_DIR / "encoding.json", "w") as handle:
        json.dump(out, handle, indent=2)

    print(f"N={len(d)}")
    for name, e in out["encodings"].items():
        print(f"  {name:<38} feat={e['n_features']:>2}  AUROC={e['auroc']:.4f} "
              f"[{e['auroc_percentile_ci'][0]:.3f},{e['auroc_percentile_ci'][1]:.3f}]  "
              f"logloss={e['log_loss']:.3f}  cal-large={e['calibration_in_the_large']:+.3f} slope={e['calibration_slope']:.2f}")


if __name__ == "__main__":
    main()
