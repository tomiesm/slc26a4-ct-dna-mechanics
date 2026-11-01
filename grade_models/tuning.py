"""Sensitivity of the grade model to the penalty-selection rule.

The grade model selects the logistic penalty by a one-standard-error rule. This script measures
how much that choice matters, with the same
group-aware outer folds and the six-structure one-hot encoding:

  A. one-standard-error AUC selection on the standard grid (the rule of the grade model);
  B. maximum inner mean AUC on the standard grid;
  C. one-standard-error AUC selection on a grid extended to lower C (adds 1e-5 to 1e-3);
  D. minimum inner mean log loss on the extended grid.

AUROC is the main criterion; calibration and log loss are also reported. The rule of the grade
model is not chosen by comparing these results.

Usage: python -m grade_models.tuning
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

import config
from grade_models.analysis import (
    C_GRID, STRUCTS, onehot_block, make_pipeline, score_metrics, load_corrected_patient_data, SEED,
)

EXTENDED_GRID = (1e-5, 1e-4, 1e-3) + C_GRID


def inner_splits(X, y, groups):
    if groups is not None and len(np.unique(groups)) < len(groups):
        return list(StratifiedGroupKFold(3, shuffle=True, random_state=SEED).split(X, y, groups))
    return list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y))


def select_C(X, y, groups, grid, rule):
    splits = inner_splits(X, y, groups)
    auc, ll = {}, {}
    for c in grid:
        fa, fl = [], []
        for tr, va in splits:
            m = make_pipeline(c); m.fit(X[tr], y[tr])
            p = m.predict_proba(X[va])[:, 1]
            fa.append(roc_auc_score(y[va], p)); fl.append(log_loss(y[va], p, labels=[0, 1]))
        auc[c] = (float(np.mean(fa)), float(np.std(fa, ddof=1) / np.sqrt(len(fa))))
        ll[c] = float(np.mean(fl))
    if rule == "one_se_auc":
        best = max(grid, key=lambda c: auc[c][0]); thr = auc[best][0] - auc[best][1]
        return next(c for c in grid if auc[c][0] >= thr - 1e-12)
    if rule == "literal_max_auc":
        return max(grid, key=lambda c: auc[c][0])
    if rule == "log_loss":
        return min(grid, key=lambda c: ll[c])
    raise ValueError(rule)


def nested(X, y, folds, groups, grid, rule):
    pred = np.full(len(y), np.nan); chosen = {}
    for f in sorted(np.unique(folds)):
        te = folds == f; tr = ~te
        c = select_C(X[tr], y[tr], None if groups is None else groups[tr], grid, rule)
        m = make_pipeline(c); m.fit(X[tr], y[tr]); pred[te] = m.predict_proba(X[te])[:, 1]
        chosen[str(int(f))] = float(c)
    return pred, chosen


def main():
    config.ensure_dirs()
    df, _ = load_corrected_patient_data()
    man = pd.read_csv(config.FOLDS)[["patient_id", "fold", "binary_label"]]
    d = man.merge(df, on="patient_id", how="left", validate="one_to_one").set_index("patient_id")
    y = (d["binary_label"] == 0).astype(int).to_numpy()
    folds = d["fold"].astype(int).to_numpy()
    groups = d["cv_group_id"].astype(str).to_numpy()
    X = onehot_block(d, STRUCTS).to_numpy()

    variants = {
        "A_one_se_auc_primary_grid": (C_GRID, "one_se_auc"),
        "B_literal_max_auc_primary_grid": (C_GRID, "literal_max_auc"),
        "C_one_se_auc_extended_grid": (EXTENDED_GRID, "one_se_auc"),
        "D_log_loss_extended_grid": (EXTENDED_GRID, "log_loss"),
    }
    out = {"n": int(len(d)), "variants": {}}
    for name, (grid, rule) in variants.items():
        pred, chosen = nested(X, y, folds, groups, grid, rule)
        m = score_metrics(y, pred, name)
        out["variants"][name] = {
            "auroc": m["auroc"], "log_loss": m["log_loss"], "brier": m["brier"],
            "calibration_in_the_large": m["calibration_in_the_large"], "calibration_slope": m["calibration_slope"],
            "selected_C_by_fold": chosen}
    with open(config.GRADE_DIR / "tuning.json", "w") as handle:
        json.dump(out, handle, indent=2)
    print(f"N={len(d)}")
    for name, v in out["variants"].items():
        cs = sorted(set(v["selected_C_by_fold"].values()))
        print(f"  {name:<34} AUROC={v['auroc']:.4f}  logloss={v['log_loss']:.3f}  "
              f"cal-large={v['calibration_in_the_large']:+.3f} slope={v['calibration_slope']:.2f}  C={cs}")


if __name__ == "__main__":
    main()
