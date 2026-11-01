"""Repeat the CNN added-value comparison for each of the three CNN seed runs.

cnn.added_value uses seed run 1. This script repeats its comparison of the score-space stack
(defined in cnn.added_value) with the grade-model score (paired DeLong test) for the three seed
runs of cnn.train and cnn.seed_runs and both CNN modes. The stacks are fitted on the saved
out-of-fold scores with the same folds and are descriptive.

Usage: python -m cnn.added_value_seeds
"""
import json

import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from cnn.added_value import _conditional_meta_map, _delong_pair, load_cnn_predictions


def main():
    config.ensure_dirs()
    coded = pd.read_csv(config.GRADE_PREDICTIONS).set_index("patient_id")
    y = coded["y_SLC26A4"].to_numpy()
    folds = coded["fold"].to_numpy()
    p_coded = coded["p_4_anatomy_PRIMARY"].to_numpy()
    auc_coded = float(roc_auc_score(y, p_coded))

    results = {"auc_coded": auc_coded, "by_seed": {}}
    all_above = True
    for seed_run in (1, 2, 3):
        results["by_seed"][seed_run] = {}
        for mode in ("single", "multitask"):
            c = load_cnn_predictions(mode, seed_run).loc[coded.index]
            assert (c["y_SLC26A4"].to_numpy() == y).all() and (c["fold"].to_numpy() == folds).all()
            p_cnn = c["p_ent_SLC26A4"].to_numpy()
            full_oof, _reduced, _coef = _conditional_meta_map(y, folds, p_coded, p_cnn)
            dl = _delong_pair(y, full_oof, p_coded)          # stack vs exact coded
            delta = float(dl["delta"]); p = dl["p_two_sided"]
            above = p > 0.05
            all_above = all_above and above
            results["by_seed"][seed_run][mode] = {
                "cnn_auc": float(roc_auc_score(y, p_cnn)),
                "stack_auc": float(roc_auc_score(y, full_oof)),
                "stack_minus_coded_delta": delta,
                "delong_p_two_sided": p, "delong_p_above_0_05": bool(above)}
    results["all_delong_p_above_0_05"] = bool(all_above)
    with open(config.CNN_DIR / "added_value_seeds.json", "w") as fh:
        json.dump(results, fh, indent=2)

    print(f"coded AUC {auc_coded:.4f}")
    for seed_run in (1, 2, 3):
        for mode in ("single", "multitask"):
            r = results["by_seed"][seed_run][mode]
            print(f"  seed run {seed_run} {mode:<10}: CNN {r['cnn_auc']:.3f}  stack {r['stack_auc']:.4f}  "
                  f"delta {r['stack_minus_coded_delta']:+.4f}  DeLong p={r['delong_p_two_sided']:.3f}")
    print("all DeLong p > 0.05:", all_above)


if __name__ == "__main__":
    main()
