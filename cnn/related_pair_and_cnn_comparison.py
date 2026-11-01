"""Check the grade-model AUROC without the possibly related pair, and compare the two CNNs pairwise.

Both analyses use the fixed saved out-of-fold scores; no model is retrained and no training
uncertainty is propagated.

(A) Possibly related pair. The only cross-validation group with more than one patient holds two
    patients who may be related (<PATIENT_ID_19> and <PATIENT_ID_20>). Keeping them in one group
    protects model training and inner selection but does not make the patient-level bootstrap or
    DeLong inference independent of relatedness. The grade-model AUROC is therefore recomputed
    leaving out each member in turn and both together.

(B) Single-task versus multitask CNN. For each of the three whole-pipeline seed runs, the two CNN
    scores on the same patients and folds are compared by a paired DeLong test and a paired
    patient bootstrap.

Usage: python -m cnn.related_pair_and_cnn_comparison
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from cnn.added_value import _delong_pair, load_cnn_predictions
from grade_models.analysis import derived_seed

B_BOOT = 10000


def auc(y, p):
    return float(roc_auc_score(y, p))


def paired_delta_ci(y, a, b, seed):
    """Paired patient-bootstrap 95% CI of AUROC(a) - AUROC(b) on identical resamples."""
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            ds.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]


def main():
    config.ensure_dirs()
    man = pd.read_csv(config.FOLDS)
    coded = pd.read_csv(config.GRADE_PREDICTIONS).set_index("patient_id")
    y = coded["y_SLC26A4"].to_numpy()
    p_coded = coded["p_4_anatomy_PRIMARY"].to_numpy()
    folds = coded["fold"].to_numpy()
    ids = coded.index.to_numpy()

    # ---- (A) leave-one-member-out sensitivity for the possibly related pair ----
    grp = man.set_index("patient_id")["cv_group_id"].astype(str)
    vc = grp.value_counts()
    nonsingleton = vc[vc > 1].index.tolist()
    fam_members = sorted(grp[grp.isin(nonsingleton)].index.tolist())
    lab = man.set_index("patient_id").loc[fam_members, "binary_label"].to_dict()
    famA = {
        "non_singleton_groups": nonsingleton,
        "members": fam_members,
        "member_labels": {k: ("SLC26A4" if v == 0 else "comparator") for k, v in lab.items()},
        "primary_auroc_full": auc(y, p_coded),
        "leave_one_out": {},
    }
    for mid in fam_members:
        keep = ids != mid
        famA["leave_one_out"][mid] = {"n": int(keep.sum()),
                                      "n_SLC26A4": int(y[keep].sum()),
                                      "auroc": auc(y[keep], p_coded[keep])}
    keep = ~np.isin(ids, fam_members)
    famA["leave_all_members_out"] = {"n": int(keep.sum()),
                                     "n_SLC26A4": int(y[keep].sum()),
                                     "auroc": auc(y[keep], p_coded[keep])}

    # ---- (B) paired single-task vs multitask across seed runs ----
    pairB = {
        "design": (f"paired DeLong + paired patient bootstrap on the same {len(ids)} patients/folds; "
                   "delta = multitask minus single-task; fixed saved OOF scores"),
        "by_seed": {},
    }
    deltas = []
    for seed_run in (1, 2, 3):
        s = load_cnn_predictions("single", seed_run).loc[ids]
        m = load_cnn_predictions("multitask", seed_run).loc[ids]
        assert (s["y_SLC26A4"].to_numpy() == y).all() and (m["y_SLC26A4"].to_numpy() == y).all()
        assert (s["fold"].to_numpy() == folds).all() and (m["fold"].to_numpy() == folds).all()
        ps = s["p_ent_SLC26A4"].to_numpy()
        pm = m["p_ent_SLC26A4"].to_numpy()
        dl = _delong_pair(y, pm, ps)                     # multitask - single
        ci = paired_delta_ci(y, pm, ps, derived_seed(f"multi_minus_single_seed{seed_run}"))
        d = auc(y, pm) - auc(y, ps)
        deltas.append(d)
        pairB["by_seed"][str(seed_run)] = {
            "single_auroc": auc(y, ps), "multitask_auroc": auc(y, pm),
            "multitask_minus_single_delta": float(d),
            "delong_p_two_sided": float(dl["p_two_sided"]),
            "paired_bootstrap_ci": ci,
        }
    signs = {int(np.sign(x)) for x in deltas}
    pairB["summary"] = {
        "mean_delta": float(np.mean(deltas)),
        "delta_range": [float(min(deltas)), float(max(deltas))],
        "sign_changes_across_seeds": bool(len(signs) > 1),
        "all_delong_p_above_0_05": bool(all(pairB["by_seed"][str(r)]["delong_p_two_sided"] > 0.05
                                            for r in (1, 2, 3))),
    }

    out = {"n": int(len(ids)), "bootstrap_replicates": B_BOOT,
           "seed_scheme": "derived_seed (SHA-256 of the label plus SEED)",
           "family_sensitivity": famA, "paired_single_vs_multitask": pairB}
    with open(config.CNN_DIR / "related_pair_and_cnn_comparison.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print("related pair members:", fam_members, famA["member_labels"])
    print("related pair: full %.4f | leave one out %s | both out %.4f"
          % (famA["primary_auroc_full"],
             {k: round(v["auroc"], 4) for k, v in famA["leave_one_out"].items()},
             famA["leave_all_members_out"]["auroc"]))
    for seed_run in (1, 2, 3):
        b = pairB["by_seed"][str(seed_run)]
        print(f"seed run {seed_run}: single {b['single_auroc']:.3f} multi {b['multitask_auroc']:.3f} "
              f"delta {b['multitask_minus_single_delta']:+.3f} DeLong p={b['delong_p_two_sided']:.3f} "
              f"CI [{b['paired_bootstrap_ci'][0]:+.3f}, {b['paired_bootstrap_ci'][1]:+.3f}]")
    print("paired summary:", pairB["summary"])


if __name__ == "__main__":
    main()
