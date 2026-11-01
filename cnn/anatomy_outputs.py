"""Anatomy outputs of the multitask CNN, and SLC26A4 status predicted from them.

Runs inference with the five multitask checkpoints of cnn.train to obtain out-of-fold categorical
anatomy probabilities for all patients of the analysis cohort (each patient is scored only by the
model that did not train on it). Then:

  * anatomy prediction quality per structure: class support; per-class and macro one-vs-rest AUC;
    quadratic-weighted Cohen kappa (argmax versus true worst-ear grade); AUC for abnormal anatomy
    (P(grade > 0)) with a patient bootstrap CI; mean calibration bias of the abnormality probability.
  * CT -> predicted anatomy -> recorded SLC26A4 status: a group-aware nested logistic regression on
    the fixed out-of-fold anatomy probabilities, compared with the multitask genotype output, the
    expert grades and a combination of the two CNN outputs.

The second part uses the saved out-of-fold anatomy probabilities and is descriptive. In the
output, "bottleneck" is the anatomy-path score (SLC26A4 status predicted from the anatomy
probabilities alone) and "direct" is the genotype head of the multitask CNN.

Usage: python -m cnn.anatomy_outputs
"""
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, cohen_kappa_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from cnn import train
from grade_models.analysis import nested_oof, derived_seed
from cnn.added_value import _delong_pair

STRUCTS = train.STRUCTS
NCLASS = train.STRUCT_NCLASS
B_BOOT = 10000


def infer_oof_anatomy():
    """OOF full categorical anatomy probs (mean over central slices) + genotype prob, per patient."""
    geno, anat_true, fold_ids = train.load_labels()
    device = torch.device("cuda:0")
    aug_ev = train.SliceAugmentation(training=False).to(device)
    rows = {}
    for f in range(config.N_FOLDS):
        model = train.SliceResNet("multitask").to(device)
        model.load_state_dict(torch.load(config.CNN_CHECKPOINT_DIR / f"multitask_fold{f}.pt",
                                         map_location=device, weights_only=True))
        model.eval()
        test_ids = fold_ids[f]
        train.CentralSliceLevelDataset.prepare_cache(test_ids)
        acc = {p: {s: [] for s in STRUCTS} for p in test_ids}
        gacc = {p: [] for p in test_ids}
        with torch.no_grad():
            for imgs, glb, pids, _ in train.make_loader(test_ids, geno, train=False):
                out = model(train.normalize(aug_ev(imgs.to(device)), device))
                for s in STRUCTS:
                    ps = torch.softmax(out[s].float(), 1).cpu().numpy()
                    for i, p in enumerate(pids):
                        acc[p][s].append(ps[i])
                pg = torch.softmax(out["geno"].float(), 1).cpu().numpy()
                for i, p in enumerate(pids):
                    gacc[p].append(float(pg[i, 0]))       # P(SLC26A4)
        for p in test_ids:
            rows[p] = {"fold": f, "geno_mean_prob": float(np.mean(gacc[p]))}
            for s in STRUCTS:
                mean_prob = np.mean(acc[p][s], axis=0)     # categorical prob vector
                for c in range(NCLASS[s]):
                    rows[p][f"{s}_p{c}"] = float(mean_prob[c])
        del model; torch.cuda.empty_cache()
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("patient_id").reset_index()


def bootstrap_auc_ci(y, p, seed):
    """Percentile 95% CI of the AUC over patient bootstrap resamples."""
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size == 2:
            v.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    config.ensure_dirs()
    man = pd.read_csv(config.FOLDS)
    res = pd.read_parquet(config.PEOPLE).set_index("patient_id")
    pred = infer_oof_anatomy()
    pred.to_csv(config.CNN_DIR / "anatomy_predictions.csv", index=False)
    pred_folds = pred.set_index("patient_id")["fold"].reindex(man.patient_id).to_numpy()
    assert (pred_folds == man.fold.to_numpy()).all(), "fold of the anatomy predictions differs from config.FOLDS"
    d = man.merge(pred.drop(columns="fold"), on="patient_id", how="left").set_index("patient_id")
    y = (d.binary_label == 0).astype(int).to_numpy()                 # SLC26A4 positive
    folds = d.fold.to_numpy()
    groups = man.set_index("patient_id").loc[d.index, "cv_group_id"].astype(str).to_numpy()

    # ---- anatomy prediction quality per structure ----
    anat_metrics = {}
    for s in STRUCTS:
        true = res.loc[d.index, [f"{s}_R", f"{s}_L"]].max(axis=1).fillna(0).round().astype(int)
        true = true.clip(0, NCLASS[s] - 1).to_numpy()
        P = d[[f"{s}_p{c}" for c in range(NCLASS[s])]].to_numpy()
        pred_grade = P.argmax(1)
        support = {int(g): int((true == g).sum()) for g in range(NCLASS[s])}
        per_class = {}
        for c in range(NCLASS[s]):
            yc = (true == c).astype(int)
            per_class[c] = float(roc_auc_score(yc, P[:, c])) if 0 < yc.sum() < len(yc) else None
        estimable = [v for v in per_class.values() if v is not None]
        macro = float(np.mean(estimable)) if estimable else None
        kappa = (float(cohen_kappa_score(true, pred_grade, weights="quadratic"))
                 if len(np.unique(true)) > 1 else None)
        pab = 1.0 - P[:, 0]
        yab = (true > 0).astype(int)
        bin_auc = float(roc_auc_score(yab, pab)) if 0 < yab.sum() < len(yab) else None
        bin_ci = bootstrap_auc_ci(yab, pab, derived_seed(f"anat::{s}")) if bin_auc is not None else None
        # mean(predicted abnormality prob) - mean(observed abnormal) is a mean calibration bias,
        # not the logit-intercept calibration-in-the-large reported for the grade models.
        mean_cal_bias = float(np.mean(pab) - np.mean(yab))
        anat_metrics[s] = {"support": support, "per_class_ovr_auc": per_class, "macro_ovr_auc": macro,
                           "quadratic_weighted_kappa": kappa, "binary_abnormal_auc": bin_auc,
                           "binary_abnormal_auc_ci": bin_ci, "mean_calibration_bias": mean_cal_bias}

    # ---- CT -> predicted anatomy -> genotype (fixed-score decomposition) ----
    cat_cols = [f"{s}_p{c}" for s in STRUCTS for c in range(NCLASS[s])]     # 22-dim soft one-hot
    Xcat = d[cat_cols].to_numpy()
    # expected grade + abnormal probability per structure (12 interpretable features)
    interp = np.column_stack(
        [d[[f"{s}_p{c}" for c in range(NCLASS[s])]].to_numpy() @ np.arange(NCLASS[s]) for s in STRUCTS]
        + [1.0 - d[f"{s}_p0"].to_numpy() for s in STRUCTS])

    bottleneck_full, _ = nested_oof(Xcat, y, folds, groups=groups)
    bottleneck_interp, _ = nested_oof(interp, y, folds, groups=groups)
    auc_bottleneck_full = float(roc_auc_score(y, bottleneck_full))
    auc_bottleneck_interp = float(roc_auc_score(y, bottleneck_interp))

    multitask_genotype_score = (pd.read_csv(config.cnn_predictions("multitask")).set_index("patient_id")
                                .loc[d.index, "p_ent_SLC26A4"].to_numpy())
    coded = pd.read_csv(config.GRADE_PREDICTIONS).set_index("patient_id").loc[d.index, "p_4_anatomy_PRIMARY"].to_numpy()
    auc_genotype_head = float(roc_auc_score(y, multitask_genotype_score))
    auc_coded = float(roc_auc_score(y, coded))
    # fixed-score genotype head + bottleneck combination (nested logistic over the two fixed OOF scores)
    def _logit(p): return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    combo_feat = np.column_stack([_logit(multitask_genotype_score), _logit(bottleneck_full)])
    combo, _ = nested_oof(combo_feat, y, folds, groups=groups)
    auc_combo = float(roc_auc_score(y, combo))

    # Paired comparison of the predicted-anatomy model with the genotype head of the same network.
    dl_bottleneck_vs_head = _delong_pair(y, bottleneck_full, multitask_genotype_score)

    def _paired_delta_ci(a, b, seed, B=10000):
        rng = np.random.default_rng(seed)
        deltas = []
        for _ in range(B):
            idx = rng.integers(0, len(y), len(y))
            if np.unique(y[idx]).size == 2:
                deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
        return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]

    bottleneck_vs_head_delta_ci = _paired_delta_ci(bottleneck_full, multitask_genotype_score,
                                                   derived_seed("bottleneck_minus_direct"))
    # The genotype-head score above is the entropy-weighted held-out score of cnn.train. The inference
    # in infer_oof_anatomy also gives a mean-aggregated genotype probability; its AUC is reported as well.
    auc_genotype_head_mean_agg = float(roc_auc_score(y, d["geno_mean_prob"].to_numpy()))

    out = {
        "n": int(len(d)),
        "bootstrap_replicates": B_BOOT,
        "seed_scheme": "derived_seed (SHA-256 of the label plus SEED)",
        "design": (f"OOF full categorical anatomy inference from the {config.N_FOLDS} multitask "
                   "checkpoints; the bottleneck is a descriptive score over the fixed OOF "
                   "predicted-anatomy features"),
        "anatomy_reading": anat_metrics,
        "bottleneck": {
            "direct_multitask_genotype_head_auc": auc_genotype_head,
            "direct_head_aggregation": "entropy-weighted patient aggregation (as in cnn.train)",
            "reinference_mean_aggregated_genotype_auc": auc_genotype_head_mean_agg,
            "predicted_anatomy_bottleneck_full_categorical_auc": auc_bottleneck_full,
            "predicted_anatomy_bottleneck_interpretable12_auc": auc_bottleneck_interp,
            "expert_structured_grades_auc": auc_coded,
            "fixed_score_direct_plus_bottleneck_auc": auc_combo,
            "bottleneck_minus_direct_delta": float(auc_bottleneck_full - auc_genotype_head),
            "bottleneck_minus_direct_delong_p_two_sided": float(dl_bottleneck_vs_head["p_two_sided"]),
            "bottleneck_minus_direct_paired_bootstrap_ci": bottleneck_vs_head_delta_ci,
        },
    }
    with (config.CNN_DIR / "anatomy_outputs.json").open("w") as fh:
        json.dump(out, fh, indent=2)

    build_figure(out)
    print(f"N={len(d)}")
    print("anatomy binary-abnormal AUC:", {s: round(anat_metrics[s]["binary_abnormal_auc"], 3) for s in STRUCTS})
    print("macro OvR AUC:", {s: (round(anat_metrics[s]["macro_ovr_auc"], 3) if anat_metrics[s]["macro_ovr_auc"] else None) for s in STRUCTS})
    print(f"genotype head {auc_genotype_head:.3f} | bottleneck(full) {auc_bottleneck_full:.3f} | "
          f"bottleneck(interp) {auc_bottleneck_interp:.3f} | genotype head+bottleneck {auc_combo:.3f} | "
          f"expert {auc_coded:.3f}")


def build_figure(result):
    """Draw the figure from the saved estimates (no inference, fitting or resampling)."""
    anat_metrics = result["anatomy_reading"]
    b = result["bottleneck"]
    auc_genotype_head = b["direct_multitask_genotype_head_auc"]
    auc_bottleneck_full = b["predicted_anatomy_bottleneck_full_categorical_auc"]
    auc_combo = b["fixed_score_direct_plus_bottleneck_auc"]
    auc_coded = b["expert_structured_grades_auc"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    # A: AUC for abnormal anatomy per structure
    labs = [s for s in STRUCTS]
    aucs = [anat_metrics[s]["binary_abnormal_auc"] for s in STRUCTS]
    kaps = [anat_metrics[s]["quadratic_weighted_kappa"] for s in STRUCTS]
    yy = np.arange(len(STRUCTS))
    axes[0].barh(yy, aucs, color="#2980b9", edgecolor="black")
    for i, s in enumerate(STRUCTS):
        ci = anat_metrics[s]["binary_abnormal_auc_ci"]
        if ci:
            axes[0].plot(ci, [i, i], color="black", lw=1)
        # kappa label inside the visible x-range (the axis starts at 0.4)
        axes[0].text(0.415, i, f"$\\kappa$ {kaps[i]:.2f}" if kaps[i] is not None else "",
                     va="center", ha="left", fontsize=8, color="white", fontweight="bold")
    axes[0].set_yticks(yy); axes[0].set_yticklabels(labs); axes[0].invert_yaxis()
    axes[0].axvline(0.5, color="grey", ls=":"); axes[0].set_xlim(0.4, 1.0)
    axes[0].set_xlabel("AUC for normal versus abnormal anatomy (95% CI)")
    axes[0].set_title("A  Prediction of anatomical abnormalities")
    # B: gene-status AUROC of the CNN outputs and the expert grades
    names = ["CNN gene-status\noutput", "CNN anatomy\noutputs", "Combined CNN\noutputs", "Expert CT\ngrades"]
    vals = [auc_genotype_head, auc_bottleneck_full, auc_combo, auc_coded]
    cols = ["#e67e22", "#8e44ad", "#7f8c8d", "#c0392b"]
    axes[1].bar(range(4), vals, color=cols, edgecolor="black")
    for i, v in enumerate(vals):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    axes[1].set_xticks(range(4)); axes[1].set_xticklabels(names, fontsize=7.5, rotation=12, ha="center")
    axes[1].tick_params(axis="x", pad=2)
    axes[1].axhline(0.5, color="grey", ls=":"); axes[1].set_ylim(0.4, 1.02)
    axes[1].set_ylabel("AUROC for recorded SLC26A4 status")
    axes[1].set_title("B  Gene-status prediction from CNN outputs or CT grades")
    fig.suptitle("Anatomy and gene-status predictions from the multitask CNN\n"
                 "Analysis of fixed out-of-fold predictions",
                 fontsize=10.5, y=1.02)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(config.CNN_DIR / "anatomy_outputs.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
