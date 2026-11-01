"""Check how stable the ranking of the DNA descriptors is when one genotype pair is left out.

Ranks the eligible cgNA+ descriptors by absolute Spearman correlation with the worst-ear vestibule
grade, then leaves out one genotype pair at a time and ranks them again. Stability is summarized by
the Spearman correlation between the reduced and the full ranking, the Jaccard overlap of the top 10
descriptors, and how often the top descriptor changes.

Usage: python -m variants.descriptor_ranking
"""
import json
import numpy as np
import pandas as pd
from scipy import stats

import config

TOPK = 10


def main():
    config.ensure_dirs()
    j = pd.read_csv(config.VARIANT_TABLE)
    endpoint = j["vestibule_worst"].astype(float)
    have = endpoint.notna()
    j, endpoint = j[have].reset_index(drop=True), endpoint[have].reset_index(drop=True)
    cg_cols = [c for c in j.columns if c.startswith("cgna_")]
    # eligible descriptors: complete over the analyzed patients and with at least three distinct values
    eligible = [c for c in cg_cols if j[c].notna().all() and j[c].nunique() > 2]
    y = endpoint.to_numpy()
    pairs = j["genotype_pair"].to_numpy()
    uniq = sorted(set(pairs))

    def ranking(idx):
        yy = y[idx]
        scores = {}
        for c in eligible:
            v = j[c].to_numpy()[idx]
            if np.unique(v).size < 3 or np.unique(yy).size < 2:
                scores[c] = 0.0
            else:
                rho, _ = stats.spearmanr(v, yy)
                scores[c] = abs(rho) if np.isfinite(rho) else 0.0
        order = sorted(eligible, key=lambda c: -scores[c])
        return order, scores

    full_order, _ = ranking(np.arange(len(j)))
    full_rankpos = {c: i for i, c in enumerate(full_order)}
    full_top = set(full_order[:TOPK])
    top1_full = full_order[0]

    spearman_stab, jaccard, top1_changes = [], [], 0
    per_pair = {}
    for u in uniq:
        idx = np.flatnonzero(pairs != u)
        order, _ = ranking(idx)
        pos = {c: i for i, c in enumerate(order)}
        common = list(full_rankpos)
        rho, _ = stats.spearmanr([full_rankpos[c] for c in common], [pos[c] for c in common])
        spearman_stab.append(float(rho))
        jac = len(full_top & set(order[:TOPK])) / len(full_top | set(order[:TOPK]))
        jaccard.append(float(jac))
        changed = order[0] != top1_full
        top1_changes += int(changed)
        per_pair[u] = {"spearman_rank_corr": float(rho), "top10_jaccard": float(jac), "top1": order[0]}

    out = {
        "n_patients": int(len(j)), "n_eligible_descriptors": len(eligible),
        "n_genotype_pair_perturbations": len(uniq),
        "endpoint": "worst-ear vestibule grade",
        "full_ranking_top5": full_order[:5], "full_top1": top1_full,
        "rank_stability": {
            "spearman_rank_corr_median": float(np.median(spearman_stab)),
            "spearman_rank_corr_min": float(np.min(spearman_stab)),
            "top10_jaccard_median": float(np.median(jaccard)),
            "top10_jaccard_min": float(np.min(jaccard)),
            "top1_descriptor_changed_in_perturbations": int(top1_changes),
            "top1_descriptor_change_fraction": float(top1_changes / len(uniq)),
        },
        "per_pair": per_pair,
    }
    with open(config.VARIANT_DIR / "descriptor_ranking.json", "w") as fh:
        json.dump(out, fh, indent=2)
    r = out["rank_stability"]
    print(f"N={len(j)} patients, {len(eligible)} eligible descriptors, {len(uniq)} genotype-pair perturbations")
    print(f"  full top-1 descriptor: {top1_full}")
    print(f"  Spearman rank-corr: median {r['spearman_rank_corr_median']:.3f}  min {r['spearman_rank_corr_min']:.3f}")
    print(f"  top-10 Jaccard: median {r['top10_jaccard_median']:.3f}  min {r['top10_jaccard_min']:.3f}")
    print(f"  top-1 descriptor changed in {r['top1_descriptor_changed_in_perturbations']}/{len(uniq)} perturbations")


if __name__ == "__main__":
    main()
