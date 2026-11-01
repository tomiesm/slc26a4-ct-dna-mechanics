"""Bootstrap the change in leave-genotype-pair-out prediction when DNA descriptors are added.

variants/analysis.py predicts vestibular enlargement in the variant subset (N=35, 20 genotype pairs)
with one genotype pair held out at a time. This script compares two baselines with and without the
cgNA+ descriptors:

  * simple sequence/consequence baseline vs simple + cgNA;
  * variant identity baseline vs variant identity + cgNA.

It reports the observed AUC, log-loss and Brier differences (cgNA model minus baseline) with 95%
percentile intervals from a bootstrap that resamples genotype pairs.

Usage: python -m variants.descriptor_prediction
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

import config

B = 20000
EPS = 1e-6
SEED = 20260716


def metrics(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan
    return auc, log_loss(y, p, labels=[0, 1]), brier_score_loss(y, p)


def main():
    config.ensure_dirs()
    t = pd.read_csv(config.DESCRIPTOR_PREDICTIONS)
    d = t[t.scheme == "leave_genotype_pair_out"].reset_index(drop=True)
    y = d.vestibule_enlarged.to_numpy()
    pairs = d.genotype_pair.to_numpy()
    uniq = sorted(set(pairs))
    pair_idx = {u: np.flatnonzero(pairs == u) for u in uniq}

    comparisons = {
        "simple_plus_cgna_minus_simple": ("p_simple", "p_simple_cgna"),
        "identity_plus_cgna_minus_identity": ("p_identity", "p_identity_cgna"),
    }

    def deltas(idx):
        yy = y[idx]
        res = {}
        for name, (base, aug) in comparisons.items():
            ab, lb, bb = metrics(yy, d[base].to_numpy()[idx])
            aa, la, ba = metrics(yy, d[aug].to_numpy()[idx])
            res[name] = {"d_auc": aa - ab, "d_logloss": la - lb, "d_brier": ba - bb}
        return res

    obs = deltas(np.arange(len(d)))

    rng = np.random.default_rng(SEED)
    boot = {name: {k: [] for k in ("d_auc", "d_logloss", "d_brier")} for name in comparisons}
    n_valid = 0
    for _ in range(B):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)     # resample genotype-pair clusters
        idx = np.concatenate([pair_idx[u] for u in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        n_valid += 1
        dd = deltas(idx)
        for name in comparisons:
            for k in ("d_auc", "d_logloss", "d_brier"):
                v = dd[name][k]
                if np.isfinite(v):
                    boot[name][k].append(v)

    out = {"n": int(len(d)), "n_genotype_pair_clusters": len(uniq),
           "endpoint": "vestibular enlargement", "scheme": "leave_genotype_pair_out",
           "bootstrap": "genotype-pair-clustered percentile (B=%d)" % B,
           "seed": SEED, "bootstrap_replicates": B, "valid_replicates": int(n_valid),
           "invalid_single_class_replicates": int(B - n_valid), "results": {}}
    for name in comparisons:
        out["results"][name] = {}
        for k in ("d_auc", "d_logloss", "d_brier"):
            arr = np.asarray(boot[name][k])
            out["results"][name][k] = {
                "observed": float(obs[name][k]),
                "ci95": [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))],
                "boot_median": float(np.median(arr))}
    with open(config.VARIANT_DIR / "descriptor_prediction.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"N={len(d)}, {len(uniq)} genotype-pair clusters")
    for name in comparisons:
        r = out["results"][name]
        print(f"  {name}:")
        for k in ("d_auc", "d_logloss", "d_brier"):
            e = r[k]
            print(f"     {k:<10} obs={e['observed']:+.4f}  95% CI [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
