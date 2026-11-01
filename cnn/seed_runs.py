"""Repeat cnn.train with two further seeds.

The run of cnn.train is seed run 1 (seed_inner = CNN_SEED + fold, seed_refit = CNN_SEED + fold + 100).
Seed runs 2 and 3 call cnn.train.run_job for both networks and all folds with both seeds shifted by
1000 times the run number, and write only the held-out fold predictions. The summary gives the
pooled genotype AUROC of each of the three runs and their spread. Run after cnn.train has finished;
its files are not changed.

Usage: python -m cnn.seed_runs
"""
import time, json
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import config
from cnn import train

SEED_RUNS = [2, 3]                           # seed run 1 is the run of cnn.train


def pooled_auroc(mode, seed_run):
    dfs = [pd.read_csv(config.cnn_fold_predictions(mode, f, seed_run)) for f in range(config.N_FOLDS)]
    oof = pd.concat(dfs, ignore_index=True)
    return float(roc_auc_score(oof.y_SLC26A4, oof.p_ent_SLC26A4))


def main():
    config.ensure_dirs()
    train.preflight()
    n_gpus = torch.cuda.device_count()
    t0 = time.time()
    jobs = [(mode, f, r) for r in SEED_RUNS for mode in ("single", "multitask") for f in range(config.N_FOLDS)]
    job_args = [(m, f, r, i % n_gpus) for i, (m, f, r) in enumerate(jobs)]
    for start in range(0, len(job_args), n_gpus):
        wave = job_args[start:start + n_gpus]
        print(f"\n== seed wave: {[(m, f, r) for m, f, r, _ in wave]} ==", flush=True)
        procs = [mp.Process(target=train.run_job, args=(m, f, g), kwargs={"seed_run": r})
                 for m, f, r, g in wave]
        for p in procs: p.start()
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"seed worker exited {p.exitcode}")
    mins = (time.time() - t0) / 60

    # pooled AUROC of the three seed runs (seed run 1 = the fold files of cnn.train)
    summary = {"minutes": round(mins, 1), "seeds": {}}
    for mode in ("single", "multitask"):
        aucs = {r: pooled_auroc(mode, r) for r in [1] + SEED_RUNS}
        vals = list(aucs.values())
        summary["seeds"][mode] = {
            "pooled_auroc_by_seed": {str(k): round(v, 4) for k, v in aucs.items()},
            "mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals, ddof=1)), 4),
            "min": round(float(np.min(vals)), 4), "max": round(float(np.max(vals)), 4)}
    summary_path = config.CNN_DIR / "seed_runs.json"
    with summary_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print("\n" + "=" * 60)
    for mode in ("single", "multitask"):
        s = summary["seeds"][mode]
        print(f"{mode:<10} seed AUROCs {s['pooled_auroc_by_seed']}  mean={s['mean']} sd={s['sd']}")
    print(f"{mins:.1f} min -> {summary_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
