"""Train the single-task and multitask CNNs on coronal CT slices and predict the held-out folds.

The two networks are the same except that the multitask network adds six auxiliary anatomy heads
and their weighted losses:

  * ImageNet-pretrained ResNet-18 backbone, dropout and genotype head (Linear(512, 2));
  * central 20% of the coronal slices and entropy-weighted aggregation to the patient;
  * strong slice augmentation followed by ImageNet channel normalization (the backbone is
    ImageNet-pretrained);
  * WeightedRandomSampler, AdamW with separate backbone and head learning rates, weight decay,
    dropout and label smoothing;
  * MixUp (alpha 0.8) on the genotype loss and, in the multitask network, on the six anatomy
    losses with the same mixing weight and permutation;
  * for each outer fold, the number of epochs is chosen on a 20% inner validation split, the
    network is refitted on the whole outer training set for that number of epochs, and the
    held-out outer fold is predicted.

Before any training the script checks the cohort counts, that related patients share a fold, that
every image is present, and that no identical image file appears in two outer folds.

SLC26A4 is the positive class and corresponds to binary_label == 0; the reported probability is
softmax[:, 0] and the AUROC target is (binary_label == 0).

This script is seed run 1; cnn.seed_runs calls run_job for seed runs 2 and 3.

Usage:
  python -m cnn.train                        # both networks, 5 folds, all GPUs
  python -m cnn.train --onejob single 0 0    # one (network, fold, GPU) job
"""
import sys, time, json, hashlib, platform
import multiprocessing as mp
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

import config
from cnn.components import (
    CentralSliceLevelDataset, SliceAugmentation,
    BACKBONE_LR, HEAD_LR, WEIGHT_DECAY, MAX_EPOCHS, MIN_EPOCHS, PATIENCE,
    DROPOUT, MIXUP_ALPHA, SLICE_BATCH_SIZE, SLICE_NUM_WORKERS, SLICE_LABEL_SMOOTH,
)
from cnn.components import _binary_auroc, _aggregate_patient_entropy

INNER_VAL_FRAC = 0.20
ANAT_W = 0.3                                    # weight of each auxiliary anatomy loss
STRUCT_NCLASS = {"cochlea": 7, "vestibule": 4, "scc": 4, "iac": 3, "eva": 2, "aperture": 2}
STRUCTS = list(STRUCT_NCLASS)
# ImageNet channel statistics (normalize after [0,1] conversion).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def preflight() -> dict:
    """Check the cohort and the images before any training; return input hashes and counts."""
    manifest = pd.read_csv(config.FOLDS)
    fp = pd.read_csv(config.IMAGE_FINGERPRINTS)
    assert len(manifest) == 229, f"expected 229, got {len(manifest)}"
    assert int((manifest.binary_label == 0).sum()) == 88, "expected 88 SLC26A4"
    assert int((manifest.binary_label == 1).sum()) == 141, "expected 141 comparators"
    assert manifest.groupby("cv_group_id").fold.nunique().max() == 1, "a group spans >1 fold"
    # every image present
    missing = [p for p in manifest.patient_id
               if not (config.SLICE_DIR / f"{p}.npy").exists()
               or not (config.CORONAL_CROPS_DIR / f"{p}.npz").exists()]
    assert not missing, f"missing images: {missing}"
    # no identical image file in two outer folds
    for col in ("cache_sha256", "coronal_npz_sha256"):
        dup = fp[fp[col].duplicated(keep=False)]
        if len(dup):
            assert dup.groupby(col).fold.nunique().max() == 1, f"{col} crosses folds"
    return {
        "folds_sha256": sha256(config.FOLDS),
        "image_fingerprints_sha256": sha256(config.IMAGE_FINGERPRINTS),
        "people_sha256": sha256(config.PEOPLE),
        "n_patients": int(len(manifest)),
        "n_SLC26A4": int((manifest.binary_label == 0).sum()),
        "n_comparators": int((manifest.binary_label == 1).sum()),
    }


def load_groups():
    """patient_id -> cv_group_id (keeps the possibly related pair together)."""
    m = pd.read_csv(config.FOLDS)
    return dict(zip(m.patient_id, m.cv_group_id.astype(str)))


def group_aware_inner_split(ids, y_ids, group_ids, seed, frac=INNER_VAL_FRAC):
    """Deterministic stratified 20% inner-validation split that keeps grouped patients (the
    possibly related pair) on one side. Uses a plain stratified split when every group is a
    singleton."""
    ids = list(ids)
    y_ids = np.asarray(y_ids)
    group_ids = np.asarray(group_ids)
    if len(np.unique(group_ids)) < len(group_ids):
        n_splits = int(round(1.0 / frac))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        _, val_idx = next(iter(sgkf.split(ids, y_ids, group_ids)))
        val = set(int(i) for i in val_idx)
        inner_val = [ids[i] for i in range(len(ids)) if i in val]
        inner_train = [ids[i] for i in range(len(ids)) if i not in val]
        return inner_train, inner_val
    return train_test_split(ids, test_size=frac, stratify=y_ids, random_state=seed)


def load_labels():
    """Genotype labels (0 = SLC26A4), worst-ear anatomy grade per structure, and fold members."""
    manifest = pd.read_csv(config.FOLDS)
    df = pd.read_parquet(config.PEOPLE).set_index("patient_id")
    geno = dict(zip(manifest.patient_id, manifest.binary_label))          # 0 = SLC26A4
    fold_ids = {f: manifest[manifest.fold == f].patient_id.tolist() for f in range(config.N_FOLDS)}
    anat = {}
    for s in STRUCTS:
        worst = df[[f"{s}_R", f"{s}_L"]].max(axis=1).fillna(0).round().astype(int)
        worst = worst.clip(0, STRUCT_NCLASS[s] - 1)
        anat[s] = worst.to_dict()
    return geno, anat, fold_ids


class SliceResNet(nn.Module):
    """ResNet-18 backbone + genotype head; the multitask network adds six anatomy heads."""

    def __init__(self, mode: str, dropout: float = DROPOUT):
        super().__init__()
        assert mode in ("single", "multitask")
        self.mode = mode
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(p=dropout)
        self.geno = nn.Linear(512, 2)
        if mode == "multitask":
            self.anat = nn.ModuleDict({s: nn.Linear(512, STRUCT_NCLASS[s]) for s in STRUCTS})

    def forward(self, x, return_features=False):
        f = self.features(x).flatten(1)
        fd = self.dropout(f)
        out = {"geno": self.geno(fd)}
        if self.mode == "multitask":
            for s in STRUCTS:
                out[s] = self.anat[s](fd)
        if return_features:
            return out, f
        return out

    def head_parameters(self):
        names = ("geno", "anat")
        return [p for n, p in self.named_parameters() if n.startswith(names)]


def make_loader(ids, geno, train):
    ds = CentralSliceLevelDataset(ids, geno)
    if train:
        s = WeightedRandomSampler(ds.get_sample_weights(), len(ds), replacement=True)
        return DataLoader(ds, batch_size=SLICE_BATCH_SIZE, sampler=s, num_workers=SLICE_NUM_WORKERS,
                          pin_memory=True, drop_last=True, persistent_workers=True)
    return DataLoader(ds, batch_size=SLICE_BATCH_SIZE, shuffle=False, num_workers=SLICE_NUM_WORKERS,
                      pin_memory=True, persistent_workers=True)


def build(mode, device):
    aug_tr = SliceAugmentation(training=True).to(device)
    aug_ev = SliceAugmentation(training=False).to(device)
    model = SliceResNet(mode).to(device)
    opt = AdamW([{"params": list(model.features.parameters()), "lr": BACKBONE_LR},
                 {"params": model.head_parameters(), "lr": HEAD_LR}], weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss(label_smoothing=SLICE_LABEL_SMOOTH)
    return model, aug_tr, aug_ev, opt, crit, torch.amp.GradScaler()


def normalize(x, device):
    """Apply ImageNet channel normalization after augmentation (inputs are in [0,1])."""
    return (x - IMAGENET_MEAN.to(device)) / IMAGENET_STD.to(device)


def train_epoch(model, aug_tr, opt, crit, scaler, loader, device, anat):
    """One epoch. MixUp uses the same lam and permutation for the genotype loss and, in the
    multitask network, for the six anatomy losses."""
    model.train()
    for imgs, glb, pids, _ in loader:
        imgs = normalize(aug_tr(imgs.to(device)), device)
        glb = glb.to(device)
        lam = float(np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA))
        idx = torch.randperm(imgs.size(0), device=device)
        mixed = lam * imgs + (1 - lam) * imgs[idx]
        opt.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(mixed)
            loss = lam * crit(out["geno"], glb) + (1 - lam) * crit(out["geno"], glb[idx])
            if model.mode == "multitask":
                for s in STRUCTS:
                    slb = torch.tensor([anat[s][p] for p in pids], device=device)
                    loss = loss + ANAT_W * (lam * crit(out[s], slb) + (1 - lam) * crit(out[s], slb[idx]))
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()


def geno_entropy_auroc(model, aug_ev, loader, device):
    model.eval(); preds = []
    with torch.no_grad():
        for imgs, glb, pids, _ in loader:
            with torch.amp.autocast("cuda"):
                out = model(normalize(aug_ev(imgs.to(device)), device))
            p = torch.softmax(out["geno"].float(), 1).cpu().numpy()
            for i in range(len(pids)):
                preds.append({"pid": pids[i], "prob": p[i], "label": glb[i].item()})
    _, _, ent, lab, _, _ = _aggregate_patient_entropy(preds)
    return _binary_auroc(lab, ent)


def run_job(mode, outer_fold, gpu_id, result_dict=None, seed_run=1):
    """Train one network for one outer fold and predict the held-out fold.

    Seed run 1 uses seed_inner = CNN_SEED + fold and seed_refit = CNN_SEED + fold + 100; seed runs
    2 and 3 add 1000 * seed_run to both. Every seed run writes the held-out fold predictions. Only
    seed run 1 also saves the checkpoint, the patient features and, for the multitask network, the
    anatomy predictions. If result_dict is given, the fold summary is stored under (mode, fold).
    """
    device = torch.device(f"cuda:{gpu_id}")
    offset = 0 if seed_run == 1 else 1000 * seed_run
    seed_inner = config.CNN_SEED + outer_fold + offset
    seed_refit = config.CNN_SEED + outer_fold + 100 + offset
    tag = f"{mode} fold {outer_fold}" + ("" if seed_run == 1 else f" seed run {seed_run}")
    torch.manual_seed(seed_inner); np.random.seed(seed_inner)
    geno, anat, fold_ids = load_labels()
    cvgroup = load_groups()
    outer_test = fold_ids[outer_fold]
    outer_train = [p for f in range(config.N_FOLDS) if f != outer_fold for p in fold_ids[f]]
    y_ot = np.array([geno[p] for p in outer_train])
    g_ot = [cvgroup[p] for p in outer_train]
    inner_train, inner_val = group_aware_inner_split(outer_train, y_ot, g_ot, seed_inner)
    CentralSliceLevelDataset.prepare_cache(outer_train + outer_test)

    # STEP 1: choose the epoch on the 20% inner validation split (genotype entropy AUROC)
    tr_ld = make_loader(inner_train, geno, True); iv_ld = make_loader(inner_val, geno, False)
    model, aug_tr, aug_ev, opt, crit, scaler = build(mode, device)
    best_auc, best_E = -1.0, 1
    for epoch in range(MAX_EPOCHS):
        train_epoch(model, aug_tr, opt, crit, scaler, tr_ld, device, anat)
        a = geno_entropy_auroc(model, aug_ev, iv_ld, device)
        if a > best_auc:
            best_auc, best_E = a, epoch + 1
        if epoch >= MIN_EPOCHS - 1 and (epoch + 1 - best_E) >= PATIENCE:
            break
    del model, opt, scaler; torch.cuda.empty_cache()
    print(f"[{tag} GPU {gpu_id}] inner-select best_E={best_E} "
          f"(inner-val geno AUROC {best_auc:.3f})", flush=True)

    # STEP 2: refit on ALL outer_train for exactly best_E epochs (no selection)
    torch.manual_seed(seed_refit); np.random.seed(seed_refit)
    ot_ld = make_loader(outer_train, geno, True)
    model, aug_tr, aug_ev, opt, crit, scaler = build(mode, device)
    for _ in range(best_E):
        train_epoch(model, aug_tr, opt, crit, scaler, ot_ld, device, anat)
    if seed_run == 1:
        torch.save(model.state_dict(), config.CNN_CHECKPOINT_DIR / f"{mode}_fold{outer_fold}.pt")

    # STEP 3: predict the untouched outer_test (genotype, and anatomy heads if multitask)
    model.eval(); tp = []; slice_anat = {s: [] for s in STRUCTS}
    with torch.no_grad():
        for imgs, glb, pids, _ in make_loader(outer_test, geno, False):
            with torch.amp.autocast("cuda"):
                out, feats = model(normalize(aug_ev(imgs.to(device)), device), return_features=True)
            pg = torch.softmax(out["geno"].float(), 1).cpu().numpy()
            fz = feats.float().cpu().numpy()
            for i in range(len(pids)):
                tp.append({"pid": pids[i], "prob": pg[i], "feat": fz[i], "label": glb[i].item()})
            if mode == "multitask" and seed_run == 1:
                for s in STRUCTS:
                    ps = torch.softmax(out[s].float(), 1).cpu().numpy()
                    pab = 1.0 - ps[:, 0]                       # P(abnormal) = 1 - P(grade 0)
                    for i in range(len(pids)):
                        slice_anat[s].append((pids[i], float(pab[i])))
    pids, mean_p, ent_p, lab, mfeat, _ = _aggregate_patient_entropy(tp)
    y_pos = [1 if l == 0 else 0 for l in lab]                  # SLC26A4 positive
    p_mean_slc = [float(p[0]) for p in mean_p]
    p_ent_slc = [float(p[0]) for p in ent_p]
    auc_mean = float(roc_auc_score(y_pos, p_mean_slc)) if len(set(y_pos)) > 1 else None
    auc_ent = float(roc_auc_score(y_pos, p_ent_slc)) if len(set(y_pos)) > 1 else None
    print(f"[{tag}] outer-test SLC26A4 AUROC mean={auc_mean:.3f} "
          f"ent={auc_ent:.3f} (refit {best_E} ep)", flush=True)

    pd.DataFrame({"patient_id": pids, "fold": outer_fold, "y_SLC26A4": y_pos,
                  "p_mean_SLC26A4": p_mean_slc, "p_ent_SLC26A4": p_ent_slc}).to_csv(
        config.cnn_fold_predictions(mode, outer_fold, seed_run), index=False)

    if seed_run == 1:
        np.save(config.CNN_PREDICTION_DIR / f"{mode}_features_fold{outer_fold}.npy",
                np.array([mfeat[i] for i in range(len(pids))], dtype=np.float32))
        np.save(config.CNN_PREDICTION_DIR / f"{mode}_patients_fold{outer_fold}.npy", np.array(pids))

    if mode == "multitask" and seed_run == 1:
        # Patient-level anatomy predictions of this fold, pooled over folds in aggregate():
        # P(abnormal) averaged over the patient's slices and the true abnormal status, per structure.
        rows = []
        for s in STRUCTS:
            agg = {}
            for pid, pab in slice_anat[s]:
                agg.setdefault(pid, []).append(pab)
            for pid in agg:
                rows.append({"patient_id": pid, "fold": outer_fold, "structure": s,
                             "p_abnormal": float(np.mean(agg[pid])),
                             "y_abnormal": int(anat[s][pid] > 0)})
        pd.DataFrame(rows).to_csv(
            config.CNN_PREDICTION_DIR / f"multitask_anatomy_fold{outer_fold}.csv", index=False)

    if result_dict is not None:
        result_dict[(mode, outer_fold)] = {
            "mode": mode, "outer_fold": outer_fold, "best_E": best_E,
            "seed_inner": seed_inner, "seed_refit": seed_refit,
            "inner_val_geno_auroc": float(best_auc),
            "auroc_mean": auc_mean, "auroc_ent": auc_ent, "n_outer_test": len(outer_test),
        }


def aggregate(mode, result_dict):
    dfs = [pd.read_csv(config.cnn_fold_predictions(mode, f)) for f in range(config.N_FOLDS)]
    oof = pd.concat(dfs, ignore_index=True)
    oof.to_csv(config.cnn_predictions(mode), index=False)
    y = oof.y_SLC26A4.values
    summary = {
        "mode": mode,
        "pooled_nested_auroc_mean": float(roc_auc_score(y, oof.p_mean_SLC26A4)),
        "pooled_nested_auroc_ent": float(roc_auc_score(y, oof.p_ent_SLC26A4)),
        "folds": [result_dict[(mode, f)] for f in range(config.N_FOLDS)],
        "design": ("inner-select epoch on 20% inner-val, refit on all outer-train for best_E, "
                   "predict untouched outer-test; SLC26A4 positive = binary_label 0"),
    }
    if mode == "multitask":
        # pooled patient-level anatomy AUROC (single AUROC over pooled OOF, not a mean of folds)
        adf = pd.concat([pd.read_csv(config.CNN_PREDICTION_DIR / f"multitask_anatomy_fold{f}.csv")
                         for f in range(config.N_FOLDS)], ignore_index=True)
        adf.to_csv(config.CNN_PREDICTION_DIR / "multitask_anatomy_oof.csv", index=False)
        pooled = {}
        for s in STRUCTS:
            sub = adf[adf.structure == s]
            pooled[s] = (float(roc_auc_score(sub.y_abnormal, sub.p_abnormal))
                         if sub.y_abnormal.nunique() > 1 else None)
        summary["pooled_patient_level_anat_auroc"] = pooled
    return summary


def main(onejob=None):
    config.ensure_dirs()
    provenance = preflight()
    provenance.update({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(), "seed_base": config.CNN_SEED,
        "mixup_alpha": MIXUP_ALPHA, "anat_weight": ANAT_W,
        "dropout": DROPOUT, "backbone_lr": BACKBONE_LR, "head_lr": HEAD_LR,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    })
    manager = mp.Manager(); result_dict = manager.dict(); t0 = time.time()

    if onejob is not None:
        mode, fold, gpu = onejob
        run_job(mode, fold, gpu, result_dict)
        print(json.dumps(dict(result_dict), default=str, indent=2))
        return

    n_gpus = torch.cuda.device_count()
    print(f"GPUs: {n_gpus} -> {provenance['gpu_names']}", flush=True)
    jobs = [(mode, f) for mode in ("single", "multitask") for f in range(config.N_FOLDS)]
    job_args = [(mode, f, i % n_gpus) for i, (mode, f) in enumerate(jobs)]
    for start in range(0, len(job_args), n_gpus):
        wave = job_args[start:start + n_gpus]
        print(f"\n== wave: {[(m, f) for m, f, _ in wave]} on GPUs {[g for *_, g in wave]} ==", flush=True)
        procs = [mp.Process(target=run_job, args=(m, f, g, result_dict)) for m, f, g in wave]
        for p in procs: p.start()
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker exited {p.exitcode}")
    mins = (time.time() - t0) / 60

    out = {"provenance": provenance, "minutes": round(mins, 1),
           "single": aggregate("single", result_dict),
           "multitask": aggregate("multitask", result_dict)}
    with config.CNN_SUMMARY.open("w") as fh:
        json.dump(out, fh, indent=2, default=str)
    print("\n" + "=" * 64)
    print(f"SINGLE-TASK   pooled genotype AUROC (entropy) = {out['single']['pooled_nested_auroc_ent']:.3f}")
    print(f"MULTITASK     pooled genotype AUROC (entropy) = {out['multitask']['pooled_nested_auroc_ent']:.3f}")
    if out["multitask"].get("pooled_patient_level_anat_auroc"):
        print("anatomy outputs pooled patient-level AUROC: "
              + json.dumps({k: (round(v, 3) if v else v)
                            for k, v in out["multitask"]["pooled_patient_level_anat_auroc"].items()}))
    print(f"total {mins:.1f} min -> {config.CNN_SUMMARY}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--onejob":
        main(onejob=(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))
    else:
        main()
