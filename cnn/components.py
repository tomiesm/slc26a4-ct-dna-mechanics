"""Parts of the slice-level CNN: training settings, augmentation, dataset, metrics and patient aggregation."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import config
from imaging import slice_cache

# ── Settings ───────────────────────────────────────────────────────────────────
CENTRAL_FRAC       = 0.2
BACKBONE_LR        = 1e-5
HEAD_LR            = 1e-4
WEIGHT_DECAY       = 1e-4
MAX_EPOCHS         = 100
MIN_EPOCHS         = 50
PATIENCE           = 15
DROPOUT            = 0.9
MIXUP_ALPHA        = 0.8
SLICE_BATCH_SIZE   = 8
SLICE_NUM_WORKERS  = 4
SLICE_LABEL_SMOOTH = 0.1

N_SLC26A4_CLASSES = 2


def _log(msg):
    print(msg, flush=True)


# ── Augmentation ───────────────────────────────────────────────────────────────

class SliceAugmentation(torch.nn.Module):
    """Slice augmentation: geometric transforms, intensity changes, blur and random erasing.

    Values are clamped to [0, 1] after the intensity changes and again at the end.
    """

    def __init__(self, training=True):
        super().__init__()
        self._is_training = training
        if training:
            import kornia.augmentation as K
            self.geom_aug = torch.nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomAffine(
                    degrees=30.0,
                    translate=(0.15, 0.15),
                    scale=(0.80, 1.20),
                    padding_mode="zeros",
                ),
            )
            self.intensity_aug = torch.nn.Sequential(
                K.RandomGaussianNoise(mean=0.0, std=0.15, p=0.8),
                K.RandomBrightness(brightness=(0.50, 1.50), p=0.8),
                K.RandomContrast(contrast=(0.50, 1.50),    p=0.8),
            )
            self.post_aug = torch.nn.Sequential(
                K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.5), p=0.7),
                K.RandomErasing(p=0.6, scale=(0.05, 0.30)),
                K.RandomErasing(p=0.4, scale=(0.03, 0.20)),
                K.RandomErasing(p=0.2, scale=(0.02, 0.10)),
            )

    @torch.no_grad()
    def forward(self, x):
        if self._is_training:
            x = self.geom_aug(x)
            x = self.intensity_aug(x)
            x = x.clamp(0.0, 1.0)
            x = self.post_aug(x)
            x = x.clamp(0.0, 1.0)
        return x


# ── Dataset ────────────────────────────────────────────────────────────────────

class CentralSliceLevelDataset:
    """Slice-level dataset restricted to the central CENTRAL_FRAC of each patient's slices."""

    def __init__(self, patient_ids, labels):
        self.labels       = labels
        self.index        = []
        self.slice_counts = {}
        self._data        = {}

        for pid in patient_ids:
            slices = np.load(config.SLICE_DIR / f"{pid}.npy")
            N      = slices.shape[0]
            margin = int((1.0 - CENTRAL_FRAC) / 2.0 * N)
            start  = margin
            end    = max(N - margin, start + 1)
            central_indices = list(range(start, end))
            self.slice_counts[pid] = len(central_indices)
            for s in central_indices:
                self.index.append((pid, s))
            self._data[pid] = slices

        total = sum(self.slice_counts.values())
        _log(f"    CentralSliceLevelDataset: {len(patient_ids)} patients, "
             f"{total} central slices (central {int(CENTRAL_FRAC*100)}%)")

    def get_sample_weights(self):
        """Per-slice sampling weights: equal total weight per class, split equally over each class's
        patients and then over each patient's slices."""
        from collections import Counter
        patient_list = sorted(set(pid for pid, _ in self.index))
        class_counts = Counter(self.labels[pid] for pid in patient_list)
        patient_weight = {pid: 1.0 / class_counts[self.labels[pid]] for pid in patient_list}
        return [patient_weight[pid] / self.slice_counts[pid] for pid, _ in self.index]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        pid, slice_idx = self.index[idx]
        sl = self._data[pid][slice_idx]
        t  = torch.from_numpy(sl.copy()).float().unsqueeze(0) / 255.0
        t  = t.expand(3, -1, -1).contiguous()
        return t, self.labels[pid], pid, slice_idx

    @staticmethod
    def prepare_cache(patient_ids):
        """Make sure the resized slices of these patients exist in config.SLICE_DIR."""
        slice_cache.prepare_cache(patient_ids)


# ── Metrics and patient-level aggregation ──────────────────────────────────────

def _binary_auroc(y_true, probs):
    """AUROC with probs[:, 1] as the score; label 1 (comparator) is the positive class here."""
    return roc_auc_score(y_true, probs[:, 1])


def _aggregate_patient_entropy(slice_preds):
    """Aggregate slice-level predictions to patient level.

    Parameters
    ----------
    slice_preds : list of dict with keys "pid", "prob" (ndarray shape [2]),
                  "label" (int), optionally "feat" (ndarray shape [512])

    Returns
    -------
    pids, mean_probs, ent_probs, labels, mean_feats, ent_feats
        Patients are sorted by identifier. ent_* are averages weighted by
        (log 2 - slice entropy), so confident slices count more.
    """
    from collections import defaultdict
    patient_data = defaultdict(lambda: {"probs": [], "labels": [], "feats": []})

    for item in slice_preds:
        pid = item["pid"]
        patient_data[pid]["probs"].append(item["prob"])
        patient_data[pid]["labels"].append(item["label"])
        if "feat" in item:
            patient_data[pid]["feats"].append(item["feat"])

    pids, mean_probs, ent_probs, labels = [], [], [], []
    mean_feats, ent_feats = [], []
    eps = 1e-8

    for pid, data in sorted(patient_data.items()):
        pa = np.array(data["probs"])     # (N_slices, 2)
        entropy = -np.sum(pa * np.log(pa + eps), axis=1)
        max_ent = np.log(N_SLC26A4_CLASSES)
        w = (max_ent - entropy) + eps
        w /= w.sum()

        pids.append(pid)
        mean_probs.append(pa.mean(axis=0))
        ent_probs.append(np.average(pa, axis=0, weights=w))
        labels.append(data["labels"][0])

        if data["feats"]:
            fa = np.array(data["feats"])  # (N_slices, 512)
            mean_feats.append(fa.mean(axis=0))
            ent_feats.append(np.average(fa, axis=0, weights=w))
        else:
            mean_feats.append(None)
            ent_feats.append(None)

    return (pids,
            np.array(mean_probs), np.array(ent_probs),
            np.array(labels),
            mean_feats, ent_feats)
