"""Sensitivity analyses of the grade model by record source, comparator gene and CT date.

The script runs five analyses on the 229 patients of the six-structure grade model:

1. Two logistic models built only on the sheet source of the clinical dataset ("malformation" or
   "normal") and only on the letter prefix of the patient identifier, with the nested
   cross-validation of the grade model.
2. AUROC of the saved held-out scores of the grade model within each sheet source.
3. AUROC of the saved held-out scores for all SLC26A4 patients against the patients of each
   comparator gene group in turn.
4. Leave-source-out: the grade model is trained on the patients of one sheet source and tested on
   those of the other, in both directions.
5. Temporal (chronological) split: the grade model is trained on the 70% of patients with the
   earliest first CT and tested on the latest 30%, and in the reverse direction.

In analyses 4 and 5 (output keys leave_source_out and temporal_transport), a model trained on one
part of the cohort is applied to the other part, with C selected using only the training part.
The intervals are percentile bootstraps of the fixed test predictions; they do not repeat model
selection or training and are not external-validation uncertainty.

Writes sensitivity.json, sensitivity_leave_source_out.csv and sensitivity_temporal.csv to
config.GRADE_DIR.

Usage: python -m grade_models.sensitivity
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import config
from grade_models.analysis import (
    B_BOOT,
    STRUCTS,
    calibration_metrics,
    dump_json,
    load_corrected_patient_data,
    make_pipeline,
    nested_oof,
    onehot_block,
    tune_c,
)


def seed_for(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")


def fixed_auc_ci(y: np.ndarray, p: np.ndarray, label: str) -> list[float]:
    rng = np.random.default_rng(seed_for(label))
    values: list[float] = []
    for _ in range(B_BOOT):
        index = rng.integers(0, len(y), len(y))
        if np.unique(y[index]).size == 2:
            values.append(float(roc_auc_score(y[index], p[index])))
    return [float(value) for value in np.percentile(values, (2.5, 97.5))]


def score_test(y: np.ndarray, p: np.ndarray, label: str) -> dict:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "n_SLC26A4": int(np.sum(y)),
        "auroc": float(roc_auc_score(y, p)),
        "auroc_percentile_ci": fixed_auc_ci(y, p, label),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        **calibration_metrics(y, p),
    }


def fit_training_only(
    X: pd.DataFrame, y: np.ndarray, train: np.ndarray, test: np.ndarray, label: str,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    best_c, inner_auc, _ = tune_c(
        X.to_numpy()[train], y[train],
        groups=None if groups is None else np.asarray(groups)[train],
    )
    model = make_pipeline(best_c)
    model.fit(X.to_numpy()[train], y[train])
    probability = np.round(model.predict_proba(X.to_numpy()[test])[:, 1], 12)
    details = {
        "training_only_selected_C": best_c,
        "training_inner_3fold_mean_auroc": inner_auc,
        "n_train": int(train.sum()),
        "train_SLC26A4": int(y[train].sum()),
        **score_test(y[test], probability, label),
    }
    return probability, details


def main() -> None:
    config.ensure_dirs()
    patient_df, _ = load_corrected_patient_data()
    manifest = pd.read_csv(config.FOLDS)[["patient_id", "fold", "binary_label"]]
    data = manifest.merge(
        patient_df, on="patient_id", how="left", validate="one_to_one"
    ).set_index("patient_id")
    y = (data.binary_label == 0).astype(int).to_numpy()
    folds = data.fold.astype(int).to_numpy()
    groups = data["cv_group_id"].astype(str).to_numpy()
    anatomy = onehot_block(data, STRUCTS)

    primary_oof = (
        pd.read_csv(config.GRADE_PREDICTIONS)
        .set_index("patient_id")
        .loc[data.index, "p_4_anatomy_PRIMARY"]
        .to_numpy()
    )
    if len(primary_oof) != 229 or not np.isfinite(primary_oof).all():
        raise AssertionError("Held-out predictions of the grade model are incomplete")

    sheet = data.sheet_source.astype(str)
    source_onehot = pd.get_dummies(sheet, dtype=float)
    prefix_onehot = pd.get_dummies(data.id_prefix.astype(str), dtype=float)
    source_probability, _ = nested_oof(source_onehot, y, folds, groups=groups)
    prefix_probability, _ = nested_oof(prefix_onehot, y, folds, groups=groups)

    within_source = {}
    for level in sorted(sheet.unique()):
        selected = sheet.eq(level).to_numpy()
        within_source[level] = {
            **score_test(
                y[selected], primary_oof[selected], f"within_source_{level}"
            ),
            "scope": (
                "Subgroup evaluation of the across-source held-out predictions; "
                "not a source-held-out estimate."
            ),
        }

    comparator_results = {}
    comparator_gene = data.gene_class_fine.astype(str).to_numpy()
    for gene in sorted(set(comparator_gene[y == 0])):
        selected = (y == 1) | ((y == 0) & (comparator_gene == gene))
        comparator_results[gene] = {
            "n_comparator": int(((y == 0) & (comparator_gene == gene)).sum()),
            "auroc": float(roc_auc_score(y[selected], primary_oof[selected])),
            "auroc_percentile_ci": fixed_auc_ci(
                y[selected], primary_oof[selected], f"comparator_{gene}"
            ),
        }

    leave_source = {}
    source_predictions = []
    for train_level, test_level in (("malformation", "normal"), ("normal", "malformation")):
        train = sheet.eq(train_level).to_numpy()
        test = sheet.eq(test_level).to_numpy()
        probability, details = fit_training_only(
            anatomy,
            y,
            train,
            test,
            f"source_{train_level}_to_{test_level}",
            groups=groups,
        )
        key = f"train_{train_level}_test_{test_level}"
        leave_source[key] = details
        source_predictions.append(
            pd.DataFrame(
                {
                    "patient_id": data.index[test],
                    "analysis": key,
                    "y_SLC26A4": y[test],
                    "probability": probability,
                }
            )
        )

    ct_manifest = pd.read_csv(config.CT_MANIFEST)
    first_ct = (
        ct_manifest.dropna(subset=["ct_date"])
        .assign(ct_date=lambda frame: pd.to_datetime(frame.ct_date))
        .groupby("patient_id").ct_date.min()
    )
    data["first_ct"] = first_ct.reindex(data.index)
    if data.first_ct.isna().any():
        raise AssertionError("Temporal analysis requires a first CT date for all 229 patients")
    order = np.argsort(data.first_ct.to_numpy())
    cutoff = int(0.70 * len(data))
    early_index, late_index = order[:cutoff], order[cutoff:]
    early = np.zeros(len(data), dtype=bool)
    late = np.zeros(len(data), dtype=bool)
    early[early_index] = True
    late[late_index] = True

    temporal = {}
    temporal_predictions = []
    # The keys also seed the bootstrap intervals.
    for train, test, key in (
        (early, late, "train_earliest_70_test_latest_30"),
        (late, early, "reverse_train_latest_30_test_earliest_70"),
    ):
        probability, details = fit_training_only(
            anatomy, y, train, test, f"temporal_{key}", groups=groups
        )
        temporal[key] = details
        temporal_predictions.append(
            pd.DataFrame(
                {
                    "patient_id": data.index[test],
                    "analysis": key,
                    "y_SLC26A4": y[test],
                    "probability": probability,
                }
            )
        )

    metrics = {
        "analysis": "grade_model_sensitivity",
        "primary_oof_auroc": float(roc_auc_score(y, primary_oof)),
        "provenance_only_nested_oof": {
            "sheet_source_auroc": float(roc_auc_score(y, source_probability)),
            "id_prefix_auroc": float(roc_auc_score(y, prefix_probability)),
        },
        "primary_oof_within_source": within_source,
        "primary_oof_by_comparator_gene": {
            "groups": comparator_results,
            "unweighted_macro_auroc": float(
                np.mean([value["auroc"] for value in comparator_results.values()])
            ),
            "scope": "Descriptive case-mix analysis of the fixed held-out predictions.",
        },
        "leave_source_out": {
            "results": leave_source,
            "scope": (
                "Leave-one-sheet-out analysis with training-only "
                "hyperparameter selection; one training direction contains only "
                f"{leave_source['train_normal_test_malformation']['train_SLC26A4']} "
                "SLC26A4 patients. Sheet source is not an external acquisition site."
            ),
        },
        "temporal_transport": {
            "early_date_range": [
                str(data.first_ct.iloc[early_index[0]].date()),
                str(data.first_ct.iloc[early_index[-1]].date()),
            ],
            "late_date_range": [
                str(data.first_ct.iloc[late_index[0]].date()),
                str(data.first_ct.iloc[late_index[-1]].date()),
            ],
            "results": temporal,
            "scope": (
                "Retrospective single-center temporal sensitivity with training-only "
                "hyperparameter selection; not prospective or external validation."
            ),
        },
        "uncertainty_scope": (
            "Percentile patient bootstraps condition on fixed test predictions and do "
            "not repeat model selection or training."
        ),
    }

    dump_json(config.GRADE_DIR / "sensitivity.json", metrics)
    pd.concat(source_predictions, ignore_index=True).to_csv(
        config.GRADE_DIR / "sensitivity_leave_source_out.csv", index=False
    )
    pd.concat(temporal_predictions, ignore_index=True).to_csv(
        config.GRADE_DIR / "sensitivity_temporal.csv", index=False
    )
    print("Grade-model sensitivity analyses done")
    print(metrics["leave_source_out"]["results"])
    print(metrics["temporal_transport"]["results"])


if __name__ == "__main__":
    main()
