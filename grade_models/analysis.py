"""Fit the logistic models on the coded CT grades, run their sensitivity analyses and build the cohort flow.

The grade models form a ladder (outer-fold prevalence, then logistic models on demographics, EVA,
EVA + cochlea, all six structures, all structures + demographics); each logistic model is fitted
with nested cross-validation: C is chosen by three-fold inner cross-validation inside each of the five outer training folds, with fold-local median
imputation of age. The script adds paired bootstrap and DeLong comparisons, a fixed-score label
permutation null, calibration-in-the-large and slope, a testing-propensity weighted AUC, optimal
age matching, a model without EVA, and forward structure selection inside each outer training fold.
The bootstrap and permutation analyses are conditional on fixed held-out scores; they do not
include model-development or training uncertainty. A documented birthday correction is applied in
memory before any analysis.

Usage: python -m grade_models.analysis
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

SEED = 20260702
B_BOOT = 2000
B_PERM = 10_000
C_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)

LEVELS = {
    "cochlea": (1, 2, 3, 4, 5, 6),
    "vestibule": (1, 2, 3),
    "scc": (1, 2, 3),
    "iac": (1, 2),
    "eva": (1,),
    "aperture": (1,),
}
STRUCTS = tuple(LEVELS)

# Tested patients added to the clinical dataset after the cohort list was fixed; they are not part
# of the analysis cohort.
ADDED_AFTER_COHORT_LIST = {
    "<PATIENT_ID_48>",
    "<PATIENT_ID_49>",
    "<PATIENT_ID_50>",
    "<PATIENT_ID_51>",
    "<PATIENT_ID_52>",
    "<PATIENT_ID_53>",
    "<PATIENT_ID_54>",
    "<PATIENT_ID_55>",
    "<PATIENT_ID_56>",
    "<PATIENT_ID_57>",
}


def _json_value(value):
    """Convert numpy values and non-finite floats into strict-JSON values."""
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dump_json(path: Path, obj) -> None:
    path.write_text(
        json.dumps(obj, indent=2, default=_json_value, allow_nan=False) + "\n"
    )


def derived_seed(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return (SEED + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def load_corrected_patient_data() -> tuple[pd.DataFrame, dict]:
    """Load the person table and apply the documented data corrections in memory.

    Returns the corrected table and a summary of the corrections.
    """
    df = pd.read_parquet(config.PEOPLE).copy()
    master = pd.read_parquet(config.RECORDS)
    corrections = pd.read_csv(config.DATA_CORRECTIONS)
    correction_rows = []

    for row in corrections.itertuples(index=False):
        if row.field != "birthday":
            raise ValueError(f"Unsupported correction field: {row.field}")
        mask = df["patient_id"].eq(row.patient_id)
        if mask.sum() != 1:
            raise AssertionError(f"Expected one patient row for {row.patient_id}")
        observed = pd.Timestamp(df.loc[mask, "birthday"].iloc[0]).date().isoformat()
        if observed != row.original_value:
            raise AssertionError(
                f"Correction precondition failed for {row.patient_id}: "
                f"observed {observed}, expected {row.original_value}"
            )
        first_ct = pd.to_datetime(
            master.loc[master["patient_id"].eq(row.patient_id), "ct_date"]
        ).min()
        corrected_birthday = pd.Timestamp(row.corrected_value)
        corrected_age = (first_ct - corrected_birthday).days / 365.25
        df.loc[mask, "birthday"] = corrected_birthday
        df.loc[mask, "age_at_first_ct"] = corrected_age
        correction_rows.append(
            {
                "patient_id": row.patient_id,
                "field": row.field,
                "original_value": row.original_value,
                "corrected_value": row.corrected_value,
                "first_ct": first_ct.date().isoformat(),
                "corrected_age_at_first_ct": float(corrected_age),
                "evidence": row.evidence,
                "status": row.status,
            }
        )

    ages = df["age_at_first_ct"].dropna().astype(float)
    if (ages < 0).any():
        bad = df.loc[df["age_at_first_ct"].astype(float) < 0, "patient_id"].tolist()
        raise AssertionError(f"Negative ages remain after correction: {bad}")
    summary = {
        "corrections": correction_rows,
        "n_age_missing": int(df["age_at_first_ct"].isna().sum()),
        "age_min": float(ages.min()),
        "age_max": float(ages.max()),
        "negative_age_count": int((ages < 0).sum()),
        "source_data_mutated": False,
        "note": "The correction is applied in memory; the person table on disk is not modified.",
    }
    return df, summary


def onehot_block(d: pd.DataFrame, structs) -> pd.DataFrame:
    cols = {}
    for struct in structs:
        for side in ("R", "L"):
            raw = (
                d[f"{struct}_{side}"]
                .fillna(0)
                .astype(float)
                .round()
                .astype(int)
                .to_numpy()
            )
            for level in LEVELS[struct]:
                cols[f"{struct}_{side}_{level}"] = (raw == level).astype(int)
    return pd.DataFrame(cols, index=d.index)


def make_pipeline(c: float, class_weight="balanced", penalty="l2") -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    penalty=penalty,
                    max_iter=5000,
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )


def tune_c(X: np.ndarray, y: np.ndarray, groups=None, penalty="l2", class_weight="balanced"):
    """Choose the most regularized model within one SE of the best mean AUC.

    The selected C is the smallest C whose mean inner AUC is within one standard error of the
    best mean inner AUC. This rule is used wherever C is tuned.

    When ``groups`` is supplied and contains any non-singleton group (e.g. the
    conservatively grouped possible-family pair), the inner three-fold split is
    group-aware so that grouped patients never straddle inner train/validation.
    """
    if groups is not None and len(np.unique(groups)) < len(groups):
        splitter = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = StratifiedKFold(3, shuffle=True, random_state=SEED)
        splits = list(splitter.split(X, y))
    scores = {}
    fold_scores = {}
    for c in C_GRID:
        fold_auc = []
        for tr, va in splits:
            model = make_pipeline(c, class_weight=class_weight, penalty=penalty)
            model.fit(X[tr], y[tr])
            pred = model.predict_proba(X[va])[:, 1]
            fold_auc.append(roc_auc_score(y[va], pred))
        score = float(np.mean(fold_auc))
        scores[str(c)] = score
        fold_scores[str(c)] = [float(value) for value in fold_auc]

    # C_GRID is ascending from strongest to weakest regularization.
    best_mean_c = max(C_GRID, key=lambda value: scores[str(value)])
    best_fold_scores = np.asarray(fold_scores[str(best_mean_c)], dtype=float)
    best_standard_error = float(best_fold_scores.std(ddof=1) / np.sqrt(len(best_fold_scores)))
    eligibility_threshold = scores[str(best_mean_c)] - best_standard_error
    selected_c = next(
        value for value in C_GRID if scores[str(value)] >= eligibility_threshold - 1e-12
    )
    details = {
        "mean_auc_by_C": scores,
        "fold_auc_by_C": fold_scores,
        "best_mean_C": float(best_mean_c),
        "best_mean_auc": float(scores[str(best_mean_c)]),
        "best_mean_standard_error": best_standard_error,
        "one_standard_error_threshold": float(eligibility_threshold),
        "selected_C": float(selected_c),
        "selection_rule": "smallest C with mean inner AUC within one standard error of the best mean inner AUC",
    }
    return float(selected_c), float(scores[str(selected_c)]), details


def nested_oof(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    *,
    groups=None,
    class_weight="balanced",
    penalty="l2",
):
    values = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    groups = None if groups is None else np.asarray(groups)
    oof = np.full(len(y), np.nan)
    chosen = {}
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        best_c, best_inner_auc, tuning_details = tune_c(
            values[train], y[train],
            groups=None if groups is None else groups[train],
            penalty=penalty, class_weight=class_weight,
        )
        model = make_pipeline(best_c, class_weight=class_weight, penalty=penalty)
        model.fit(values[train], y[train])
        oof[test] = model.predict_proba(values[test])[:, 1]
        chosen[str(int(fold))] = {
            "C": best_c,
            "inner_mean_auc": best_inner_auc,
            "best_mean_C": tuning_details["best_mean_C"],
            "best_mean_standard_error": tuning_details["best_mean_standard_error"],
            "selection_rule": tuning_details["selection_rule"],
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
        }
    if np.isnan(oof).any():
        raise AssertionError("Nested OOF predictions are incomplete")
    # Probabilities are rounded to 12 decimals, so equal scores are stored as ties.
    return np.round(oof, 12), chosen


def bootstrap_auc_ci(y, pred, label: str, B=B_BOOT):
    rng = np.random.default_rng(derived_seed("auc_ci:" + label))
    estimates = []
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size == 2:
            estimates.append(roc_auc_score(y[idx], pred[idx]))
    return [float(v) for v in np.percentile(estimates, (2.5, 97.5))]


def calibration_metrics(y, pred):
    pred = np.clip(np.asarray(pred), 1e-6, 1 - 1e-6)
    lp = np.log(pred / (1 - pred))
    # Calibration-in-the-large: intercept with the model logit as an offset.
    citl = sm.GLM(
        y,
        np.ones((len(y), 1)),
        family=sm.families.Binomial(),
        offset=lp,
    ).fit(disp=0)
    if np.unique(lp).size < 2:
        return {
            "calibration_in_the_large": float(citl.params[0]),
            "calibration_slope": None,
            "calibration_model_intercept": None,
        }
    joint = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit(disp=0)
    return {
        "calibration_in_the_large": float(citl.params[0]),
        "calibration_slope": float(joint.params[1]),
        "calibration_model_intercept": float(joint.params[0]),
    }


def score_metrics(y, pred, label: str):
    pred = np.asarray(pred)
    result = {
        "auroc": float(roc_auc_score(y, pred)) if np.unique(pred).size > 1 else 0.5,
        "auroc_percentile_ci": bootstrap_auc_ci(y, pred, label)
        if np.unique(pred).size > 1
        else None,
        "pr_auc": float(average_precision_score(y, pred)),
        "log_loss": float(log_loss(y, np.clip(pred, 1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y, pred)),
        "balanced_accuracy_at_0_5": float(
            ((pred >= 0.5)[y == 1].mean() + (pred < 0.5)[y == 0].mean()) / 2
        ),
    }
    result.update(calibration_metrics(y, pred))
    return result


def paired_bootstrap_auc(y, pred_a, pred_b, label: str, B=B_BOOT):
    rng = np.random.default_rng(derived_seed("paired_auc:" + label))
    n = len(y)
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if np.unique(y[idx]).size == 2:
            diffs.append(
                roc_auc_score(y[idx], pred_a[idx])
                - roc_auc_score(y[idx], pred_b[idx])
            )
    diffs = np.asarray(diffs)
    return {
        "delta_auroc": float(roc_auc_score(y, pred_a) - roc_auc_score(y, pred_b)),
        "percentile_ci": [float(v) for v in np.percentile(diffs, (2.5, 97.5))],
        "bootstrap_tail_area_two_sided_descriptive": float(
            min(1.0, 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0)))
        ),
        "note": "Tail area is descriptive bootstrap output, not a label-permutation p-value.",
    }


def delong_auc_test(y, pred_a, pred_b):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]

    def placements(scores):
        pos_scores, neg_scores = scores[pos], scores[neg]
        v10 = np.asarray(
            [np.mean(neg_scores < value) + 0.5 * np.mean(neg_scores == value) for value in pos_scores]
        )
        v01 = np.asarray(
            [np.mean(pos_scores > value) + 0.5 * np.mean(pos_scores == value) for value in neg_scores]
        )
        return v10, v01

    v10_a, v01_a = placements(pred_a)
    v10_b, v01_b = placements(pred_b)
    cov_pos = np.cov(np.vstack((v10_a, v10_b)))
    cov_neg = np.cov(np.vstack((v01_a, v01_b)))
    cov = cov_pos / len(pos) + cov_neg / len(neg)
    variance = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    delta = float(v10_a.mean() - v10_b.mean())
    if variance <= 0:
        return {"z": None, "p_two_sided": None, "delta_auroc": delta}
    z = delta / np.sqrt(variance)
    return {
        "z": float(z),
        "p_two_sided": float(2 * stats.norm.sf(abs(z))),
        "delta_auroc": delta,
    }


def nested_intercept_recalibration(X, y, folds, outer_tuning, groups=None):
    """Learn only a probability intercept inside each outer training partition."""
    values = X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)
    groups = None if groups is None else np.asarray(groups)
    calibrated = np.full(len(y), np.nan)
    outer_intercepts = {}
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        X_train, y_train = values[train], y[train]
        g_train = None if groups is None else groups[train]
        chosen_c = float(outer_tuning[str(int(fold))]["C"])
        inner_oof = np.full(train.sum(), np.nan)
        if g_train is not None and len(np.unique(g_train)) < len(g_train):
            splitter = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
            inner_splits = splitter.split(X_train, y_train, g_train)
        else:
            splitter = StratifiedKFold(3, shuffle=True, random_state=SEED)
            inner_splits = splitter.split(X_train, y_train)
        for inner_train, inner_valid in inner_splits:
            model = make_pipeline(chosen_c)
            model.fit(X_train[inner_train], y_train[inner_train])
            inner_oof[inner_valid] = model.predict_proba(X_train[inner_valid])[:, 1]
        inner_logit = np.log(
            np.clip(inner_oof, 1e-6, 1 - 1e-6)
            / (1 - np.clip(inner_oof, 1e-6, 1 - 1e-6))
        )
        intercept = float(
            sm.GLM(
                y_train,
                np.ones((len(y_train), 1)),
                family=sm.families.Binomial(),
                offset=inner_logit,
            )
            .fit(disp=0)
            .params[0]
        )
        final_model = make_pipeline(chosen_c)
        final_model.fit(X_train, y_train)
        outer_prob = np.clip(
            final_model.predict_proba(values[test])[:, 1], 1e-6, 1 - 1e-6
        )
        outer_logit = np.log(outer_prob / (1 - outer_prob))
        calibrated[test] = 1 / (1 + np.exp(-(outer_logit + intercept)))
        outer_intercepts[str(int(fold))] = intercept
    if np.isnan(calibrated).any():
        raise AssertionError("Intercept-recalibrated OOF predictions are incomplete")
    return calibrated, outer_intercepts


def run_ladder(df: pd.DataFrame):
    manifest = pd.read_csv(config.FOLDS)[["patient_id", "fold", "binary_label"]]
    d = manifest.merge(df, on="patient_id", how="left", validate="one_to_one").set_index("patient_id")
    y = (d["binary_label"] == 0).astype(int).to_numpy()
    folds = d["fold"].astype(int).to_numpy()
    # Group-aware inner selection: cv_group_id keeps the conservatively grouped possible-family
    # pair together in the inner three-fold split as well as the outer folds.
    groups = d["cv_group_id"].astype(str).to_numpy()

    age = d["age_at_first_ct"].astype(float)
    gender = d["gender"].astype(str).str.strip().replace({"Ｍ": "M", "Ｆ": "F"})
    demo = pd.DataFrame(
        {
            "age_at_ct": age,
            "age_missing": age.isna().astype(int),
            "male": gender.eq("M").astype(int),
        },
        index=d.index,
    )
    eva = onehot_block(d, ("eva",))
    eva_cochlea = onehot_block(d, ("eva", "cochlea"))
    anatomy = onehot_block(d, STRUCTS)
    anatomy_demo = pd.concat((anatomy, demo), axis=1)
    # The model labels also derive the bootstrap seeds, so they must not be renamed.
    rungs = {
        "0_prevalence": None,
        "1_demographics": demo,
        "2_eva_only": eva,
        "3_eva_cochlea": eva_cochlea,
        "4_anatomy_PRIMARY": anatomy,
        "5_anatomy_demo": anatomy_demo,
    }

    predictions = {}
    tuning = {}
    for name, X in rungs.items():
        if X is None:
            pred = np.full(len(y), np.nan)
            for fold in sorted(np.unique(folds)):
                test = folds == fold
                pred[test] = y[~test].mean()
            predictions[name] = pred
            tuning[name] = {"model": "outer-training prevalence"}
        else:
            predictions[name], tuning[name] = nested_oof(X, y, folds, groups=groups)

    # Probability-focused sensitivity analyses. AUC remains the primary measure.
    pred_unweighted, tune_unweighted = nested_oof(
        anatomy, y, folds, groups=groups, class_weight=None
    )
    pred_intercept_calibrated, outer_calibration_intercepts = nested_intercept_recalibration(
        anatomy, y, folds, tuning["4_anatomy_PRIMARY"], groups=groups
    )
    anatomy_raw_columns = [
        f"{struct}_{side}" for struct in STRUCTS for side in ("R", "L")
    ]
    complete_anatomy = ~d[anatomy_raw_columns].isna().any(axis=1).to_numpy()
    pred_complete_case, tune_complete_case = nested_oof(
        anatomy.iloc[complete_anatomy], y[complete_anatomy], folds[complete_anatomy],
        groups=groups[complete_anatomy]
    )
    pred_complete_case_full = np.full(len(y), np.nan)
    pred_complete_case_full[complete_anatomy] = pred_complete_case

    result = {
        "meta": {
            "N": int(len(y)),
            "n_SLC26A4": int(y.sum()),
            "n_comparator": int((1 - y).sum()),
            "outer_folds": 5,
            "inner_folds": 3,
            "regularization_selection": (
                "smallest C within one standard error of the best three-fold inner mean AUC"
            ),
            "probability_storage_precision_decimals": 12,
            "bootstrap_B": B_BOOT,
            "seed": SEED,
            "age_imputation": "median fitted separately inside each inner/outer training partition",
            "primary_class_weight": "balanced",
            "uncertainty_scope": "conditional on fixed pooled OOF predictions",
        },
        "ladder": {},
        "outer_fold_tuning": tuning,
    }
    for name, pred in predictions.items():
        result["ladder"][name] = score_metrics(y, pred, "ladder:" + name)

    result["primary_unweighted_logistic_sensitivity"] = {
        **score_metrics(y, pred_unweighted, "primary_unweighted"),
        "outer_fold_tuning": tune_unweighted,
    }
    result["primary_crossfitted_intercept_recalibration"] = {
        **score_metrics(y, pred_intercept_calibrated, "primary_intercept_recalibrated"),
        "outer_training_calibration_intercepts": outer_calibration_intercepts,
        "method": (
            "For each outer fold, generate 3-fold OOF base scores only in the outer-training "
            "partition, fit a calibration intercept with slope fixed at 1, and apply it once "
            "to the untouched outer-test scores."
        ),
    }
    result["primary_complete_case_sensitivity"] = {
        "N": int(complete_anatomy.sum()),
        "n_excluded": int((~complete_anatomy).sum()),
        "excluded_patient_ids": d.index[~complete_anatomy].astype(str).tolist(),
        "excluded_SLC26A4": int(y[~complete_anatomy].sum()),
        **score_metrics(
            y[complete_anatomy], pred_complete_case, "primary_complete_case"
        ),
        "outer_fold_tuning": tune_complete_case,
        "method": (
            "Nested refit after excluding patients missing any of the 12 bilateral anatomy "
            "fields; contrasts with the primary missing-as-normal/all-reference convention."
        ),
    }

    comparisons = [
        ("eva_cochlea_minus_eva", "3_eva_cochlea", "2_eva_only"),
        ("anatomy_minus_eva_cochlea", "4_anatomy_PRIMARY", "3_eva_cochlea"),
        ("anatomy_minus_eva", "4_anatomy_PRIMARY", "2_eva_only"),
        ("anatomy_demo_minus_anatomy", "5_anatomy_demo", "4_anatomy_PRIMARY"),
    ]
    result["paired_comparisons"] = {}
    for label, a, b in comparisons:
        result["paired_comparisons"][label] = {
            "paired_percentile_bootstrap": paired_bootstrap_auc(
                y, predictions[a], predictions[b], label
            ),
            "delong": delong_auc_test(y, predictions[a], predictions[b]),
        }
    delong_p = [
        result["paired_comparisons"][label]["delong"]["p_two_sided"]
        for label, _, _ in comparisons
    ]
    holm_p = multipletests(delong_p, method="holm")[1]
    for (label, _, _), adjusted in zip(comparisons, holm_p):
        result["paired_comparisons"][label]["delong"][
            "holm_adjusted_p_across_four_planned_ladder_contrasts"
        ] = float(adjusted)

    # Fixed-score conditional label permutation.  Shared permutations preserve paired nulls.
    perm_names = [name for name in rungs if name != "0_prevalence"]
    observed = np.asarray([roc_auc_score(y, predictions[name]) for name in perm_names])
    null = np.empty((B_PERM, len(perm_names)), dtype=np.float32)
    rng = np.random.default_rng(SEED)
    for b in range(B_PERM):
        yp = rng.permutation(y)
        for j, name in enumerate(perm_names):
            null[b, j] = roc_auc_score(yp, predictions[name])
    np.savez_compressed(
        config.GRADE_DIR / "permutation_null.npz",
        null_auroc=null,
        rung_names=np.asarray(perm_names),
        observed_auroc=observed,
        seed=np.asarray([SEED]),
        B=np.asarray([B_PERM]),
    )
    perm_summary = {}
    for j, name in enumerate(perm_names):
        extreme = np.abs(null[:, j] - 0.5) >= abs(observed[j] - 0.5) - 1e-12
        perm_summary[name] = {
            "observed_auroc": float(observed[j]),
            "empirical_p_two_sided": float((1 + extreme.sum()) / (B_PERM + 1)),
            "null_mean": float(null[:, j].mean()),
            "null_sd": float(null[:, j].std(ddof=1)),
            "null_percentiles_2_5_50_97_5": [
                float(v) for v in np.percentile(null[:, j], (2.5, 50, 97.5))
            ],
        }
    result["conditional_fixed_score_label_permutation"] = {
        "B": B_PERM,
        "seed": SEED,
        "alternative": "two-sided distance of AUROC from 0.5",
        "scope": (
            "Global fixed-score label-independence test only; no model refitting. "
            "It is not a model-equality or incremental-value test and does not "
            "propagate pipeline-development, training, or checkpoint uncertainty."
        ),
        "null_distribution_file": "permutation_null.npz",
        "rungs": perm_summary,
    }

    pred_df = pd.DataFrame(
        {
            "patient_id": d.index,
            "y_SLC26A4": y,
            "fold": folds,
            **{f"p_{name}": pred for name, pred in predictions.items()},
            "p_4_anatomy_PRIMARY_unweighted": pred_unweighted,
            "p_4_anatomy_PRIMARY_intercept_recalibrated": pred_intercept_calibrated,
            "p_4_anatomy_PRIMARY_complete_case_refit": pred_complete_case_full,
        }
    )
    pred_df.to_csv(config.GRADE_PREDICTIONS, index=False)
    dump_json(config.GRADE_METRICS, result)

    summary_rows = []
    for name, values in result["ladder"].items():
        summary_rows.append({"model": name, "class_weight": "balanced", **values})
    summary_rows.append(
        {
            "model": "4_anatomy_PRIMARY_unweighted",
            "class_weight": "none",
            **{
                k: v
                for k, v in result["primary_unweighted_logistic_sensitivity"].items()
                if k != "outer_fold_tuning"
            },
        }
    )
    summary_rows.append(
        {
            "model": "4_anatomy_PRIMARY_complete_case_refit",
            "class_weight": "balanced",
            **{
                k: v
                for k, v in result["primary_complete_case_sensitivity"].items()
                if k
                not in (
                    "excluded_patient_ids",
                    "outer_fold_tuning",
                    "method",
                )
            },
        }
    )
    summary_rows.append(
        {
            "model": "4_anatomy_PRIMARY_intercept_recalibrated",
            "class_weight": "balanced base + outer-training intercept recalibration",
            **{
                k: v
                for k, v in result["primary_crossfitted_intercept_recalibration"].items()
                if k not in ("outer_training_calibration_intercepts", "method")
            },
        }
    )
    pd.DataFrame(summary_rows).to_csv(config.GRADE_DIR / "summary.csv", index=False)
    return d, y, folds, anatomy, predictions["4_anatomy_PRIMARY"], result


def weighted_auc_tie_correct(y, pred, weights):
    y = np.asarray(y)
    pred = np.asarray(pred)
    weights = np.asarray(weights)
    pos_pred, pos_w = pred[y == 1], weights[y == 1]
    neg_pred, neg_w = pred[y == 0], weights[y == 0]
    comparison = (pos_pred[:, None] > neg_pred[None, :]).astype(float)
    comparison += 0.5 * (pos_pred[:, None] == neg_pred[None, :])
    denominator = pos_w.sum() * neg_w.sum()
    return float((comparison * pos_w[:, None] * neg_w[None, :]).sum() / denominator)


def run_testing_selection_sensitivity(
    df: pd.DataFrame, d_primary: pd.DataFrame, y_primary, p_primary
):
    indexed = df.set_index("patient_id").copy()
    tested = indexed["gene_tested"].astype(bool).astype(int).to_numpy()

    covariates = {}
    for struct in STRUCTS:
        for side in ("R", "L"):
            covariates[f"{struct}_{side}"] = indexed[f"{struct}_{side}"].astype(float)
    covariates["age_at_first_ct"] = indexed["age_at_first_ct"].astype(float)
    covariates["age_missing"] = indexed["age_at_first_ct"].isna().astype(int)
    gender = indexed["gender"].astype(str).str.strip().replace({"Ｍ": "M", "Ｆ": "F"})
    covariates["male"] = gender.eq("M").astype(int)
    covariates["source_malformation"] = indexed["sheet_source"].eq("malformation").astype(int)
    X = pd.DataFrame(covariates, index=indexed.index)

    # Standardized differences use median-filled values only for descriptive balance.
    X_smd = X.copy()
    for column in X_smd:
        X_smd[column] = X_smd[column].fillna(X_smd[column].median())
    smd = {}
    for column in X_smd:
        a = X_smd.loc[tested == 1, column].astype(float)
        b = X_smd.loc[tested == 0, column].astype(float)
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd[column] = {
            "tested_mean": float(a.mean()),
            "untested_mean": float(b.mean()),
            "smd": float((a.mean() - b.mean()) / pooled_sd) if pooled_sd > 0 else 0.0,
        }

    # Cross-fitted probabilities reduce in-sample optimism in the selection model.
    ps = np.full(len(indexed), np.nan)
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, te in splitter.split(X, tested):
        model = make_pipeline(1.0, class_weight=None)
        model.fit(X.iloc[tr], tested[tr])
        ps[te] = model.predict_proba(X.iloc[te])[:, 1]
    if np.isnan(ps).any():
        raise AssertionError("Cross-fitted testing propensities are incomplete")

    primary_ids = d_primary.index.astype(str)
    ps_primary = pd.Series(ps, index=indexed.index).loc[primary_ids].to_numpy()
    marginal = float(tested.mean())
    weights_untrimmed = marginal / np.clip(ps_primary, 1e-6, 1)
    weights = np.clip(weights_untrimmed, 0.2, 5.0)
    auc_weighted = weighted_auc_tie_correct(y_primary, p_primary, weights)

    rng = np.random.default_rng(derived_seed("ipw_auc"))
    boot = []
    for _ in range(B_BOOT):
        idx = rng.integers(0, len(y_primary), len(y_primary))
        if np.unique(y_primary[idx]).size == 2:
            boot.append(weighted_auc_tie_correct(y_primary[idx], p_primary[idx], weights[idx]))

    eva_any = (
        indexed[["eva_R", "eva_L"]].fillna(0).astype(float).ne(0).any(axis=1).astype(int)
    )
    result = {
        "meta": {
            "N_reporting": int(len(indexed)),
            "n_tested": int(tested.sum()),
            "n_untested": int((1 - tested).sum()),
            "n_primary_scores": int(len(y_primary)),
            "propensity_evaluation": "5-fold cross-fitted",
            "propensity_features": list(X.columns),
            "seed": SEED,
        },
        "standardized_differences": smd,
        "testing_propensity": {
            "cross_fitted_auroc": float(roc_auc_score(tested, ps)),
            "tested_percentiles_5_50_95": [
                float(v) for v in np.percentile(ps[tested == 1], (5, 50, 95))
            ],
            "untested_percentiles_5_50_95": [
                float(v) for v in np.percentile(ps[tested == 0], (5, 50, 95))
            ],
        },
        "testing_propensity_weighted_sensitivity": {
            "primary_auroc_unweighted": float(roc_auc_score(y_primary, p_primary)),
            "primary_auroc_weighted_tie_correct": auc_weighted,
            "conditional_percentile_ci": [float(v) for v in np.percentile(boot, (2.5, 97.5))],
            "weight_range_untrimmed": [
                float(weights_untrimmed.min()),
                float(weights_untrimmed.max()),
            ],
            "weight_range_after_0_2_5_trim": [float(weights.min()), float(weights.max())],
            "n_trimmed": int(np.sum(weights != weights_untrimmed)),
            "effective_sample_size": float(weights.sum() ** 2 / np.sum(weights**2)),
            "estimand_and_scope": (
                "Modeled testing-selection sensitivity under MAR: stabilized inverse "
                "P(tested|observed covariates) weights are applied to the "
                f"N={len(y_primary)} analyzed fixed OOF scores. This is not an estimate for "
                "a 'full tested-eligible' population, does not correct the separate "
                "CT/model-eligibility exclusions, and cannot address dependence on "
                "unobserved genotype."
            ),
        },
        "eva_prevalence_descriptive_only": {
            "tested": float(eva_any.to_numpy()[tested == 1].mean()),
            "untested": float(eva_any.to_numpy()[tested == 0].mean()),
        },
    }
    dump_json(config.GRADE_DIR / "testing_uptake.json", result)
    pd.DataFrame(
        {
            "patient_id": primary_ids,
            "y_SLC26A4": y_primary,
            "p_anatomy_PRIMARY": p_primary,
            "cross_fitted_p_tested": ps_primary,
            "stabilized_weight": weights,
        }
    ).to_csv(config.GRADE_DIR / "testing_uptake_inputs.csv", index=False)
    return result


def standardized_mean_difference(a, b):
    denominator = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / denominator) if denominator > 0 else 0.0


def matched_pair_cluster_bootstrap_auc_ci(y, pred, label: str, B=B_BOOT):
    """Conditional AUC CI with each matched pair resampled as a cluster."""
    if len(y) % 2:
        raise ValueError("Matched vectors must contain exactly two rows per pair")
    n_pairs = len(y) // 2
    pair_rows = np.arange(len(y)).reshape(n_pairs, 2)
    rng = np.random.default_rng(derived_seed("matched_pair_auc:" + label))
    estimates = []
    for _ in range(B):
        sampled_pairs = rng.integers(0, n_pairs, n_pairs)
        idx = pair_rows[sampled_pairs].reshape(-1)
        if np.unique(y[idx]).size == 2:
            estimates.append(roc_auc_score(y[idx], pred[idx]))
    return [float(v) for v in np.percentile(estimates, (2.5, 97.5))]


def optimal_age_pairs(patient_ids, age, y, caliper=1.0):
    valid = np.isfinite(age)
    pos = np.where((y == 1) & valid)[0]
    neg = np.where((y == 0) & valid)[0]
    # Stable ID order makes tie resolution reproducible; optimization is not case-order greedy.
    pos = pos[np.argsort(patient_ids[pos])]
    neg = neg[np.argsort(patient_ids[neg])]
    distance = np.abs(age[pos, None] - age[neg][None, :])
    # Lexicographic objective: (1) maximize the number of feasible pairs, then
    # (2) minimize total distance.  One additional unmatched case must cost more
    # than the largest possible change in the sum of *all* feasible distances.
    unmatched_cost = (len(pos) + 1) * (caliper + 1.0)
    invalid_cost = unmatched_cost * (len(pos) + len(neg) + 1)
    real_cost = np.where(distance <= caliper, distance, invalid_cost)
    dummy_cost = np.full((len(pos), len(pos)), unmatched_cost)
    rows, cols = linear_sum_assignment(np.concatenate((real_cost, dummy_cost), axis=1))
    pairs = []
    for row, col in zip(rows, cols):
        if col < len(neg) and distance[row, col] <= caliper:
            pairs.append((int(pos[row]), int(neg[col]), float(distance[row, col])))
    pairs.sort(key=lambda pair: (patient_ids[pair[0]], patient_ids[pair[1]]))
    return pairs


def run_age_and_no_eva_sensitivity(
    d: pd.DataFrame, y, folds, anatomy, p_primary
):
    groups = d["cv_group_id"].astype(str).to_numpy()
    patient_ids = d.index.astype(str).to_numpy()
    age = d["age_at_first_ct"].astype(float).to_numpy()
    male = (
        d["gender"]
        .astype(str)
        .str.strip()
        .replace({"Ｍ": "M", "Ｆ": "F"})
        .eq("M")
        .astype(int)
        .to_numpy()
    )
    complete = np.isfinite(age)
    if np.any(age[complete] < 0):
        raise AssertionError("Negative age in the age sensitivity analysis")

    result = {
        "age_data": {
            "n_complete": int(complete.sum()),
            "n_missing": int((~complete).sum()),
            "minimum": float(age[complete].min()),
            "negative_count": int(np.sum(age[complete] < 0)),
        }
    }
    result["marginal_age_sex"] = {
        "age_only_auroc_complete_case": float(roc_auc_score(y[complete], age[complete])),
        "age_SLC26A4_median": float(np.median(age[complete & (y == 1)])),
        "age_comparator_median": float(np.median(age[complete & (y == 0)])),
        "mannwhitney_p": float(
            stats.mannwhitneyu(age[complete & (y == 1)], age[complete & (y == 0)]).pvalue
        ),
        "male_fraction_SLC26A4": float(male[y == 1].mean()),
        "male_fraction_comparator": float(male[y == 0].mean()),
    }

    # Tertiles are defined among complete ages; no silent age imputation.
    valid_idx = np.where(complete)[0]
    tertile = np.asarray(pd.qcut(age[complete], 3, labels=False, duplicates="drop"))
    strata = {}
    for level in sorted(np.unique(tertile)):
        idx = valid_idx[tertile == level]
        strata[f"tertile_{int(level)}"] = {
            "n": int(len(idx)),
            "n_SLC26A4": int(y[idx].sum()),
            "age_range": [float(age[idx].min()), float(age[idx].max())],
            "anatomy_auroc_fixed_oof": float(roc_auc_score(y[idx], p_primary[idx])),
        }
    result["age_stratified_fixed_oof"] = strata

    pairs = optimal_age_pairs(patient_ids, age, y, caliper=1.0)
    if len(pairs) < 20:
        raise AssertionError("Too few optimal age matches")
    rng = np.random.default_rng(SEED)
    pair_order = rng.permutation(len(pairs))
    pair_fold = np.empty(len(pairs), dtype=int)
    pair_fold[pair_order] = np.arange(len(pairs)) % 5
    matched = []
    matched_folds = []
    pair_rows = []
    for pair_number, ((pos, neg, distance), fold) in enumerate(zip(pairs, pair_fold)):
        matched.extend((pos, neg))
        matched_folds.extend((fold, fold))
        pair_rows.append(
            {
                "pair": pair_number,
                "fold": int(fold),
                "SLC26A4_patient_id": patient_ids[pos],
                "comparator_patient_id": patient_ids[neg],
                "SLC26A4_age": float(age[pos]),
                "comparator_age": float(age[neg]),
                "absolute_age_difference": distance,
            }
        )
    matched = np.asarray(matched)
    matched_folds = np.asarray(matched_folds)
    matched_pred, matched_tuning = nested_oof(
        anatomy.iloc[matched], y[matched], matched_folds, groups=groups[matched]
    )
    pre_smd = standardized_mean_difference(age[complete & (y == 1)], age[complete & (y == 0)])
    post_smd = standardized_mean_difference(
        age[matched][y[matched] == 1], age[matched][y[matched] == 0]
    )
    matched_ci = matched_pair_cluster_bootstrap_auc_ci(
        y[matched], matched_pred, "optimal_age_match"
    )
    result["optimal_age_match"] = {
        "method": "maximum-cardinality minimum-distance 1:1 assignment without replacement",
        "caliper_years": 1.0,
        "n_pairs": int(len(pairs)),
        "n_unmatched_SLC26A4": int(np.sum(complete & (y == 1)) - len(pairs)),
        "age_smd_before": pre_smd,
        "age_smd_after": post_smd,
        "median_absolute_pair_difference": float(
            np.median([pair[2] for pair in pairs])
        ),
        "anatomy_auroc_nested_pair_grouped_folds": float(
            roc_auc_score(y[matched], matched_pred)
        ),
        "auroc_percentile_ci_matched_pair_cluster_bootstrap": matched_ci,
        "ci_scope": (
            "Matched pairs resampled as clusters; conditional on the selected matched set "
            "and fixed nested OOF predictions. Matching-selection uncertainty is not propagated."
        ),
        "outer_fold_tuning": matched_tuning,
    }
    for pair_index, row in enumerate(pair_rows):
        row["SLC26A4_oof_probability"] = float(matched_pred[2 * pair_index])
        row["comparator_oof_probability"] = float(matched_pred[2 * pair_index + 1])
    pd.DataFrame(pair_rows).to_csv(config.GRADE_DIR / "age_matched_pairs.csv", index=False)

    # Association diagnostic is complete-case and does not claim predictive calibration.
    lp = np.log(
        np.clip(p_primary[complete], 1e-6, 1 - 1e-6)
        / (1 - np.clip(p_primary[complete], 1e-6, 1 - 1e-6))
    )
    Z = StandardScaler().fit_transform(np.column_stack((lp, age[complete], male[complete])))
    model = sm.Logit(y[complete], sm.add_constant(Z)).fit(disp=0)
    result["complete_case_adjusted_association"] = {
        "n": int(complete.sum()),
        "standardized_anatomy_logit_coef": float(model.params[1]),
        "anatomy_p": float(model.pvalues[1]),
        "standardized_age_coef": float(model.params[2]),
        "age_p": float(model.pvalues[2]),
        "standardized_sex_coef": float(model.params[3]),
        "sex_p": float(model.pvalues[3]),
        "scope": "post-OOF association diagnostic, not a nested predictive model",
    }

    no_eva = onehot_block(d, tuple(s for s in STRUCTS if s != "eva"))
    p_no_eva, tuning_no_eva = nested_oof(no_eva, y, folds, groups=groups)
    result["anatomy_without_EVA"] = {
        **score_metrics(y, p_no_eva, "anatomy_without_EVA"),
        "outer_fold_tuning": tuning_no_eva,
        "scope": "grade model refitted on the same patients with all structures except EVA",
    }
    dump_json(config.AGE_AND_WITHOUT_EVA, result)
    return result


def run_forward_selection(d: pd.DataFrame, y, folds):
    groups = d["cv_group_id"].astype(str).to_numpy()
    blocks = {struct: onehot_block(d, (struct,)) for struct in STRUCTS}
    all_anatomy = pd.concat([blocks[s] for s in STRUCTS], axis=1)

    # Fixed candidate estimates are valid as conditional OOF estimates, but comparing all
    # six and promoting the winner remains post hoc.
    fixed = {}
    for struct in STRUCTS:
        pred, chosen_c = nested_oof(blocks[struct], y, folds, groups=groups)
        fixed[struct] = {
            **score_metrics(y, pred, "single_structure:" + struct),
            "outer_fold_tuning": chosen_c,
        }
    fixed_ev, fixed_ev_c = nested_oof(
        pd.concat((blocks["eva"], blocks["vestibule"]), axis=1), y, folds, groups=groups
    )

    # Entire greedy structure search occurs inside each outer-training partition.
    oof_by_k = {k: np.full(len(y), np.nan) for k in range(1, len(STRUCTS) + 1)}
    fold_paths = {}
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test
        chosen = []
        remaining = set(STRUCTS)
        path = []
        for k in range(1, len(STRUCTS) + 1):
            candidates = []
            for candidate in sorted(remaining):
                subset = chosen + [candidate]
                X = pd.concat([blocks[s] for s in subset], axis=1).to_numpy()
                best_c, inner_auc, _ = tune_c(X[train], y[train], groups=groups[train])
                candidates.append((inner_auc, candidate, best_c))
            # Deterministic: highest AUC, then lexical structure name.
            candidates.sort(key=lambda item: (-item[0], item[1]))
            inner_auc, selected, best_c = candidates[0]
            chosen.append(selected)
            remaining.remove(selected)
            X_selected = pd.concat([blocks[s] for s in chosen], axis=1).to_numpy()
            model = make_pipeline(best_c)
            model.fit(X_selected[train], y[train])
            oof_by_k[k][test] = model.predict_proba(X_selected[test])[:, 1]
            path.append(
                {
                    "k": k,
                    "added": selected,
                    "selected_structures": chosen.copy(),
                    "inner_mean_auc": float(inner_auc),
                    "C": float(best_c),
                }
            )
        fold_paths[str(int(fold))] = path

    nested_curve = {}
    for k, pred in oof_by_k.items():
        if np.isnan(pred).any():
            raise AssertionError(f"Nested forward predictions incomplete at k={k}")
        selection_frequency = {
            struct: int(
                sum(struct in fold_paths[str(int(fold))][k - 1]["selected_structures"] for fold in np.unique(folds))
            )
            for struct in STRUCTS
        }
        nested_curve[str(k)] = {
            **score_metrics(y, pred, f"nested_forward_k{k}"),
            "selection_frequency_over_5_outer_folds": selection_frequency,
        }

    # L1 group-selection frequency with C tuned inside each outer-training set.
    l1_frequency = {struct: 0 for struct in STRUCTS}
    l1_details = {}
    col_to_struct = {
        column: next(s for s in STRUCTS if column.startswith(s + "_"))
        for column in all_anatomy.columns
    }
    for fold in sorted(np.unique(folds)):
        train = folds != fold
        best_c, inner_auc, _ = tune_c(
            all_anatomy.to_numpy()[train], y[train], groups=groups[train], penalty="l1"
        )
        model = make_pipeline(best_c, penalty="l1")
        model.fit(all_anatomy.to_numpy()[train], y[train])
        coef = model.named_steps["model"].coef_[0]
        picked = {
            col_to_struct[column]
            for column, value in zip(all_anatomy.columns, coef)
            if abs(value) > 1e-8
        }
        for struct in picked:
            l1_frequency[struct] += 1
        l1_details[str(int(fold))] = {
            "C": best_c,
            "inner_mean_auc": inner_auc,
            "selected_structures": sorted(picked),
        }

    result = {
        "meta": {
            "N": int(len(y)),
            "outer_folds": 5,
            "inner_folds": 3,
            "selection_rule": "greedy forward structure selection entirely within outer training data",
        },
        "single_structure_fixed_candidate": fixed,
        "posthoc_fixed_EVA_plus_vestibule": {
            **score_metrics(y, fixed_ev, "posthoc_fixed_EVA_plus_vestibule"),
            "outer_fold_tuning": fixed_ev_c,
        },
        "nested_forward_curve": nested_curve,
        "outer_fold_selection_paths": fold_paths,
        "nested_tuned_l1_structure_frequency": l1_frequency,
        "nested_tuned_l1_details": l1_details,
    }
    dump_json(config.FORWARD_SELECTION, result)
    pd.DataFrame(
        {
            "patient_id": d.index,
            "y_SLC26A4": y,
            "fold": folds,
            **{f"p_nested_forward_k{k}": p for k, p in oof_by_k.items()},
            "p_posthoc_fixed_EVA_plus_vestibule": fixed_ev,
        }
    ).to_csv(config.GRADE_DIR / "forward_selection_predictions.csv", index=False)
    rows = []
    for k, values in nested_curve.items():
        rows.append(
            {
                "k": int(k),
                "auroc": values["auroc"],
                "ci_low": values["auroc_percentile_ci"][0],
                "ci_high": values["auroc_percentile_ci"][1],
                **{
                    "selected_" + s + "_folds": values[
                        "selection_frequency_over_5_outer_folds"
                    ][s]
                    for s in STRUCTS
                },
            }
        )
    pd.DataFrame(rows).to_csv(config.GRADE_DIR / "forward_selection_curve.csv", index=False)
    return result


def run_cohort_flow(df: pd.DataFrame):
    manifest = pd.read_csv(config.FOLDS)[["patient_id", "fold", "binary_label", "original_class"]]
    primary = set(manifest["patient_id"].astype(str))
    tested = df.loc[df["gene_tested"].astype(bool)].copy()
    omitted = tested.loc[~tested["patient_id"].isin(primary)].copy()

    # Every tested patient outside the analysis cohort gets exactly one reason. The categories
    # keep design exclusions apart from image failures.
    external_site = {
        "<PATIENT_ID_34>",
        "<PATIENT_ID_35>",
        "<PATIENT_ID_36>",
        "<PATIENT_ID_37>",
        "<PATIENT_ID_38>",
        "<PATIENT_ID_39>",
        "<PATIENT_ID_40>",
        "<PATIENT_ID_42>",
    }
    insufficient_axial = {"<PATIENT_ID_30>"}
    no_eligible_coronal = {
        "<PATIENT_ID_29>",
        "<PATIENT_ID_31>",
        "<PATIENT_ID_32>",
        "<PATIENT_ID_33>",
        "<PATIENT_ID_41>",
    }
    coronal_not_cropped = {"<PATIENT_ID_46>"}
    fov_outlier = {"<PATIENT_ID_25>", "<PATIENT_ID_26>", "<PATIENT_ID_23>", "<PATIENT_ID_24>"}
    missing_source_volume = {"<PATIENT_ID_28>", "<PATIENT_ID_47>"}
    reason_sets = {
        "added_after_cohort_list": ADDED_AFTER_COHORT_LIST,
        "external_site_CT_excluded": external_site,
        "no_eligible_coronal_series": no_eligible_coronal,
        "coronal_series_not_cropped": coronal_not_cropped,
        "insufficient_axial_coverage": insufficient_axial,
        "field_of_view_outlier": fov_outlier,
        "missing_source_volume": missing_source_volume,
    }
    all_classified = set().union(*reason_sets.values())
    omitted_ids = set(omitted["patient_id"].astype(str))
    if all_classified != omitted_ids:
        raise AssertionError(
            f"Cohort exclusion lists do not match the omitted patients; "
            f"unclassified={sorted(omitted_ids-all_classified)}, "
            f"extra={sorted(all_classified-omitted_ids)}"
        )
    if sum(len(values) for values in reason_sets.values()) != len(all_classified):
        raise AssertionError("Cohort reason categories overlap")

    source_inventory = pd.read_csv(config.PNG_INVENTORY)
    rows = []
    for record in omitted.itertuples(index=False):
        pid = str(record.patient_id)
        reason = next(key for key, values in reason_sets.items() if pid in values)
        source_rows = source_inventory.loc[source_inventory["patient_id"].eq(pid)]
        rows.append(
            {
                "patient_id": pid,
                "reason_code": reason,
                "gene_category": record.gene_category,
                "gene_class_3": record.gene_class_3,
                "sheet_source": record.sheet_source,
                "has_axial_npz_flag": bool(record.has_npz),
                "source_coronal_scan_count": int(len(source_rows)),
                "source_coronal_png_count": int(source_rows["n_pngs"].fillna(0).sum()),
                "source_coronal_subfolders": ";".join(
                    source_rows["coronal_subfolder"].dropna().astype(str).tolist()
                ),
            }
        )
    exclusion_df = pd.DataFrame(rows).sort_values(["reason_code", "patient_id"])
    exclusion_df.to_csv(config.COHORT_EXCLUSIONS, index=False)

    counts = exclusion_df["reason_code"].value_counts().to_dict()
    flow = {
        "reporting_cohort": int(len(df)),
        "recorded_genetic_test": int(df["gene_tested"].sum()),
        "untested": int((~df["gene_tested"].astype(bool)).sum()),
        "excluded_after_recorded_test": int(len(exclusion_df)),
        "exclusion_counts": {key: int(counts.get(key, 0)) for key in reason_sets},
        "primary_analysis": int(len(manifest)),
        "primary_SLC26A4": int((manifest["binary_label"] == 0).sum()),
        "primary_comparators": int((manifest["binary_label"] == 1).sum()),
        "primary_negative": int((manifest["original_class"] == "negative").sum()),
        "primary_other_known_gene": int(
            (manifest["original_class"] == "other_known_gene").sum()
        ),
        "evaluation_note": (
            "All predictions use five outer folds. Inner selection is model-specific: "
            "coded logistic models use three-fold inner CV; the CNNs use a stratified 20% "
            "inner validation split."
        ),
        "coronal_series_not_cropped_source": {
            "source_inventory_coronal_pngs": int(
                source_inventory.loc[
                    source_inventory["patient_id"].eq("<PATIENT_ID_46>"), "n_pngs"
                ]
                .fillna(0)
                .sum()
            ),
        },
    }
    n_excluded = sum(flow["exclusion_counts"].values())
    if flow["recorded_genetic_test"] - n_excluded != flow["primary_analysis"]:
        raise AssertionError(
            f"Flow arithmetic failed: {flow['recorded_genetic_test']} tested - {n_excluded} "
            f"excluded != {flow['primary_analysis']} in the analysis cohort"
        )
    dump_json(config.COHORT_FLOW, flow)
    resolution = json.loads(config.DUPLICATE_RESOLUTION.read_text())

    # Flow figure.
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge="#334155", fontsize=10, weight="normal"):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.09",
            linewidth=1.3,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#475569"))

    people_text = (
        f"Source records N = {resolution['source_record_ids']}\n"
        f"identity-resolved to N = {flow['reporting_cohort']} people\n"
        f"({resolution['collapsed_alias_pairs']} alias groups collapsed)"
    )
    box(2.9, 8.35, 4.2, 1.0, people_text, "#E0F2FE", weight="bold", fontsize=9.2)
    box(3.25, 6.65, 3.5, 1.0, f"Recorded genetic-test classification\nN = {flow['recorded_genetic_test']}", "#DBEAFE", weight="bold")
    box(7.35, 6.65, 2.25, 1.0, f"Untested\nN = {flow['untested']}", "#F1F5F9")
    arrow(5.0, 8.35, 5.0, 7.65)
    arrow(6.8, 8.4, 8.35, 7.68)

    exclusion_labels = (
        ("added_after_cohort_list", "added after cohort list (coronal data present)"),
        ("external_site_CT_excluded", "external-site CTs"),
        ("no_eligible_coronal_series", "no eligible coronal series"),
        ("coronal_series_not_cropped", "source series present; coronal series not cropped"),
        ("field_of_view_outlier", "field-of-view outliers"),
        ("missing_source_volume", "missing source volumes"),
        ("insufficient_axial_coverage", "insufficient axial coverage"),
    )
    exclusion_text = "\n".join(
        [f"Excluded after recorded test, N = {flow['excluded_after_recorded_test']}"]
        + [f"{flow['exclusion_counts'][key]} {label}" for key, label in exclusion_labels]
    )
    box(0.25, 3.45, 3.8, 2.45, exclusion_text, "#FFF7ED", edge="#C2410C", fontsize=8.2)
    cohort_text = (
        f"Analysis cohort\nN = {flow['primary_analysis']}\n"
        f"{flow['primary_SLC26A4']} SLC26A4; {flow['primary_comparators']} tested comparators"
    )
    box(5.05, 3.85, 4.25, 1.75, cohort_text, "#DCFCE7", edge="#15803D", fontsize=11, weight="bold")
    arrow(5.0, 6.65, 2.15, 5.8)
    arrow(5.0, 6.65, 7.15, 5.6)
    evaluation_text = (
        f"Nested evaluation\n{manifest['fold'].nunique()} group-aware outer folds\n"
        "group-aware inner selection"
    )
    box(5.05, 1.25, 4.25, 1.45, evaluation_text, "#F0FDFA", edge="#0F766E", fontsize=9.5)
    arrow(7.15, 3.85, 7.15, 2.7)
    ax.text(
        5,
        0.35,
        f"Flow from the person table, the fold table (N = {flow['primary_analysis']}), "
        "the exclusion lists and the CT file inventory.",
        ha="center",
        va="center",
        fontsize=8,
        color="#475569",
    )
    fig.tight_layout()
    fig.savefig(config.COHORT_DIR / "flow.pdf", bbox_inches="tight")
    fig.savefig(config.COHORT_DIR / "flow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return flow


def main():
    config.ensure_dirs()
    df, age_correction = load_corrected_patient_data()
    dump_json(config.AGE_CORRECTION, age_correction)
    cohort = run_cohort_flow(df)
    d, y, folds, anatomy, p_primary, ladder = run_ladder(df)
    ipw = run_testing_selection_sensitivity(df, d, y, p_primary)
    age = run_age_and_no_eva_sensitivity(d, y, folds, anatomy, p_primary)
    forward = run_forward_selection(d, y, folds)

    headline = {
        "cohort": {
            "reporting_N": cohort["reporting_cohort"],
            "tested_N": cohort["recorded_genetic_test"],
            "primary_N": cohort["primary_analysis"],
            "added_after_cohort_list_N": cohort["exclusion_counts"]["added_after_cohort_list"],
            "exclusion_counts": cohort["exclusion_counts"],
        },
        "age_correction": age_correction["corrections"][0],
        "primary_balanced": ladder["ladder"]["4_anatomy_PRIMARY"],
        "primary_unweighted_sensitivity": {
            key: ladder["primary_unweighted_logistic_sensitivity"][key]
            for key in (
                "auroc",
                "log_loss",
                "brier",
                "calibration_in_the_large",
                "calibration_slope",
                "calibration_model_intercept",
            )
        },
        "primary_crossfitted_intercept_recalibration": {
            key: ladder["primary_crossfitted_intercept_recalibration"][key]
            for key in (
                "auroc",
                "log_loss",
                "brier",
                "calibration_in_the_large",
                "calibration_slope",
                "calibration_model_intercept",
            )
        },
        "primary_complete_case_sensitivity": {
            key: ladder["primary_complete_case_sensitivity"][key]
            for key in (
                "N",
                "n_excluded",
                "excluded_SLC26A4",
                "auroc",
                "auroc_percentile_ci",
                "log_loss",
                "calibration_in_the_large",
                "calibration_slope",
            )
        },
        "primary_fixed_score_permutation_p_two_sided": ladder[
            "conditional_fixed_score_label_permutation"
        ]["rungs"]["4_anatomy_PRIMARY"]["empirical_p_two_sided"],
        "testing_selection_weighted_auroc": ipw[
            "testing_propensity_weighted_sensitivity"
        ]["primary_auroc_weighted_tie_correct"],
        "optimal_age_match": age["optimal_age_match"],
        "nested_forward_k2": forward["nested_forward_curve"]["2"],
        "posthoc_fixed_EVA_plus_vestibule": forward[
            "posthoc_fixed_EVA_plus_vestibule"
        ],
    }
    dump_json(config.GRADE_DIR / "headlines.json", headline)

    print("Grade-model analysis complete")
    print(f"  primary balanced AUC: {headline['primary_balanced']['auroc']:.4f}")
    print(
        "  primary calibration-in-the-large / slope: "
        f"{headline['primary_balanced']['calibration_in_the_large']:.3f} / "
        f"{headline['primary_balanced']['calibration_slope']:.3f}"
    )
    print(
        "  primary unweighted log-loss / calibration-in-the-large: "
        f"{headline['primary_unweighted_sensitivity']['log_loss']:.3f} / "
        f"{headline['primary_unweighted_sensitivity']['calibration_in_the_large']:.3f}"
    )
    print(
        "  tie-correct testing-propensity weighted AUC: "
        f"{headline['testing_selection_weighted_auroc']:.4f}"
    )
    print(
        "  nested forward k=2 AUC: "
        f"{headline['nested_forward_k2']['auroc']:.4f}"
    )


if __name__ == "__main__":
    main()
