"""Test whether the saved CNN scores add information to the grade-model score.

The script reads three saved patient-level out-of-fold scores: the nested grade-model score
(grade_models.analysis) and the single-task and multitask CNN scores (cnn.train). The two CNN
pipelines are identical except for the six auxiliary anatomy heads and their losses.

All analyses are conditional on these fixed score vectors; no CNN is retrained and no
training or checkpoint-selection uncertainty is propagated. The stacks are fitted on the saved
out-of-fold scores with the same folds and are descriptive.

Terms:
- conditional meta-map: a logistic regression of the label on the logits of saved scores,
  fitted on the other folds and applied to each held-out fold;
- score-space stack: the meta-map on the grade-model logit and one CNN logit. The grade-only
  meta-map uses the grade-model logit alone and is the matched reference for the stack.

The analyses are:

1. global score-label association, by shuffling labels against each fixed raw score;
2. paired metric changes of the score-space stack, compared with the grade-model score itself
   and, separately, with the grade-only meta-map;
3. association of each CNN logit after the grade-model logit, by a logistic likelihood-ratio
   test, an efficient-score test and a partial rank (Spearman) correlation, the last two also
   tested by permuting the CNN logit within grade-score deciles; and
4. paired bootstrap, DeLong, and paired model-swap randomization tests of the fixed scores.

Usage: python -m cnn.added_value
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import config

SEED = 20260702
B_BOOT = 10_000
B_LABEL = 20_000
B_SWAP = 20_000
B_CONDITIONAL = 20_000
EPS = 1e-6


def _json_float(x: float | np.floating) -> float:
    """Return an ordinary finite Python float for strict JSON output."""
    value = float(x)
    if not np.isfinite(value):
        raise ValueError(f"non-finite result: {value}")
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))


def _auc_rank(y: np.ndarray, score: np.ndarray) -> float:
    """AUROC via the Mann-Whitney rank statistic, including average ranks for ties."""
    y = np.asarray(y, dtype=np.int8)
    m = int(y.sum())
    n = int(len(y) - m)
    if m == 0 or n == 0:
        return float("nan")
    ranks = stats.rankdata(np.asarray(score), method="average")
    return float((ranks[y == 1].sum() - m * (m + 1) / 2.0) / (m * n))


def _log_score_each(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def load_cnn_predictions(mode: str, seed_run: int = 1) -> pd.DataFrame:
    """Held-out SLC26A4 scores of one CNN ("single" or "multitask") and seed run, indexed by patient_id.

    Seed run 1 is the pooled file of cnn.train; seed runs 2 and 3 are the fold files of
    cnn.seed_runs, concatenated over the folds.
    """
    if seed_run == 1:
        df = pd.read_csv(config.cnn_predictions(mode))
    else:
        df = pd.concat([pd.read_csv(config.cnn_fold_predictions(mode, f, seed_run))
                        for f in range(config.N_FOLDS)], ignore_index=True)
    return df.set_index("patient_id")


def _load_aligned() -> tuple[pd.DataFrame, dict]:
    # All three files share one orientation: the positive class is SLC26A4, y_SLC26A4 == 1
    # for SLC26A4, and the probability column is P(SLC26A4).
    paths = {
        "coded": config.GRADE_PREDICTIONS,
        "single": config.cnn_predictions("single"),
        "multitask": config.cnn_predictions("multitask"),
    }
    coded = pd.read_csv(paths["coded"])[
        ["patient_id", "fold", "y_SLC26A4", "p_4_anatomy_PRIMARY"]
    ].rename(columns={"p_4_anatomy_PRIMARY": "p_coded_primary"})

    single = pd.read_csv(paths["single"])[
        ["patient_id", "fold", "y_SLC26A4", "p_ent_SLC26A4"]
    ].rename(columns={"fold": "fold_single", "y_SLC26A4": "y_single_positive",
                      "p_ent_SLC26A4": "p_cnn_single"})

    multitask = pd.read_csv(paths["multitask"])[
        ["patient_id", "fold", "y_SLC26A4", "p_ent_SLC26A4"]
    ].rename(columns={"fold": "fold_multitask", "y_SLC26A4": "y_multitask_positive",
                      "p_ent_SLC26A4": "p_cnn_multitask"})

    merged = coded.merge(single, on="patient_id", validate="one_to_one").merge(
        multitask, on="patient_id", validate="one_to_one"
    )
    if len(merged) != 229 or not merged["patient_id"].is_unique:
        raise AssertionError(f"expected 229 unique aligned patients, got {len(merged)}")
    if not (
        (merged["y_SLC26A4"] == merged["y_single_positive"]).all()
        and (merged["y_SLC26A4"] == merged["y_multitask_positive"]).all()
    ):
        raise AssertionError("positive-class orientation differs across OOF inputs")
    if not (
        (merged["fold"] == merged["fold_single"]).all()
        and (merged["fold"] == merged["fold_multitask"]).all()
    ):
        raise AssertionError("outer-fold assignments differ across OOF inputs")
    for col in ["p_coded_primary", "p_cnn_single", "p_cnn_multitask"]:
        if merged[col].isna().any() or not merged[col].between(0, 1).all():
            raise AssertionError(f"invalid probability column: {col}")

    merged = merged.drop(
        columns=[
            "fold_single",
            "fold_multitask",
            "y_single_positive",
            "y_multitask_positive",
        ]
    ).sort_values(["fold", "patient_id"], kind="stable").reset_index(drop=True)
    provenance = {
        name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
    }
    return merged, provenance


def _conditional_meta_map(
    y: np.ndarray,
    folds: np.ndarray,
    p_coded: np.ndarray,
    p_cnn: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Cross-fit the score-space stack and the grade-only meta-map over the fixed logits."""
    lc, ln = _logit(p_coded), _logit(p_cnn)
    full_oof = np.full(len(y), np.nan)
    reduced_oof = np.full(len(y), np.nan)
    coefficients: list[dict] = []
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        train = ~test

        x_full = np.column_stack([lc, ln])
        scale_full = StandardScaler().fit(x_full[train])
        x_full_z = scale_full.transform(x_full)
        full = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(
            x_full_z[train], y[train]
        )
        full_oof[test] = full.predict_proba(x_full_z[test])[:, 1]

        scale_reduced = StandardScaler().fit(lc[train, None])
        x_reduced_z = scale_reduced.transform(lc[:, None])
        reduced = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000).fit(
            x_reduced_z[train], y[train]
        )
        reduced_oof[test] = reduced.predict_proba(x_reduced_z[test])[:, 1]
        coefficients.append(
            {
                "fold": int(fold),
                "coded_standardized_logit": _json_float(full.coef_[0, 0]),
                "cnn_standardized_logit": _json_float(full.coef_[0, 1]),
                "intercept": _json_float(full.intercept_[0]),
            }
        )
    if np.isnan(full_oof).any() or np.isnan(reduced_oof).any():
        raise AssertionError("conditional meta-map left missing predictions")
    return full_oof, reduced_oof, coefficients


def _delong_pair(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
    """Paired DeLong U-statistic covariance for AUROC(a)-AUROC(b)."""
    pos, neg = y == 1, y == 0
    score_matrix = np.vstack([a, b])
    aucs, v10, v01 = [], [], []
    for score in score_matrix:
        sp, sn = score[pos], score[neg]
        placement = (sp[:, None] > sn[None, :]).astype(float)
        placement += 0.5 * (sp[:, None] == sn[None, :])
        aucs.append(placement.mean())
        v10.append(placement.mean(axis=1))
        v01.append(placement.mean(axis=0))
    v10 = np.asarray(v10)
    v01 = np.asarray(v01)
    cov = np.cov(v10, ddof=1) / pos.sum() + np.cov(v01, ddof=1) / neg.sum()
    contrast = np.array([1.0, -1.0])
    variance = max(float(contrast @ cov @ contrast), 0.0)
    se = np.sqrt(variance)
    delta = float(aucs[0] - aucs[1])
    z = delta / se if se > 0 else (0.0 if delta == 0 else np.sign(delta) * np.inf)
    p = 2.0 * stats.norm.sf(abs(z))
    return {
        "delta": delta,
        "se": _json_float(se),
        "ci_lo_normal": _json_float(delta - 1.959963984540054 * se),
        "ci_hi_normal": _json_float(delta + 1.959963984540054 * se),
        "z": _json_float(z),
        "p_two_sided": _json_float(p),
        "conditioning": "fixed aligned patient-level score vectors",
    }


def _bootstrap_distributions(
    y: np.ndarray, scores: dict[str, np.ndarray], rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    n = len(y)
    indices = rng.integers(0, n, size=(B_BOOT, n), dtype=np.int32)
    auc_boot = {name: np.full(B_BOOT, np.nan) for name in scores}
    logscore_boot: dict[str, np.ndarray] = {}
    for name, score in scores.items():
        for i, sample in enumerate(indices):
            auc_boot[name][i] = _auc_rank(y[sample], score[sample])
        each = _log_score_each(y, score)
        logscore_boot[name] = each[indices].mean(axis=1)
    return auc_boot, logscore_boot


def _metric_summary(
    y: np.ndarray,
    p: np.ndarray,
    auc_boot: np.ndarray,
    logscore_boot: np.ndarray,
) -> dict:
    valid_auc = auc_boot[np.isfinite(auc_boot)]
    return {
        "auroc": _json_float(roc_auc_score(y, p)),
        "auroc_percentile_ci": [
            _json_float(np.percentile(valid_auc, 2.5)),
            _json_float(np.percentile(valid_auc, 97.5)),
        ],
        "average_precision": _json_float(average_precision_score(y, p)),
        "log_loss": _json_float(log_loss(y, _clip(p))),
        "log_loss_percentile_ci": [
            _json_float(-np.percentile(logscore_boot, 97.5)),
            _json_float(-np.percentile(logscore_boot, 2.5)),
        ],
        "brier": _json_float(brier_score_loss(y, p)),
    }


def _bootstrap_pair(
    auc_a: np.ndarray,
    auc_b: np.ndarray,
    logscore_a: np.ndarray,
    logscore_b: np.ndarray,
    obs_auc_delta: float,
    obs_logscore_delta: float,
) -> dict:
    auc_delta = auc_a - auc_b
    auc_delta = auc_delta[np.isfinite(auc_delta)]
    logscore_delta = logscore_a - logscore_b
    auc_qlo, auc_qhi = np.percentile(auc_delta, [2.5, 97.5])
    log_qlo, log_qhi = np.percentile(logscore_delta, [2.5, 97.5])
    return {
        "auroc": {
            "delta": _json_float(obs_auc_delta),
            "percentile_ci": [
                _json_float(auc_qlo),
                _json_float(auc_qhi),
            ],
            "basic_ci_sensitivity": [
                _json_float(2 * obs_auc_delta - auc_qhi),
                _json_float(2 * obs_auc_delta - auc_qlo),
            ],
            "bootstrap_mean": _json_float(np.mean(auc_delta)),
            "bootstrap_bias_vs_observed": _json_float(np.mean(auc_delta) - obs_auc_delta),
            "two_sided_tail_fraction_descriptive_not_p_value": _json_float(
                min(1.0, 2.0 * min(np.mean(auc_delta <= 0), np.mean(auc_delta >= 0)))
            ),
        },
        "mean_log_score": {
            "delta_positive_favors_first": _json_float(obs_logscore_delta),
            "percentile_ci": [
                _json_float(log_qlo),
                _json_float(log_qhi),
            ],
            "basic_ci_sensitivity": [
                _json_float(2 * obs_logscore_delta - log_qhi),
                _json_float(2 * obs_logscore_delta - log_qlo),
            ],
            "two_sided_tail_fraction_descriptive_not_p_value": _json_float(
                min(
                    1.0,
                    2.0 * min(
                        np.mean(logscore_delta <= 0), np.mean(logscore_delta >= 0)
                    ),
                )
            ),
        },
        "conditioning": (
            "patient bootstrap conditional on the saved raw and derived score vectors; "
            "meta-map and base models were not refit"
        ),
    }


def _model_swap_test(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Paired randomization by independently swapping the two model scores per patient."""
    masks = rng.integers(0, 2, size=(B_SWAP, len(y)), dtype=np.int8).astype(bool)
    auc_null = np.empty(B_SWAP)
    for i, mask in enumerate(masks):
        ap = np.where(mask, b, a)
        bp = np.where(mask, a, b)
        auc_null[i] = _auc_rank(y, ap) - _auc_rank(y, bp)
    each_delta = _log_score_each(y, a) - _log_score_each(y, b)
    signs = np.where(masks, -1.0, 1.0)
    logscore_null = signs @ each_delta / len(y)
    obs_auc = _auc_rank(y, a) - _auc_rank(y, b)
    obs_logscore = each_delta.mean()

    def empirical(null: np.ndarray, obs: float) -> float:
        return float((1 + np.sum(np.abs(null) >= abs(obs))) / (len(null) + 1))

    result = {
        "auroc": {
            "delta": _json_float(obs_auc),
            "p_two_sided": empirical(auc_null, obs_auc),
            "null_quantiles": [
                _json_float(np.percentile(auc_null, 2.5)),
                _json_float(np.percentile(auc_null, 50)),
                _json_float(np.percentile(auc_null, 97.5)),
            ],
        },
        "mean_log_score": {
            "delta_positive_favors_first": _json_float(obs_logscore),
            "p_two_sided": empirical(logscore_null, obs_logscore),
            "null_quantiles": [
                _json_float(np.percentile(logscore_null, 2.5)),
                _json_float(np.percentile(logscore_null, 50)),
                _json_float(np.percentile(logscore_null, 97.5)),
            ],
        },
        "null": (
            "sharp paired exchangeability null: for each patient, the two fixed model "
            "scores may be swapped; labels and all fitted scores remain fixed"
        ),
        "n_randomizations": B_SWAP,
    }
    return result, {"auc": auc_null, "mean_log_score": logscore_null}


def _global_label_permutation(
    y: np.ndarray,
    raw_scores: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Shuffle labels against fixed raw OOF scores; no base model is refit."""
    y_perm = np.empty((B_LABEL, len(y)), dtype=np.int8)
    for i in range(B_LABEL):
        y_perm[i] = rng.permutation(y)
    m = int(y.sum())
    n = len(y) - m
    const = m * (m + 1) / 2.0
    results, nulls = {}, {}
    for name, score in raw_scores.items():
        ranks = stats.rankdata(score, method="average")
        null_auc = (y_perm @ ranks - const) / (m * n)
        obs = _auc_rank(y, score)
        p_positive = (1 + np.sum(null_auc >= obs)) / (B_LABEL + 1)
        p_two = (1 + np.sum(np.abs(null_auc - 0.5) >= abs(obs - 0.5))) / (
            B_LABEL + 1
        )
        results[name] = {
            "observed_auroc": _json_float(obs),
            "p_one_sided_positive_association": _json_float(p_positive),
            "p_two_sided_any_association": _json_float(p_two),
            "null_auc_quantiles": [
                _json_float(np.percentile(null_auc, 2.5)),
                _json_float(np.percentile(null_auc, 50)),
                _json_float(np.percentile(null_auc, 97.5)),
            ],
        }
        nulls[name] = null_auc
    return (
        {
            "scores": results,
            "n_label_permutations": B_LABEL,
            "null": "global independence of the fixed score and shuffled patient labels",
        },
        nulls,
    )


def _partial_rank_stat(y: np.ndarray, coded: np.ndarray, cnn: np.ndarray) -> float:
    ry = stats.rankdata(y, method="average")
    rc = stats.rankdata(coded, method="average")
    rx = stats.rankdata(cnn, method="average")
    design = np.column_stack([np.ones(len(y)), rc])
    yr = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    xr = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    return float(np.corrcoef(yr, xr)[0, 1])


def _conditional_association(
    y: np.ndarray,
    p_coded: np.ndarray,
    p_cnn: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    lc = _logit(p_coded)
    ln = _logit(p_cnn)
    zc = (lc - lc.mean()) / lc.std(ddof=0)
    zn = (ln - ln.mean()) / ln.std(ddof=0)
    reduced_x = sm.add_constant(zc)
    full_x = sm.add_constant(np.column_stack([zc, zn]))
    reduced = sm.Logit(y, reduced_x).fit(disp=0, maxiter=5000)
    full = sm.Logit(y, full_x).fit(disp=0, maxiter=5000)
    lr = max(0.0, 2.0 * (full.llf - reduced.llf))

    # Efficient score for the CNN coefficient after projecting it off the reduced-model
    # nuisance space under logistic-information weights.
    p0 = np.asarray(reduced.predict(reduced_x))
    weight = p0 * (1.0 - p0)

    def efficient_z(x: np.ndarray) -> float:
        xtwx = reduced_x.T @ (weight[:, None] * reduced_x)
        xtwz = reduced_x.T @ (weight * x)
        projection = reduced_x @ np.linalg.solve(xtwx, xtwz)
        x_residual = x - projection
        information = float(np.sum(weight * x_residual**2))
        return float(x_residual @ (y - p0) / np.sqrt(information))

    observed_efficient_z = efficient_z(zn)
    observed_partial = _partial_rank_stat(y, lc, ln)

    # Approximate conditional randomization: shuffle the CNN logit only among patients in
    # the same coded-risk decile.  This preserves the observed y--coded relation and the
    # coarse CNN--coded relation while breaking within-stratum CNN--outcome ordering.
    strata = np.asarray(pd.qcut(lc, q=10, labels=False, duplicates="drop"), dtype=int)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    efficient_null = np.empty(B_CONDITIONAL)
    partial_null = np.empty(B_CONDITIONAL)
    for i in range(B_CONDITIONAL):
        perm = np.arange(len(y))
        for group in groups:
            perm[group] = rng.permutation(group)
        efficient_null[i] = efficient_z(zn[perm])
        partial_null[i] = _partial_rank_stat(y, lc, ln[perm])

    def empirical(null: np.ndarray, obs: float) -> float:
        return float((1 + np.sum(np.abs(null) >= abs(obs))) / (len(null) + 1))

    beta = float(full.params[2])
    se = float(full.bse[2])
    result = {
        "fixed_score_logistic_regression": {
            "cnn_standardized_logit_beta": beta,
            "cnn_standardized_logit_se": se,
            "cnn_standardized_logit_odds_ratio": _json_float(np.exp(beta)),
            "cnn_beta_wald_ci": [
                _json_float(beta - 1.959963984540054 * se),
                _json_float(beta + 1.959963984540054 * se),
            ],
            "wald_p_two_sided": _json_float(full.pvalues[2]),
            "likelihood_ratio_chi2_df1": _json_float(lr),
            "likelihood_ratio_p": _json_float(stats.chi2.sf(lr, 1)),
            "coded_cnn_logit_pearson_correlation": _json_float(np.corrcoef(lc, ln)[0, 1]),
        },
        "efficient_score": {
            "z": _json_float(observed_efficient_z),
            "asymptotic_p_two_sided": _json_float(
                2.0 * stats.norm.sf(abs(observed_efficient_z))
            ),
            "coded_risk_stratified_permutation_p_two_sided": empirical(
                efficient_null, observed_efficient_z
            ),
        },
        "partial_spearman": {
            "rho": _json_float(observed_partial),
            "coded_risk_stratified_permutation_p_two_sided": empirical(
                partial_null, observed_partial
            ),
        },
        "conditional_permutation": {
            "n_permutations": B_CONDITIONAL,
            "n_coded_risk_strata": int(len(groups)),
            "stratum_sizes": [int(len(g)) for g in groups],
            "null": (
                "no within-coded-risk-stratum association between the fixed CNN score and "
                "outcome; CNN logits permuted within coded-score deciles"
            ),
            "qualification": (
                "approximate conditional randomization because conditioning uses deciles, "
                "not exact identical coded risks"
            ),
        },
    }
    return result, {"efficient_z": efficient_null, "partial_spearman": partial_null}


def _paired_analysis(
    y: np.ndarray,
    first_name: str,
    second_name: str,
    scores: dict[str, np.ndarray],
    auc_boot: dict[str, np.ndarray],
    logscore_boot: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[dict, dict[str, np.ndarray]]:
    first, second = scores[first_name], scores[second_name]
    obs_auc = _auc_rank(y, first) - _auc_rank(y, second)
    obs_logscore = _log_score_each(y, first).mean() - _log_score_each(y, second).mean()
    swap, nulls = _model_swap_test(y, first, second, rng)
    return (
        {
            "first": first_name,
            "second": second_name,
            "paired_bootstrap": _bootstrap_pair(
                auc_boot[first_name],
                auc_boot[second_name],
                logscore_boot[first_name],
                logscore_boot[second_name],
                obs_auc,
                obs_logscore,
            ),
            "delong_auroc": _delong_pair(y, first, second),
            "paired_model_swap_randomization": swap,
        },
        nulls,
    )


def main() -> None:
    config.ensure_dirs()
    aligned, provenance = _load_aligned()
    y = aligned["y_SLC26A4"].to_numpy(dtype=np.int8)
    folds = aligned["fold"].to_numpy(dtype=int)
    p_coded = aligned["p_coded_primary"].to_numpy(dtype=float)
    p_single = aligned["p_cnn_single"].to_numpy(dtype=float)
    p_multitask = aligned["p_cnn_multitask"].to_numpy(dtype=float)

    stack_single, reduced_single, coef_single = _conditional_meta_map(
        y, folds, p_coded, p_single
    )
    stack_multitask, reduced_multitask, coef_multitask = _conditional_meta_map(
        y, folds, p_coded, p_multitask
    )
    if not np.allclose(reduced_single, reduced_multitask, atol=1e-12, rtol=0):
        raise AssertionError("coded-only conditional reference unexpectedly depends on CNN")
    reduced = reduced_single

    aligned["p_conditional_stack_single"] = stack_single
    aligned["p_conditional_stack_multitask"] = stack_multitask
    aligned["p_conditional_coded_only_metamap"] = reduced
    pred_path = config.CNN_DIR / "added_value_predictions.csv"
    aligned.to_csv(pred_path, index=False)

    scores = {
        "coded_primary": p_coded,
        "cnn_single": p_single,
        "cnn_multitask": p_multitask,
        "conditional_stack_single": stack_single,
        "conditional_stack_multitask": stack_multitask,
        "conditional_coded_only_metamap": reduced,
    }
    seed_sequence = np.random.SeedSequence(SEED)
    child = seed_sequence.spawn(8)
    rng_boot = np.random.default_rng(child[0])
    rng_label = np.random.default_rng(child[1])
    rng_cond_single = np.random.default_rng(child[2])
    rng_cond_multitask = np.random.default_rng(child[3])
    rng_swap_single_raw = np.random.default_rng(child[4])
    rng_swap_multitask_raw = np.random.default_rng(child[5])
    rng_swap_single_matched = np.random.default_rng(child[6])
    rng_swap_multitask_matched = np.random.default_rng(child[7])

    auc_boot, logscore_boot = _bootstrap_distributions(y, scores, rng_boot)
    metric_summaries = {
        name: _metric_summary(y, p, auc_boot[name], logscore_boot[name])
        for name, p in scores.items()
    }

    label_permutation, label_nulls = _global_label_permutation(
        y,
        {
            "coded_primary": p_coded,
            "cnn_single": p_single,
            "cnn_multitask": p_multitask,
        },
        rng_label,
    )
    conditional_single, cond_null_single = _conditional_association(
        y, p_coded, p_single, rng_cond_single
    )
    conditional_multitask, cond_null_multitask = _conditional_association(
        y, p_coded, p_multitask, rng_cond_multitask
    )

    comparisons: dict[str, dict] = {}
    randomization_nulls: dict[str, np.ndarray] = {}
    comparison_specs: list[tuple[str, str, str, np.random.Generator]] = [
        (
            "single_stack_vs_exact_primary_coded",
            "conditional_stack_single",
            "coded_primary",
            rng_swap_single_raw,
        ),
        (
            "multitask_stack_vs_exact_primary_coded",
            "conditional_stack_multitask",
            "coded_primary",
            rng_swap_multitask_raw,
        ),
        (
            "single_stack_vs_matched_coded_metamap",
            "conditional_stack_single",
            "conditional_coded_only_metamap",
            rng_swap_single_matched,
        ),
        (
            "multitask_stack_vs_matched_coded_metamap",
            "conditional_stack_multitask",
            "conditional_coded_only_metamap",
            rng_swap_multitask_matched,
        ),
    ]
    for key, first, second, rng in comparison_specs:
        comparisons[key], nulls = _paired_analysis(
            y, first, second, scores, auc_boot, logscore_boot, rng
        )
        randomization_nulls[f"{key}_auc"] = nulls["auc"]
        randomization_nulls[f"{key}_mean_log_score"] = nulls["mean_log_score"]

    full = {
        "meta": {
            "analysis": "cnn_added_value",
            "design": "fixed out-of-fold scores; the CNNs and the grade model are not refitted",
            "n": int(len(y)),
            "n_SLC26A4": int(y.sum()),
            "n_comparator": int(len(y) - y.sum()),
            "seed": SEED,
            "bootstrap_replicates": B_BOOT,
            "global_label_permutations": B_LABEL,
            "paired_model_swap_randomizations": B_SWAP,
            "conditional_permutations": B_CONDITIONAL,
            "input_provenance": provenance,
            "input_generation_note": (
                "The structured score is the nested-CV grade-model output, and the single-task "
                "and multitask CNN scores are the outputs of the two CNN pipelines "
                "(identical pipelines except the six auxiliary anatomy heads and their losses); "
                "all three are out-of-fold scores on the same patients."
            ),
            "positive_class": "recorded SLC26A4 status",
        },
        "alignment": {
            "patient_ids_unique": bool(aligned["patient_id"].is_unique),
            "prediction_file": str(pred_path),
            "prediction_sha256": _sha256(pred_path),
        },
        "fixed_score_metrics": metric_summaries,
        "global_fixed_score_label_permutation": label_permutation,
        "conditional_meta_maps": {
            "description": (
                "logistic maps over the saved OOF logits, cross-fitted with the same folds; "
                "descriptive"
            ),
            "single_task_per_fold_coefficients": coef_single,
            "multitask_per_fold_coefficients": coef_multitask,
        },
        "paired_comparisons": comparisons,
        "conditional_association_after_coded_score": {
            "single_task": conditional_single,
            "multitask": conditional_multitask,
        },
    }
    metric_path = config.CNN_DIR / "added_value.json"
    with metric_path.open("w") as fh:
        json.dump(full, fh, indent=2, allow_nan=False)

    # Save the bootstrap and null distributions so the empirical results can be checked
    # without rerunning the randomization loops.
    distribution_path = config.CNN_DIR / "added_value_distributions.npz"
    distribution_payload: dict[str, np.ndarray] = {}
    for name, values in auc_boot.items():
        distribution_payload[f"bootstrap_auc__{name}"] = values
    for name, values in logscore_boot.items():
        distribution_payload[f"bootstrap_mean_log_score__{name}"] = values
    for name, values in label_nulls.items():
        distribution_payload[f"global_label_null_auc__{name}"] = values
    distribution_payload.update(randomization_nulls)
    distribution_payload["conditional_single_efficient_z"] = cond_null_single["efficient_z"]
    distribution_payload["conditional_single_partial_spearman"] = cond_null_single[
        "partial_spearman"
    ]
    distribution_payload["conditional_multitask_efficient_z"] = cond_null_multitask[
        "efficient_z"
    ]
    distribution_payload["conditional_multitask_partial_spearman"] = cond_null_multitask[
        "partial_spearman"
    ]
    np.savez_compressed(distribution_path, **distribution_payload)

    # Short summary of the main results.
    headlines = {
        "n": int(len(y)),
        "auc": {
            name: metric_summaries[name]["auroc"]
            for name in [
                "coded_primary",
                "cnn_single",
                "cnn_multitask",
                "conditional_stack_single",
                "conditional_stack_multitask",
            ]
        },
        "single_task_increment_after_coded": {
            "stack_minus_exact_coded_auc": comparisons[
                "single_stack_vs_exact_primary_coded"
            ]["delong_auroc"],
            "conditional_logistic_lrt_p": conditional_single[
                "fixed_score_logistic_regression"
            ]["likelihood_ratio_p"],
            "conditional_permutation_p": conditional_single["efficient_score"][
                "coded_risk_stratified_permutation_p_two_sided"
            ],
            "partial_spearman": conditional_single["partial_spearman"],
        },
        "multitask_increment_after_coded": {
            "stack_minus_exact_coded_auc": comparisons[
                "multitask_stack_vs_exact_primary_coded"
            ]["delong_auroc"],
            "conditional_logistic_lrt_p": conditional_multitask[
                "fixed_score_logistic_regression"
            ]["likelihood_ratio_p"],
            "conditional_permutation_p": conditional_multitask["efficient_score"][
                "coded_risk_stratified_permutation_p_two_sided"
            ],
            "partial_spearman": conditional_multitask["partial_spearman"],
        },
        "global_label_permutation_p_two_sided": {
            name: label_permutation["scores"][name]["p_two_sided_any_association"]
            for name in ["coded_primary", "cnn_single", "cnn_multitask"]
        },
        "full_metrics": str(metric_path),
        "predictions": str(pred_path),
        "inference_distributions": str(distribution_path),
    }
    headline_path = config.CNN_DIR / "added_value_summary.json"
    with headline_path.open("w") as fh:
        json.dump(headlines, fh, indent=2, allow_nan=False)

    print("=" * 79)
    print("CNN ADDED VALUE AFTER THE GRADE-MODEL SCORE")
    print("=" * 79)
    print(
        f"N={len(y)}; coded AUC={_auc_rank(y, p_coded):.4f}; "
        f"single CNN={_auc_rank(y, p_single):.4f}; multitask CNN={_auc_rank(y, p_multitask):.4f}"
    )
    for architecture in ["single", "multitask"]:
        stack_name = f"conditional_stack_{architecture}"
        comparison = comparisons[f"{architecture}_stack_vs_exact_primary_coded"]
        conditional = (
            conditional_single if architecture == "single" else conditional_multitask
        )
        print(
            f"{architecture:9s}: conditional stack AUC={_auc_rank(y, scores[stack_name]):.4f}; "
            f"delta vs exact coded={comparison['delong_auroc']['delta']:+.4f} "
            f"(DeLong p={comparison['delong_auroc']['p_two_sided']:.3f}); "
            f"conditional LRT p={conditional['fixed_score_logistic_regression']['likelihood_ratio_p']:.3f}; "
            f"stratified-permutation p={conditional['efficient_score']['coded_risk_stratified_permutation_p_two_sided']:.3f}"
        )
    print(f"saved: {pred_path}")
    print(f"saved: {metric_path}")
    print(f"saved: {headline_path}")
    print(f"saved: {distribution_path}")


if __name__ == "__main__":
    main()
