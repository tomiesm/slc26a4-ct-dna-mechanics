"""Compare inner-ear anatomy between genotype groups and test the DNA descriptors in the variant subset.

Reads the patient and allele tables of variants/variant_table.py. The primary endpoint is vestibular
enlargement (worst-ear grade 3 versus normal). The script reports exact 2x2 tests, a sensitivity analysis
that removes each recurrent compound genotype pair, permutation tests of the allele-order-invariant cgNA+
descriptors with max-|T| family-wise control, and descriptive cross-validated prediction baselines.

Usage: python -m variants.analysis
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import odds_ratio
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

import config

METRIC_OUT = config.VARIANT_DIR / "analysis.json"
EFFECT_OUT = config.VARIANT_DIR / "vestibular_effects.csv"
GROUP_OUT = config.VARIANT_DIR / "genotype_group_counts.csv"
LEAVE_GROUP_OUT = config.VARIANT_DIR / "leave_recurrent_pair_out.csv"

SEED = 20260713
N_PERMUTATIONS = 100_000
# A permuted statistic within TIE_TOL of the observed one counts as reaching it, so exact ties count despite rounding.
TIE_TOL = 1e-10


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def wilson(events: int, total: int, alpha: float = 0.05) -> tuple[float, float]:
    low, high = proportion_confint(events, total, alpha=alpha, method="wilson")
    return float(low), float(high)


def exact_binary_comparison(
    frame: pd.DataFrame,
    group1: pd.Series,
    group2: pd.Series,
    endpoint: str,
    comparison: str,
    group1_label: str,
    group2_label: str,
) -> dict[str, Any]:
    """Two-sided Fisher test, conditional exact OR CI, and Newcombe RD CI."""
    first = frame.loc[group1, endpoint].astype(int)
    second = frame.loc[group2, endpoint].astype(int)
    assert len(first) and len(second) and not (group1 & group2).any()
    event1, n1 = int(first.sum()), int(len(first))
    event2, n2 = int(second.sum()), int(len(second))
    table = np.asarray([[event1, n1 - event1], [event2, n2 - event2]], dtype=int)
    fisher = stats.fisher_exact(table, alternative="two-sided")
    conditional = odds_ratio(table, kind="conditional")
    or_ci = conditional.confidence_interval(confidence_level=0.95)
    rd_low, rd_high = confint_proportions_2indep(
        event1,
        n1,
        event2,
        n2,
        compare="diff",
        method="newcomb",
        alpha=0.05,
    )
    ci1 = wilson(event1, n1)
    ci2 = wilson(event2, n2)
    return {
        "comparison": comparison,
        "endpoint": endpoint,
        "group1": group1_label,
        "group2": group2_label,
        "group1_events": event1,
        "group1_n": n1,
        "group1_risk": event1 / n1,
        "group1_wilson_low": ci1[0],
        "group1_wilson_high": ci1[1],
        "group2_events": event2,
        "group2_n": n2,
        "group2_risk": event2 / n2,
        "group2_wilson_low": ci2[0],
        "group2_wilson_high": ci2[1],
        "risk_difference_group1_minus_group2": event1 / n1 - event2 / n2,
        "risk_difference_newcombe_low": float(rd_low),
        "risk_difference_newcombe_high": float(rd_high),
        "conditional_exact_odds_ratio_group1_vs_group2": float(conditional.statistic),
        "conditional_exact_or_low": float(or_ci.low),
        "conditional_exact_or_high": float(or_ci.high),
        "fisher_two_sided_p": float(fisher.pvalue),
    }


def fisher_freeman_halton_3x2(event_nonevent_table: np.ndarray) -> dict[str, Any]:
    """Deterministic probability-ordered exact test for a 3x2 table.

    Conditioning on row totals and the total number of events leaves a
    multivariate hypergeometric distribution.  With three rows and N=35 the
    full support is tiny, so enumeration is preferable to a stochastic default.
    """
    table = np.asarray(event_nonevent_table, dtype=int)
    assert table.shape == (3, 2) and (table >= 0).all()
    row_totals = table.sum(axis=1).tolist()
    total_events = int(table[:, 0].sum())
    total = int(table.sum())

    def probability(events: list[int]) -> float:
        numerator = math.prod(math.comb(n, k) for n, k in zip(row_totals, events))
        return numerator / math.comb(total, total_events)

    observed_events = table[:, 0].tolist()
    observed_probability = probability(observed_events)
    two_sided = 0.0
    support_size = 0
    for event0 in range(row_totals[0] + 1):
        for event1 in range(row_totals[1] + 1):
            event2 = total_events - event0 - event1
            if not 0 <= event2 <= row_totals[2]:
                continue
            support_size += 1
            current = probability([event0, event1, event2])
            if current <= observed_probability + 1e-15:
                two_sided += current
    return {
        "pvalue": float(min(two_sided, 1.0)),
        "observed_table_probability": float(observed_probability),
        "support_size": support_size,
        "definition": "sum of conditional table probabilities no greater than the observed table probability",
    }


def permutation_feature_screen(
    frame: pd.DataFrame,
    outcome: np.ndarray,
    feature_columns: list[str],
    permutations: int,
    seed: int,
    strata: np.ndarray | None = None,
    tie_tol: float = 0.0,
) -> pd.DataFrame:
    """Spearman screen with common-label max-|T| permutation control.

    Missingness masks may differ.  Features sharing a mask are evaluated
    together.  For every Monte Carlo draw the same full-patient label
    permutation is used for every feature, then the maximum absolute statistic
    is retained.  This preserves feature dependence for a max-T family-wise
    adjustment under the joint exchangeability null.  With differing
    missingness masks, no stronger subset-pivotality claim is made.

    A permuted statistic counts as reaching the observed one when it is at
    least the observed value minus ``tie_tol``.  The observed statistic comes
    from scipy.stats.spearmanr and the permuted ones from products of centered
    ranks, so an exact tie can differ in the last floating-point digit; a small
    positive ``tie_tol`` (TIE_TOL in main) counts such ties.
    """
    tests: list[dict[str, Any]] = []
    for feature in feature_columns:
        values = frame[feature].to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 20 or np.unique(values[mask]).size <= 2:
            continue
        rho, asymptotic_p = stats.spearmanr(values[mask], outcome[mask])
        if not np.isfinite(rho):
            continue
        tests.append(
            {
                "feature": feature,
                "n": int(mask.sum()),
                "rho": float(rho),
                "asymptotic_p": float(asymptotic_p),
                "mask": mask,
                "rank": stats.rankdata(values[mask], method="average").astype(float),
            }
        )
    if not tests:
        raise AssertionError("No cgNA descriptors qualified for inference")

    # Group columns by identical finite-case mask.
    mask_groups: dict[bytes, list[int]] = defaultdict(list)
    for index, test in enumerate(tests):
        mask_groups[np.packbits(test["mask"]).tobytes()].append(index)
    prepared: list[tuple[np.ndarray, np.ndarray, list[int]]] = []
    for indices in mask_groups.values():
        mask = tests[indices[0]]["mask"]
        selected = np.flatnonzero(mask)
        matrix = np.column_stack([tests[index]["rank"] for index in indices])
        matrix -= matrix.mean(axis=0, keepdims=True)
        norms = np.sqrt(np.sum(matrix**2, axis=0, keepdims=True))
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        prepared.append((selected, matrix, indices))

    rng = np.random.default_rng(seed)
    observed = np.abs(np.asarray([test["rho"] for test in tests], dtype=float))
    raw_exceed = np.zeros(len(tests), dtype=np.int64)
    max_exceed = np.zeros(len(tests), dtype=np.int64)
    batch_size = 2_000

    generated = 0
    while generated < permutations:
        batch = min(batch_size, permutations - generated)
        if strata is None:
            permutation_index = np.vstack([rng.permutation(len(frame)) for _ in range(batch)])
        else:
            strata = np.asarray(strata)
            assert len(strata) == len(frame)
            permutation_index = np.tile(np.arange(len(frame)), (batch, 1))
            for level in pd.unique(strata):
                members = np.flatnonzero(strata == level)
                if len(members) > 1:
                    permutation_index[:, members] = np.vstack(
                        [rng.permutation(members) for _ in range(batch)]
                    )
        max_statistic = np.zeros(batch, dtype=float)
        for selected, ranked_features, indices in prepared:
            permuted_y = outcome[permutation_index[:, selected]].astype(float)
            permuted_y -= permuted_y.mean(axis=1, keepdims=True)
            y_norm = np.sqrt(np.sum(permuted_y**2, axis=1, keepdims=True))
            correlations = np.divide(
                permuted_y @ ranked_features,
                y_norm,
                out=np.zeros((batch, len(indices)), dtype=float),
                where=y_norm > 0,
            )
            absolute = np.abs(correlations)
            max_statistic = np.maximum(max_statistic, absolute.max(axis=1))
            raw_exceed[indices] += np.sum(
                absolute >= observed[np.asarray(indices)][None, :] - tie_tol, axis=0
            )
        max_exceed += np.sum(max_statistic[:, None] >= observed[None, :] - tie_tol, axis=0)
        generated += batch

    raw_p = (raw_exceed + 1) / (permutations + 1)
    fwer_p = (max_exceed + 1) / (permutations + 1)
    _, bh_q, _, _ = multipletests(raw_p, alpha=0.05, method="fdr_bh")

    result = pd.DataFrame(
        {
            "feature": [test["feature"] for test in tests],
            "n": [test["n"] for test in tests],
            "spearman_rho": [test["rho"] for test in tests],
            "asymptotic_p_descriptive": [test["asymptotic_p"] for test in tests],
            "permutation_p": raw_p,
            "bh_fdr_q": bh_q,
            "maxT_fwer_p": fwer_p,
        }
    )
    result["permutations"] = permutations
    result["seed"] = seed
    result["permutation_scheme"] = (
        "unrestricted_patient_labels" if strata is None else "within_allelic_configuration"
    )
    return result.sort_values(
        ["maxT_fwer_p", "permutation_p", "feature"], ascending=[True, True, True]
    ).reset_index(drop=True)


def slot_order_diagnostic(patients: pd.DataFrame, variants: pd.DataFrame) -> dict[str, Any]:
    """Spearman correlations of cgna_var_du_point_mid, taken from slot 1 or slot 2 of the DNA mechanics file or
    the mean of the two alleles, with the composite sum and vestibular enlargement."""
    rows: list[dict[str, Any]] = []
    for patient in patients.itertuples(index=False):
        sub = variants[variants.patient_id == patient.patient_id].copy()
        # Homozygotes have the same slot of the DNA mechanics file twice; collapse it here.
        sub = sub.drop_duplicates(["mat_slot"])
        slot1 = sub.loc[sub.mat_slot == 1, "cgna_var_du_point_mid"]
        slot2 = sub.loc[sub.mat_slot == 2, "cgna_var_du_point_mid"]
        value1 = float(slot1.iloc[0]) if len(slot1) else math.nan
        value2 = float(slot2.iloc[0]) if len(slot2) else value1
        rows.append(
            {
                "patient_id": patient.patient_id,
                "mat_slot1": value1,
                "mat_slot2": value2,
                "symmetric_mean": (
                    float(np.mean([value1, value2]))
                    if np.isfinite(value1) and np.isfinite(value2)
                    else math.nan
                ),
                "composite": patient.composite_sum_sensitivity,
                "vestibule": patient.vestibule_enlarged,
            }
        )
    diagnostic = pd.DataFrame(rows)
    output: dict[str, Any] = {}
    for representation in ("mat_slot1", "mat_slot2", "symmetric_mean"):
        mask = np.isfinite(diagnostic[representation])
        rho_composite, p_composite = stats.spearmanr(
            diagnostic.loc[mask, representation], diagnostic.loc[mask, "composite"]
        )
        rho_vestibule, p_vestibule = stats.spearmanr(
            diagnostic.loc[mask, representation], diagnostic.loc[mask, "vestibule"]
        )
        output[representation] = {
            "n": int(mask.sum()),
            "rho_composite": float(rho_composite),
            "p_composite_unadjusted": float(p_composite),
            "rho_vestibule": float(rho_vestibule),
            "p_vestibule_unadjusted": float(p_vestibule),
        }
    return output


def make_preprocessor(
    model_name: str,
    baseline_numeric: list[str],
    baseline_categorical: list[str],
    identity_columns: list[str],
    cgna_columns: list[str],
) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []
    if "simple" in model_name or "identity" in model_name:
        transformers.append(
            (
                "baseline_numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                baseline_numeric,
            )
        )
        transformers.append(
            (
                "baseline_categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                baseline_categorical,
            )
        )
    if "identity" in model_name:
        transformers.append(("variant_identity", "passthrough", identity_columns))
    if "cgna" in model_name:
        transformers.append(
            (
                "cgna",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        ("pca", PCA(n_components=3, svd_solver="full")),
                    ]
                ),
                cgna_columns,
            )
        )
    if not transformers:
        raise KeyError(model_name)
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def baseline_cv(patients: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Descriptive leave-one-patient-out and leave-genotype-pair-out baseline comparisons."""
    data = patients.copy()
    all_variants = sorted(set(data.allele1_cdna) | set(data.allele2_cdna))
    identity_columns: list[str] = []
    for index, variant in enumerate(all_variants):
        column = f"variant_identity_{index:02d}"
        data[column] = (data.allele1_cdna == variant).astype(int) + (
            data.allele2_cdna == variant
        ).astype(int)
        identity_columns.append(column)

    baseline_numeric = [
        "c919_dosage",
        "predicted_lof_allele_count",
        "position_mean",
        "position_range",
        "reference_gc_mean",
        "reference_gc_range",
        "alternative_gc_mean",
        "alternative_gc_range",
    ]
    baseline_categorical = ["allelic_configuration", "consequence_pair"]
    cgna_columns = sorted(
        column
        for column in data
        if column.startswith("cgna_")
        and column.endswith("_allele_mean")
        and data[column].notna().sum() >= 20
        and data[column].nunique(dropna=True) > 2
    )
    assert cgna_columns

    model_names = [
        "simple",
        "identity",
        "cgna",
        "simple_cgna",
        "identity_cgna",
    ]
    schemes = {
        "leave_one_patient_out": np.arange(len(data)),
        "leave_genotype_pair_out": data.genotype_pair.to_numpy(),
    }
    y = data.vestibule_enlarged.to_numpy(dtype=int)
    prediction_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for scheme, groups in schemes.items():
        splitter = LeaveOneGroupOut()
        predictions = {name: np.full(len(data), np.nan) for name in model_names}
        fold_count = 0
        for train, test in splitter.split(data, y, groups):
            assert np.unique(y[train]).size == 2
            fold_count += 1
            for model_name in model_names:
                preprocessor = make_preprocessor(
                    model_name,
                    baseline_numeric,
                    baseline_categorical,
                    identity_columns,
                    cgna_columns,
                )
                model = Pipeline(
                    [
                        ("features", preprocessor),
                        (
                            "logistic",
                            LogisticRegression(
                                penalty="l2",
                                C=0.1,
                                solver="liblinear",
                                max_iter=5_000,
                                random_state=SEED,
                            ),
                        ),
                    ]
                )
                model.fit(data.iloc[train], y[train])
                predictions[model_name][test] = model.predict_proba(data.iloc[test])[:, 1]

        scheme_summary: dict[str, Any] = {"folds": fold_count, "models": {}}
        for model_name, probability in predictions.items():
            assert np.isfinite(probability).all()
            probability = np.clip(probability, 1e-6, 1 - 1e-6)
            scheme_summary["models"][model_name] = {
                "auc": float(roc_auc_score(y, probability)),
                "log_loss": float(log_loss(y, probability, labels=[0, 1])),
                "brier": float(brier_score_loss(y, probability)),
            }
        scheme_summary["incremental_cgna"] = {
            "over_simple": {
                "auc_delta": scheme_summary["models"]["simple_cgna"]["auc"]
                - scheme_summary["models"]["simple"]["auc"],
                "log_loss_delta": scheme_summary["models"]["simple_cgna"]["log_loss"]
                - scheme_summary["models"]["simple"]["log_loss"],
                "brier_delta": scheme_summary["models"]["simple_cgna"]["brier"]
                - scheme_summary["models"]["simple"]["brier"],
            },
            "over_variant_identity": {
                "auc_delta": scheme_summary["models"]["identity_cgna"]["auc"]
                - scheme_summary["models"]["identity"]["auc"],
                "log_loss_delta": scheme_summary["models"]["identity_cgna"]["log_loss"]
                - scheme_summary["models"]["identity"]["log_loss"],
                "brier_delta": scheme_summary["models"]["identity_cgna"]["brier"]
                - scheme_summary["models"]["identity"]["brier"],
            },
        }
        summaries[scheme] = scheme_summary
        for index, patient in data.iterrows():
            record: dict[str, Any] = {
                "scheme": scheme,
                "patient_id": patient.patient_id,
                "genotype_pair": patient.genotype_pair,
                "vestibule_enlarged": int(y[index]),
            }
            for model_name, probability in predictions.items():
                record[f"p_{model_name}"] = float(probability[index])
            prediction_rows.append(record)

    metadata = {
        "baseline_numeric": baseline_numeric,
        "baseline_categorical": baseline_categorical,
        "variant_identity_vocabulary": all_variants,
        "n_variant_identity_columns": len(identity_columns),
        "cgna_representation": (
            "all eligible swap-invariant allele-mean descriptors, median imputation and 3-PC PCA fit within each training fold"
        ),
        "n_cgna_columns_entering_foldwise_pca": len(cgna_columns),
        "regularization": "fixed L2 logistic regression, C=0.1",
        "results": summaries,
    }
    return pd.DataFrame(prediction_rows), metadata


def main() -> None:
    config.ensure_dirs()
    patients = pd.read_csv(config.VARIANT_TABLE)
    variants = pd.read_csv(config.VARIANTS)
    assert len(patients) == 35 and patients.patient_id.is_unique
    assert len(variants) == 70 and (variants.groupby("patient_id").size() == 2).all()
    assert set(patients.vestibule_enlarged.unique()).issubset({0, 1})

    homo = patients.allelic_configuration == "c919_homozygous"
    compound = patients.zygosity == "compound_heterozygous"
    c919_compound = patients.allelic_configuration == "c919_compound"
    non_c919_compound = patients.allelic_configuration == "non_c919_compound"

    effect_rows = [
        exact_binary_comparison(
            patients,
            homo,
            compound,
            "vestibule_enlarged",
            "PRIMARY_c919_homozygous_vs_all_compound",
            "c.919-2A>G homozygous",
            "all compound heterozygous",
        ),
        exact_binary_comparison(
            patients,
            homo,
            c919_compound,
            "vestibule_enlarged",
            "SENSITIVITY_c919_homozygous_vs_c919_compound",
            "c.919-2A>G homozygous",
            "c.919-2A>G-containing compound",
        ),
        exact_binary_comparison(
            patients,
            c919_compound,
            non_c919_compound,
            "vestibule_enlarged",
            "SECONDARY_c919_compound_vs_non_c919_compound",
            "c.919-2A>G-containing compound",
            "non-c.919 compound",
        ),
        exact_binary_comparison(
            patients,
            homo,
            compound,
            "cochlea_abnormal",
            "SENSITIVITY_cochlea_c919_homozygous_vs_all_compound",
            "c.919-2A>G homozygous",
            "all compound heterozygous",
        ),
    ]
    effects = pd.DataFrame(effect_rows)
    secondary_mask = ~effects.comparison.str.startswith("PRIMARY_")
    effects["holm_adjusted_p_secondary_family"] = np.nan
    effects.loc[secondary_mask, "holm_adjusted_p_secondary_family"] = multipletests(
        effects.loc[secondary_mask, "fisher_two_sided_p"], method="holm"
    )[1]

    primary = effects.iloc[0]
    assert int(primary.group1_n) == 9 and int(primary.group2_n) == 26

    # Counts per allelic configuration.
    group_counts = (
        patients.groupby("allelic_configuration", observed=True)
        .agg(
            n=("patient_id", "size"),
            vestibule_enlarged=("vestibule_enlarged", "sum"),
            cochlea_abnormal=("cochlea_abnormal", "sum"),
            median_vestibule=("vestibule_worst", "median"),
            median_cochlea=("cochlea_worst", "median"),
            median_composite_sensitivity=("composite_sum_sensitivity", "median"),
        )
        .reset_index()
    )
    group_counts["vestibule_risk"] = group_counts.vestibule_enlarged / group_counts.n

    # Exact 3x2 global test across the three observed allelic configurations.
    config_order = ["c919_homozygous", "c919_compound", "non_c919_compound"]
    global_table = np.asarray(
        [
            [
                int(patients.loc[patients.allelic_configuration == group, "vestibule_enlarged"].sum()),
                int(
                    (patients.allelic_configuration == group).sum()
                    - patients.loc[
                        patients.allelic_configuration == group, "vestibule_enlarged"
                    ].sum()
                ),
            ]
            for group in config_order
        ]
    )
    global_fisher = fisher_freeman_halton_3x2(global_table)

    # The composite sum is a sensitivity analysis.
    composite_homo = patients.loc[homo, "composite_sum_sensitivity"]
    composite_compound = patients.loc[compound, "composite_sum_sensitivity"]
    composite_mwu = stats.mannwhitneyu(
        composite_homo, composite_compound, alternative="two-sided", method="auto"
    )

    # Remove every recurrent compound genotype pair in turn.
    recurrent = (
        patients.loc[compound, "genotype_pair"].value_counts().loc[lambda x: x >= 2].index.tolist()
    )
    leave_rows: list[dict[str, Any]] = []
    for genotype_pair in recurrent:
        retained = patients.genotype_pair != genotype_pair
        result = exact_binary_comparison(
            patients,
            homo & retained,
            compound & retained,
            "vestibule_enlarged",
            f"remove_{genotype_pair}",
            "c.919-2A>G homozygous",
            "remaining compound heterozygous",
        )
        result["removed_genotype_pair"] = genotype_pair
        result["removed_n"] = int((patients.genotype_pair == genotype_pair).sum())
        leave_rows.append(result)
    leave_frame = pd.DataFrame(leave_rows)

    cgna_columns = sorted(column for column in patients if column.startswith("cgna_"))
    # Availability is variant/edit dependent.  Summarize each distinct mask so
    # complete-case feature screens cannot be mistaken for a common N=35 panel.
    availability_patterns: list[dict[str, Any]] = []
    pattern_to_columns: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for column in cgna_columns:
        missing_ids = tuple(
            sorted(patients.loc[patients[column].isna(), "patient_id"].astype(str))
        )
        pattern_to_columns[missing_ids].append(column)
    for missing_ids, columns in sorted(
        pattern_to_columns.items(), key=lambda item: (len(item[0]), item[0])
    ):
        missing = patients.patient_id.astype(str).isin(missing_ids)
        availability_patterns.append(
            {
                "n_descriptors": len(columns),
                "available_n": int((~missing).sum()),
                "missing_n": int(missing.sum()),
                "missing_vestibule_enlarged_n": int(
                    patients.loc[missing, "vestibule_enlarged"].sum()
                ),
                "missing_patient_ids": list(missing_ids),
                "example_descriptor": columns[0],
            }
        )
    cgna_tests = permutation_feature_screen(
        patients,
        patients.vestibule_enlarged.to_numpy(dtype=int),
        cgna_columns,
        N_PERMUTATIONS,
        SEED,
        tie_tol=TIE_TOL,
    )
    assert (cgna_tests.maxT_fwer_p >= cgna_tests.permutation_p - 1e-12).all()
    cgna_conditional = permutation_feature_screen(
        patients,
        patients.vestibule_enlarged.to_numpy(dtype=int),
        cgna_columns,
        N_PERMUTATIONS,
        SEED + 1,
        strata=patients.allelic_configuration.to_numpy(),
        tie_tol=TIE_TOL,
    )
    assert (cgna_conditional.maxT_fwer_p >= cgna_conditional.permutation_p - 1e-12).all()

    order_diagnostic = slot_order_diagnostic(patients, variants)
    baseline_predictions, baseline_summary = baseline_cv(patients)

    consequence_summary = {
        "allele_counts": variants.derived_consequence.value_counts().to_dict(),
        "patient_consequence_pair_counts": patients.consequence_pair.value_counts().to_dict(),
        "c919_partner_counts": patients.loc[
            c919_compound, "c919_partner_consequence"
        ].value_counts().to_dict(),
        "rule": (
            "Anchored HGVS-derived categories; pairs are sorted and therefore allele-order invariant."
        ),
    }

    metrics = {
        "analysis": "variant_analysis",
        "population": {
            "unique_patients": int(len(patients)),
            "c919_homozygous": int(homo.sum()),
            "all_compound": int(compound.sum()),
            "c919_compound": int(c919_compound.sum()),
            "non_c919_compound": int(non_c919_compound.sum()),
        },
        "primary_endpoint": {
            "definition": "worst-ear vestibular grade 3 (enlarged) versus grade 0 (normal)",
            "comparison": effect_rows[0],
        },
        "allelic_configuration_global": {
            "groups": config_order,
            "event_nonevent_table": global_table.tolist(),
            "fisher_freeman_halton_p": float(global_fisher["pvalue"]),
            "exact_test_details": global_fisher,
        },
        "secondary_and_sensitivity_effects": effects.iloc[1:].to_dict(orient="records"),
        "composite_sum_sensitivity": {
            "construction": "sum of six worst-ear raw structure codes; not treated as a validated severity scale",
            "median_c919_homozygous": float(composite_homo.median()),
            "median_all_compound": float(composite_compound.median()),
            "mann_whitney_u": float(composite_mwu.statistic),
            "mann_whitney_two_sided_p": float(composite_mwu.pvalue),
        },
        "cochlea_description": {
            "abnormal_total": int(patients.cochlea_abnormal.sum()),
            "total": int(len(patients)),
            "normal_total": int((1 - patients.cochlea_abnormal).sum()),
            "median_c919_homozygous": float(patients.loc[homo, "cochlea_worst"].median()),
            "median_all_compound": float(patients.loc[compound, "cochlea_worst"].median()),
        },
        "consequence_summary": consequence_summary,
        "leave_recurrent_genotype_out": {
            "recurrent_compound_pairs_tested": recurrent,
            "n_sensitivities": int(len(leave_frame)),
            "fisher_p_range": [
                float(leave_frame.fisher_two_sided_p.min()),
                float(leave_frame.fisher_two_sided_p.max()),
            ],
            "risk_difference_range": [
                float(leave_frame.risk_difference_group1_minus_group2.min()),
                float(leave_frame.risk_difference_group1_minus_group2.max()),
            ],
        },
        "cgna_multiplicity": {
            "endpoint": "vestibule_enlarged",
            "tested_descriptors": int(len(cgna_tests)),
            "permutations": N_PERMUTATIONS,
            "seed": SEED,
            "tie_tolerance": TIE_TOL,
            "method": (
                "common patient-label permutations against fixed swap-invariant descriptors; max absolute Spearman statistic "
                "across every screened descriptor for max-T family-wise adjustment under the joint exchangeability null; "
                "BH q also reported"
            ),
            "nominal_permutation_p_lt_0_05": int((cgna_tests.permutation_p < 0.05).sum()),
            "bh_q_lt_0_05": int((cgna_tests.bh_fdr_q < 0.05).sum()),
            "maxT_fwer_p_lt_0_05": int((cgna_tests.maxT_fwer_p < 0.05).sum()),
            "top_six": cgna_tests.head(6).to_dict(orient="records"),
            "conditional_on_allelic_configuration": {
                "method": (
                    "patient labels permuted only within c.919 homozygous, c.919-containing compound, "
                    "and non-c.919 compound strata; max-|T| and BH repeated across the same descriptor family"
                ),
                "tested_descriptors": int(len(cgna_conditional)),
                "nominal_permutation_p_lt_0_05": int(
                    (cgna_conditional.permutation_p < 0.05).sum()
                ),
                "bh_q_lt_0_05": int((cgna_conditional.bh_fdr_q < 0.05).sum()),
                "maxT_fwer_p_lt_0_05": int(
                    (cgna_conditional.maxT_fwer_p < 0.05).sum()
                ),
                "top_six": cgna_conditional.head(6).to_dict(orient="records"),
            },
            "slot_order_diagnostic": order_diagnostic,
        },
        "cgna_availability": {
            "patterns": availability_patterns,
        },
        "cgna_baseline_prediction": baseline_summary,
    }

    effects.to_csv(EFFECT_OUT, index=False)
    group_counts.to_csv(GROUP_OUT, index=False)
    leave_frame.to_csv(LEAVE_GROUP_OUT, index=False)
    cgna_tests.to_csv(config.DESCRIPTOR_TESTS, index=False)
    cgna_conditional.to_csv(config.DESCRIPTOR_TESTS_WITHIN_GROUPS, index=False)
    baseline_predictions.to_csv(config.DESCRIPTOR_PREDICTIONS, index=False)
    with METRIC_OUT.open("w") as handle:
        json.dump(jsonable(metrics), handle, indent=2, allow_nan=False)

    print(
        "  primary vestibular endpoint: "
        f"{int(primary.group1_events)}/{int(primary.group1_n)} vs "
        f"{int(primary.group2_events)}/{int(primary.group2_n)}; "
        f"conditional OR={primary.conditional_exact_odds_ratio_group1_vs_group2:.4g} "
        f"[{primary.conditional_exact_or_low:.4g}, {primary.conditional_exact_or_high:.4g}], "
        f"Fisher p={primary.fisher_two_sided_p:.6g}"
    )
    print(
        f"  cgNA descriptors tested={len(cgna_tests)}; nominal={int((cgna_tests.permutation_p < .05).sum())}; "
        f"BH-significant={int((cgna_tests.bh_fdr_q < .05).sum())}; "
        f"maxT-significant={int((cgna_tests.maxT_fwer_p < .05).sum())}"
    )
    print(
        "  conditional on allelic configuration: "
        f"nominal={int((cgna_conditional.permutation_p < .05).sum())}; "
        f"BH-significant={int((cgna_conditional.bh_fdr_q < .05).sum())}; "
        f"maxT-significant={int((cgna_conditional.maxT_fwer_p < .05).sum())}"
    )
    top = cgna_tests.iloc[0]
    print(
        f"  best adjusted cgNA: {top.feature}, rho={top.spearman_rho:+.3f}, "
        f"raw permutation p={top.permutation_p:.4g}, maxT p={top.maxT_fwer_p:.4g}"
    )
    for scheme, result in baseline_summary["results"].items():
        delta = result["incremental_cgna"]["over_variant_identity"]
        print(
            f"  {scheme}: cgNA over identity delta AUC={delta['auc_delta']:+.3f}, "
            f"delta log-loss={delta['log_loss_delta']:+.3f}"
        )
    print(f"  saved {METRIC_OUT}")
    print(f"  saved {config.DESCRIPTOR_TESTS}")


if __name__ == "__main__":
    main()
