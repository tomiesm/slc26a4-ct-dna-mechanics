"""Compare the CNN scores between genotype groups and by vestibular enlargement in the variant subset.

No model is fitted. The saved patient-level out-of-fold SLC26A4 scores of both CNNs are joined
to the variant table and compared descriptively between c.919-2A>G homozygotes and compound
heterozygotes and between normal and enlarged vestibules (Mann-Whitney U test, AUROC with a
bootstrap interval that resamples within groups, Holm adjustment over the four tests).

Usage: python -m cnn.by_genotype_group
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

import config

GENOMIC = config.VARIANT_TABLE
SINGLE_TASK = config.cnn_predictions("single")
MULTITASK = config.cnn_predictions("multitask")
STRUCTURED = config.GRADE_PREDICTIONS

JOIN_OUT = config.CNN_DIR / "scores_by_genotype_group.csv"
TEST_OUT = config.CNN_DIR / "scores_by_genotype_group_tests.csv"
METRIC_OUT = config.CNN_DIR / "scores_by_genotype_group.json"

SEED = 20260713
BOOTSTRAPS = 20_000


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def stratified_auc_ci(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    negative = score[y == 0]
    positive = score[y == 1]
    draws = np.empty(BOOTSTRAPS, dtype=float)
    # AUC is the probability that a positive score exceeds a negative score,
    # with half credit for ties.  Vectorizing that identity avoids tens of
    # thousands of expensive sklearn calls.
    batch_size = 2_000
    generated = 0
    while generated < BOOTSTRAPS:
        batch = min(batch_size, BOOTSTRAPS - generated)
        sampled_negative = rng.choice(negative, size=(batch, len(negative)), replace=True)
        sampled_positive = rng.choice(positive, size=(batch, len(positive)), replace=True)
        pairwise = sampled_positive[:, :, None] - sampled_negative[:, None, :]
        draws[generated : generated + batch] = (
            (pairwise > 0).mean(axis=(1, 2)) + 0.5 * (pairwise == 0).mean(axis=(1, 2))
        )
        generated += batch
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def score_test(
    data: pd.DataFrame,
    score: str,
    endpoint: str,
    label: str,
) -> dict[str, Any]:
    y = data[endpoint].to_numpy(dtype=int)
    values = data[score].to_numpy(dtype=float)
    assert set(np.unique(y)) == {0, 1}
    group0, group1 = values[y == 0], values[y == 1]
    test = stats.mannwhitneyu(group0, group1, alternative="two-sided", method="auto")
    auc = roc_auc_score(y, values)
    low, high = stratified_auc_ci(y, values)
    return {
        "comparison": label,
        "score": score,
        "endpoint": endpoint,
        "group0_n": int(len(group0)),
        "group0_median": float(np.median(group0)),
        "group0_q1": float(np.quantile(group0, 0.25)),
        "group0_q3": float(np.quantile(group0, 0.75)),
        "group1_n": int(len(group1)),
        "group1_median": float(np.median(group1)),
        "group1_q1": float(np.quantile(group1, 0.25)),
        "group1_q3": float(np.quantile(group1, 0.75)),
        "auc_probability_group1_higher": float(auc),
        "auc_bootstrap_low": low,
        "auc_bootstrap_high": high,
        "mann_whitney_u": float(test.statistic),
        "mann_whitney_two_sided_p": float(test.pvalue),
    }


def main() -> None:
    config.ensure_dirs()
    genomic = pd.read_csv(GENOMIC)
    single_task = pd.read_csv(SINGLE_TASK)
    multitask = pd.read_csv(MULTITASK)
    structured = pd.read_csv(STRUCTURED)
    for name, frame in {
        "genomic": genomic,
        "single_task": single_task,
        "multitask": multitask,
        "structured": structured,
    }.items():
        assert frame.patient_id.is_unique, f"Duplicate ID in {name}"

    # The grade-model, single-task and multitask out-of-fold scores share one orientation:
    # y_SLC26A4 == 1 for SLC26A4 and the probability is P(SLC26A4). Check that the labels agree
    # across the three sources and compute the global single-task CNN AUC.
    label_check = structured[["patient_id", "y_SLC26A4"]].merge(
        single_task[["patient_id", "y_SLC26A4"]], on="patient_id", suffixes=("_canonical", "_single")
    ).merge(
        multitask[["patient_id", "y_SLC26A4"]].rename(columns={"y_SLC26A4": "y_SLC26A4_multitask"}),
        on="patient_id",
    )
    assert (label_check.y_SLC26A4_single == label_check.y_SLC26A4_canonical).all()
    assert (label_check.y_SLC26A4_multitask == label_check.y_SLC26A4_canonical).all()
    single_task_global_auc = float(roc_auc_score(
        label_check.y_SLC26A4_canonical,
        single_task.set_index("patient_id").loc[label_check.patient_id, "p_ent_SLC26A4"],
    ))

    joined = (
        genomic.merge(
            single_task[["patient_id", "p_ent_SLC26A4"]].rename(
                columns={"p_ent_SLC26A4": "single_task_cnn_oof_score"}
            ),
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            multitask[["patient_id", "p_ent_SLC26A4"]].rename(
                columns={"p_ent_SLC26A4": "multitask_cnn_oof_score"}
            ),
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            structured[["patient_id", "y_SLC26A4", "p_4_anatomy_PRIMARY"]].rename(
                columns={"p_4_anatomy_PRIMARY": "structured_anatomy_oof_score"}
            ),
            on="patient_id",
            how="inner",
            validate="one_to_one",
        )
    )
    missing = sorted(set(genomic.patient_id) - set(joined.patient_id))
    assert len(joined) == 34 and missing == ["<PATIENT_ID_29>"]
    assert (joined.y_SLC26A4 == 1).all(), "Every joined genomic patient must be a canonical SLC26A4 case"
    joined["compound_vs_c919_homozygous"] = (
        joined.zygosity == "compound_heterozygous"
    ).astype(int)

    # Only the two CNN scores are tested.  The structured score contains the
    # vestibular grade itself, so testing it against vestibular enlargement
    # would be circular.
    score_columns = ["single_task_cnn_oof_score", "multitask_cnn_oof_score"]
    tests: list[dict[str, Any]] = []
    for score in score_columns:
        tests.append(
            score_test(
                joined,
                score,
                "compound_vs_c919_homozygous",
                "c.919 homozygous (0) versus all compound (1)",
            )
        )
        tests.append(
            score_test(
                joined,
                score,
                "vestibule_enlarged",
                "normal vestibule (0) versus enlarged vestibule (1)",
            )
        )
    test_frame = pd.DataFrame(tests)
    test_frame["holm_adjusted_p_four_posthoc_tests"] = multipletests(
        test_frame.mann_whitney_two_sided_p, method="holm"
    )[1]

    metrics = {
        "analysis": "cnn_by_genotype_group",
        "population": {
            "variant_table_patients": int(len(genomic)),
            "joined_to_all_three_oof_scores": int(len(joined)),
            "not_in_analysis_cohort": missing,
        },
        "score_provenance": {
            "single_task_cnn": str(SINGLE_TASK),
            "multitask_cnn": str(MULTITASK),
            "structured_anatomy": str(STRUCTURED),
            "no_refitting": True,
            "orientation_note": (
                "grade-model, single-task and multitask out-of-fold scores share one orientation: "
                "y_SLC26A4==1 for SLC26A4, probability = P(SLC26A4)"
            ),
            "single_task_cnn_global_auroc": round(single_task_global_auc, 4),
        },
        "tests": test_frame.to_dict(orient="records"),
    }

    joined.to_csv(JOIN_OUT, index=False)
    test_frame.to_csv(TEST_OUT, index=False)
    with METRIC_OUT.open("w") as handle:
        json.dump(jsonable(metrics), handle, indent=2, allow_nan=False)

    print("CNN scores by genotype group")
    print(f"  joined {len(joined)}/{len(genomic)}; missing={missing}")
    for row in test_frame.itertuples(index=False):
        print(
            f"  {row.score} | {row.endpoint}: AUC={row.auc_probability_group1_higher:.3f} "
            f"[{row.auc_bootstrap_low:.3f}, {row.auc_bootstrap_high:.3f}], "
            f"MWU p={row.mann_whitney_two_sided_p:.4g}, Holm p={row.holm_adjusted_p_four_posthoc_tests:.4g}"
        )
    print(f"  saved {METRIC_OUT}")


if __name__ == "__main__":
    main()
