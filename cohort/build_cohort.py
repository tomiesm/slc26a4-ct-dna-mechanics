"""Merge the records of the same person and assign the analysis cohort to five cross-validation folds.

The clinical dataset holds some people under two identifiers. Each such pair is merged into one
row per person before the cohort is built and the folds are assigned, so that one person never
appears in two folds; every merge decision and every field that differs between the two records
is written out. Two distinct people who may be related are kept in one cross-validation group.
The analysis cohort is the patients of config.COHORT_LIST; the folds are stratified by SLC26A4
status and sheet, and no group crosses folds.

Usage: python -m cohort.build_cohort
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config

SEED = 20260702

# The canonical record supplies one coherent bilateral anatomy annotation block.
# Genetics are reconciled separately: a documented test overrides ``not_tested``.
# Eight cross-sheet decisions have multiple concordant identifiers/acquisitions. The
# ninth, <PATIENT_ID_01>, lacks a second image copy and is merged as a probable match
# because it has the same sheet, identifier suffix, DOB, sex, two CT dates, anatomy, and genetics.
IDENTITY_DECISIONS: tuple[dict[str, str], ...] = (
    {
        "canonical_id": "<PATIENT_ID_02>",
        "alias_id": "<PATIENT_ID_03>",
        "confidence": "high",
        "evidence": "same DOB/sex; two overlapping CT dates and matching series inventories; SLC26A4 classification agrees",
    },
    {
        "canonical_id": "<PATIENT_ID_04>",
        "alias_id": "<PATIENT_ID_05>",
        "confidence": "confirmed_exact_image",
        "evidence": "same DOB/sex/date/genotype/anatomy; axial and coronal arrays are byte-identical",
    },
    {
        "canonical_id": "<PATIENT_ID_06>",
        "alias_id": "<PATIENT_ID_07>",
        "confidence": "high",
        "evidence": "same DOB/sex and two CT dates with matching series inventories; normal-sheet alias adds negative NGS",
    },
    {
        "canonical_id": "<PATIENT_ID_08>",
        "alias_id": "<PATIENT_ID_09>",
        "confidence": "high",
        "evidence": "same DOB/sex and two CT dates with matching series inventories",
    },
    {
        "canonical_id": "<PATIENT_ID_10>",
        "alias_id": "<PATIENT_ID_11>",
        "confidence": "high",
        "evidence": "same DOB/sex/date and matching series inventory; genetic results agree",
    },
    {
        "canonical_id": "<PATIENT_ID_12>",
        "alias_id": "<PATIENT_ID_13>",
        "confidence": "high",
        "evidence": "same DOB/sex/date and matching series inventory; one record carries the genetic result",
    },
    {
        "canonical_id": "<PATIENT_ID_14>",
        "alias_id": "<PATIENT_ID_15>",
        "confidence": "high",
        "evidence": "same DOB/sex and two CT dates; same source series inventory with alternate coronal selection",
    },
    {
        "canonical_id": "<PATIENT_ID_16>",
        "alias_id": "<PATIENT_ID_17>",
        "confidence": "high",
        "evidence": "same DOB/sex and overlapping CT dates/series; one record carries the genetic result",
    },
    {
        "canonical_id": "<PATIENT_ID_18>",
        "alias_id": "<PATIENT_ID_01>",
        "confidence": "probable",
        "evidence": "same normal sheet and identifier suffix; exact DOB/sex/two CT dates/anatomy/genetics; alias has no separate image files",
    },
)

# Two records that may belong to related people are kept as distinct people in one CV group.
POSSIBLE_FAMILY_GROUPS: tuple[tuple[str, ...], ...] = (
    ("<PATIENT_ID_19>", "<PATIENT_ID_20>"),
)

ANATOMY_FIELDS = tuple(
    f"{structure}_{side}"
    for structure in ("cochlea", "vestibule", "scc", "eva", "iac", "aperture", "cn")
    for side in ("R", "L")
)
GENETIC_FIELDS = (
    "gene_category",
    "gene_class_3",
    "gene_class_fine",
    "genetic_mutation_raw",
)
BOOLEAN_OR_FIELDS = (
    "gene_tested",
    "had_surgery",
    "has_npz",
    "has_any_abnormality",
    "charge_syndrome",
    "waardenburg",
    "digeorge",
    "goiter",
    "vacterl",
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def dump_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_json_value, allow_nan=False) + "\n")


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_sex(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().upper()
    return text


def same_value(first: Any, second: Any) -> bool:
    if pd.isna(first) and pd.isna(second):
        return True
    if isinstance(first, (pd.Timestamp, np.datetime64)) or isinstance(
        second, (pd.Timestamp, np.datetime64)
    ):
        return pd.Timestamp(first) == pd.Timestamp(second)
    if isinstance(first, (float, np.floating)) or isinstance(second, (float, np.floating)):
        try:
            return bool(np.isclose(float(first), float(second), equal_nan=True))
        except (TypeError, ValueError):
            pass
    return str(first) == str(second)


def source_ct_dates(master: pd.DataFrame, patient_id: str) -> list[str]:
    dates = pd.to_datetime(
        master.loc[master["patient_id"].eq(patient_id), "ct_date"], errors="coerce"
    ).dropna()
    return sorted({value.date().isoformat() for value in dates})


def build_resolved_patient_table(
    source: pd.DataFrame, master: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = source.copy()
    source["gender"] = source["gender"].map(normalize_sex)
    source_indexed = source.set_index("patient_id", drop=False)
    alias_to_canonical = {
        decision["alias_id"]: decision["canonical_id"] for decision in IDENTITY_DECISIONS
    }
    source["canonical_person_id"] = source["patient_id"].replace(alias_to_canonical)

    decision_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    resolved_rows: list[pd.Series] = []

    decision_by_canonical = {
        decision["canonical_id"]: decision for decision in IDENTITY_DECISIONS
    }
    for canonical_id, group in source.groupby("canonical_person_id", sort=True):
        group = group.copy()
        if canonical_id not in set(group["patient_id"]):
            raise AssertionError(f"Canonical record missing for {canonical_id}")
        canonical = group.loc[group["patient_id"].eq(canonical_id)].iloc[0].copy()

        birthdays = pd.to_datetime(group["birthday"], errors="coerce").dropna().dt.date.unique()
        sexes = sorted(set(group["gender"].dropna().astype(str)))
        if len(birthdays) > 1 or len(sexes) > 1:
            raise AssertionError(
                f"Demographic disagreement in identity group {canonical_id}: "
                f"birthdays={birthdays}, sexes={sexes}"
            )

        tested = group.loc[group["gene_tested"].astype(bool)].copy()
        if not tested.empty:
            tested_categories = sorted(set(tested["gene_category"].astype(str)))
            if len(tested_categories) > 1:
                raise AssertionError(
                    f"Tested genetic-class disagreement for {canonical_id}: {tested_categories}"
                )
            genetic_source = (
                tested.loc[tested["patient_id"].eq(canonical_id)].iloc[0]
                if tested["patient_id"].eq(canonical_id).any()
                else tested.iloc[0]
            )
            canonical["gene_tested"] = True
            for field in GENETIC_FIELDS:
                canonical[field] = genetic_source[field]

        for field in BOOLEAN_OR_FIELDS:
            if field in group.columns:
                canonical[field] = bool(group[field].fillna(False).astype(bool).any())
        for field in ("n_ct_scans", "n_npz_files"):
            if field in group.columns:
                canonical[field] = pd.to_numeric(group[field], errors="coerce").max()

        if "underlying_text" in group.columns:
            tagged = []
            for row in group.itertuples(index=False):
                value = getattr(row, "underlying_text")
                if pd.notna(value) and str(value).strip():
                    item = f"{row.patient_id}: {str(value).strip()}"
                    if item not in tagged:
                        tagged.append(item)
            canonical["underlying_text"] = " | ".join(tagged) if tagged else None

        canonical["patient_id"] = canonical_id
        canonical["canonical_person_id"] = canonical_id
        aliases = sorted(set(group["patient_id"]) - {canonical_id})
        canonical["identity_aliases"] = ";".join(aliases)
        canonical["identity_source_record_count"] = int(len(group))
        canonical["identity_resolution_status"] = (
            decision_by_canonical.get(canonical_id, {}).get("confidence", "singleton")
        )
        canonical["cv_group_id"] = canonical_id
        for family_index, members in enumerate(POSSIBLE_FAMILY_GROUPS, start=1):
            if canonical_id in members:
                canonical["cv_group_id"] = f"possible_family_{family_index:02d}"
        resolved_rows.append(canonical)

        if len(group) > 1:
            decision = decision_by_canonical[canonical_id]
            alias_id = decision["alias_id"]
            alias = source_indexed.loc[alias_id]
            for field in source.columns:
                if field in ("patient_id", "canonical_person_id"):
                    continue
                first, second = source_indexed.loc[canonical_id, field], alias[field]
                if not same_value(first, second):
                    conflict_rows.append(
                        {
                            "canonical_id": canonical_id,
                            "alias_id": alias_id,
                            "field": field,
                            "canonical_value": None if pd.isna(first) else str(first),
                            "alias_value": None if pd.isna(second) else str(second),
                            "resolution": (
                                "documented_test_overrides_no_test"
                                if field in ("gene_tested", *GENETIC_FIELDS) and not tested.empty
                                else "canonical_record_block_retained"
                            ),
                        }
                    )
            canonical_cache = config.SLICE_DIR / f"{canonical_id}.npy"
            alias_cache = config.SLICE_DIR / f"{alias_id}.npy"
            canonical_npz = config.CORONAL_CROPS_DIR / f"{canonical_id}.npz"
            alias_npz = config.CORONAL_CROPS_DIR / f"{alias_id}.npz"
            decision_rows.append(
                {
                    **decision,
                    "action": "COLLAPSE_ALIAS_INTO_CANONICAL",
                    "canonical_sheet": source_indexed.loc[canonical_id, "sheet_source"],
                    "alias_sheet": alias["sheet_source"],
                    "canonical_gene_category_source": canonical["gene_category"],
                    "canonical_ct_dates": ";".join(source_ct_dates(master, canonical_id)),
                    "alias_ct_dates": ";".join(source_ct_dates(master, alias_id)),
                    "canonical_cache_sha256": sha256(canonical_cache),
                    "alias_cache_sha256": sha256(alias_cache),
                    "cache_byte_identical": (
                        sha256(canonical_cache) is not None
                        and sha256(canonical_cache) == sha256(alias_cache)
                    ),
                    "canonical_coronal_npz_sha256": sha256(canonical_npz),
                    "alias_coronal_npz_sha256": sha256(alias_npz),
                    "coronal_npz_byte_identical": (
                        sha256(canonical_npz) is not None
                        and sha256(canonical_npz) == sha256(alias_npz)
                    ),
                    "anatomy_conflict_count": int(
                        sum(
                            not same_value(
                                source_indexed.loc[canonical_id, field], alias[field]
                            )
                            for field in ANATOMY_FIELDS
                            if field in source.columns
                        )
                    ),
                    "probable_match": decision["confidence"].startswith("probable"),
                }
            )

    resolved = pd.DataFrame(resolved_rows).reset_index(drop=True)
    if resolved["patient_id"].duplicated().any():
        raise AssertionError("Merged patient table still contains duplicate IDs")
    decisions = pd.DataFrame(decision_rows).sort_values(["confidence", "canonical_id"])
    conflicts = pd.DataFrame(conflict_rows).sort_values(
        ["canonical_id", "field"]
    )
    return resolved, decisions, conflicts


def build_primary_manifest(resolved: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_list = pd.read_csv(config.COHORT_LIST)
    alias_to_canonical = {
        decision["alias_id"]: decision["canonical_id"] for decision in IDENTITY_DECISIONS
    }
    cohort_list["canonical_person_id"] = cohort_list["patient_id"].replace(alias_to_canonical)

    rows = []
    for canonical_id, group in cohort_list.groupby("canonical_person_id", sort=True):
        if group["binary_label"].nunique() != 1 or group["original_class"].nunique() != 1:
            raise AssertionError(f"Label disagreement within {canonical_id}")
        rows.append(
            {
                "patient_id": canonical_id,
                "binary_label": int(group["binary_label"].iloc[0]),
                "original_class": group["original_class"].iloc[0],
                "cohort_list_ids": ";".join(sorted(group["patient_id"].astype(str))),
            }
        )
    manifest = pd.DataFrame(rows)
    resolved_index = resolved.set_index("patient_id")
    missing = sorted(set(manifest["patient_id"]) - set(resolved_index.index))
    if missing:
        raise AssertionError(f"Merged patient data missing cohort IDs: {missing}")
    manifest["sheet_source"] = manifest["patient_id"].map(
        resolved_index["sheet_source"]
    )
    manifest["cv_group_id"] = manifest["patient_id"].map(
        resolved_index["cv_group_id"]
    )
    manifest["stratum"] = (
        manifest["binary_label"].astype(str) + "|" + manifest["sheet_source"].astype(str)
    )

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    manifest["fold"] = -1
    for fold, (_, test_index) in enumerate(
        splitter.split(
            np.zeros((len(manifest), 1)),
            manifest["stratum"].to_numpy(),
            groups=manifest["cv_group_id"].to_numpy(),
        )
    ):
        manifest.loc[test_index, "fold"] = fold
    if (manifest["fold"] < 0).any():
        raise AssertionError("Outer-fold assignment is incomplete")
    if manifest.groupby("cv_group_id")["fold"].nunique().max() != 1:
        raise AssertionError("A grouped identity/family crosses outer folds")
    for fold, group in manifest.groupby("fold"):
        if group["binary_label"].nunique() != 2:
            raise AssertionError(f"Fold {fold} lacks one binary class")

    fingerprints = []
    for row in manifest.itertuples(index=False):
        cache = config.SLICE_DIR / f"{row.patient_id}.npy"
        coronal = config.CORONAL_CROPS_DIR / f"{row.patient_id}.npz"
        if not cache.exists() or not coronal.exists():
            raise AssertionError(
                f"Image missing for {row.patient_id}: "
                f"slices={cache.exists()}, coronal={coronal.exists()}"
            )
        fingerprints.append(
            {
                "patient_id": row.patient_id,
                "fold": int(row.fold),
                "cv_group_id": row.cv_group_id,
                "cache_sha256": sha256(cache),
                "coronal_npz_sha256": sha256(coronal),
            }
        )
    fingerprint_df = pd.DataFrame(fingerprints)
    for field in ("cache_sha256", "coronal_npz_sha256"):
        duplicate = fingerprint_df.loc[
            fingerprint_df[field].duplicated(keep=False)
        ].copy()
        if not duplicate.empty and duplicate.groupby(field)["cv_group_id"].nunique().max() > 1:
            raise AssertionError(
                f"Duplicate acquisition fingerprint remains across people for {field}:\n"
                f"{duplicate.to_string(index=False)}"
            )

    manifest = manifest[
        [
            "patient_id",
            "fold",
            "binary_label",
            "original_class",
            "sheet_source",
            "cv_group_id",
            "cohort_list_ids",
        ]
    ].sort_values(["fold", "patient_id"]).reset_index(drop=True)
    return manifest, fingerprint_df.sort_values(["fold", "patient_id"])


def main() -> None:
    config.ensure_dirs()
    source = pd.read_parquet(config.PATIENT_RECORDS)
    master = pd.read_parquet(config.RECORDS)
    resolved, decisions, conflicts = build_resolved_patient_table(source, master)
    manifest, fingerprints = build_primary_manifest(resolved)

    source_tested = int(source["gene_tested"].astype(bool).sum())
    resolved_tested = int(resolved["gene_tested"].astype(bool).sum())
    counts = {
        "source_record_ids": int(source["patient_id"].nunique()),
        "unique_people_after_resolution": int(resolved["patient_id"].nunique()),
        "source_records_with_genetic_test": source_tested,
        "unique_people_with_genetic_test": resolved_tested,
        "unique_people_untested": int(len(resolved) - resolved_tested),
        "collapsed_alias_pairs": int(len(IDENTITY_DECISIONS)),
        "high_confidence_or_exact_pairs": int(
            sum(not row["confidence"].startswith("probable") for row in IDENTITY_DECISIONS)
        ),
        "probable_pairs": int(
            sum(row["confidence"].startswith("probable") for row in IDENTITY_DECISIONS)
        ),
        "kept_distinct_possible_family_members": [
            list(members) for members in POSSIBLE_FAMILY_GROUPS
        ],
        "primary_unique_people": int(len(manifest)),
        "primary_SLC26A4": int((manifest["binary_label"] == 0).sum()),
        "primary_comparators": int((manifest["binary_label"] == 1).sum()),
        "primary_fold_counts": {
            str(int(fold)): int(count)
            for fold, count in manifest["fold"].value_counts().sort_index().items()
        },
        "primary_fold_SLC26A4_counts": {
            str(int(fold)): int((group["binary_label"] == 0).sum())
            for fold, group in manifest.groupby("fold")
        },
        "anatomy_resolution": (
            "one coherent bilateral block from the canonical record; alias annotations "
            "are preserved in the conflict table and never componentwise combined"
        ),
        "genetic_resolution": (
            "a documented genetic test overrides no-test; tested disagreements fail"
        ),
        "possibly_related_pair": (
            "<PATIENT_ID_19> and <PATIENT_ID_20> are kept in one group for the outer folds and the "
            "inner splits"
        ),
    }

    assert len(source) == 354
    assert len(resolved) == 345
    assert source_tested == 263 and resolved_tested == 260
    assert len(manifest) == 229
    assert int((manifest["binary_label"] == 0).sum()) == 88
    assert int((manifest["binary_label"] == 1).sum()) == 141

    resolved.to_parquet(config.PEOPLE, index=False)
    decisions.to_csv(config.DUPLICATE_RECORDS, index=False)
    conflicts.to_csv(config.DUPLICATE_RECORD_CONFLICTS, index=False)
    manifest.to_csv(config.FOLDS, index=False)
    fingerprints.to_csv(config.IMAGE_FINGERPRINTS, index=False)
    dump_json(config.DUPLICATE_RESOLUTION, counts)

    print("Merged duplicate records and assigned folds")
    print(
        f"  source IDs {len(source)} -> unique people {len(resolved)}; "
        f"tested {source_tested} -> {resolved_tested}"
    )
    print(
        f"  analysis cohort N={len(manifest)}: "
        f"{(manifest.binary_label == 0).sum()} SLC26A4, "
        f"{(manifest.binary_label == 1).sum()} comparators"
    )


if __name__ == "__main__":
    main()
