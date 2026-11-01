"""Join the variant list, the patient records of the clinical dataset and the DNA mechanics file into one row per
patient and one row per allele.

Variant-list identifiers are matched to patient-record identifiers (config.PATIENT_RECORDS) directly, after
removing a letter suffix, or through the identifier aliases; the one person listed twice
(<PATIENT_ID_43>/<PATIENT_ID_44>-H) is counted once after checking that both records are identical. Zygosity
comes from the Genotype field, and consequences are derived from the HGVS strings with anchored rules. Each
variant is matched to a slot of the DNA mechanics file by its +/-500-bp window, and the genomic coordinate and
the reference/alternative edit are checked against the slot's sequences. Every patient gets two allele rows (a
homozygous variant is repeated), and the cgNA+ descriptors are combined over the two alleles with mean, minimum
and maximum, which do not depend on allele order.

Usage: python -m variants.variant_table
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import scipy.io as sio

import config

SOURCE_OUT = config.VARIANT_DIR / "source_records.csv"
VALIDATION_OUT = config.VARIANT_DIR / "variant_table_checks.json"

TRANSCRIPT = "NM_000441.2"
GENOME_BUILD = "GRCh38/hg38"
FOUNDER = "c.919-2A>G"
STRUCTURES = ["cochlea", "vestibule", "scc", "eva", "iac", "aperture"]
VECTOR_STATS = ("mid", "mean", "std", "max", "min")
DIST_STATS = ("median", "mean", "std", "max", "min")


def jsonable(value: Any) -> Any:
    """Convert NumPy/pandas values recursively for strict JSON output."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def parse_window(value: Any) -> tuple[str, int, int]:
    match = re.fullmatch(r"(chr[^:]+):(\d+)-(\d+)", clean_text(value))
    if not match:
        raise ValueError(f"Invalid target window: {value!r}")
    chrom, start, end = match.group(1), int(match.group(2)), int(match.group(3))
    if start > end:
        raise ValueError(f"Reversed target window: {value!r}")
    return chrom, start, end


def window_string(value: Any) -> str:
    chrom, start, end = parse_window(value)
    return f"{chrom}:{start}-{end}"


def parse_locus(value: Any) -> tuple[str, int, str, str]:
    parts = clean_text(value).split("-", 3)
    if len(parts) != 4:
        raise ValueError(f"Invalid hg38 locus/ref/alt: {value!r}")
    chrom, pos_text, ref, alt = parts
    ref = "" if ref in ("", "-") else ref.upper()
    alt = "" if alt in ("", "-") else alt.upper()
    if not re.fullmatch(r"chr[^:]+", chrom):
        raise ValueError(f"Invalid chromosome in {value!r}")
    if not re.fullmatch(r"[ACGTN]*", ref) or not re.fullmatch(r"[ACGTN]*", alt):
        raise ValueError(f"Invalid ref/alt bases in {value!r}")
    if not ref and not alt:
        raise ValueError(f"Empty ref and alt in {value!r}")
    return chrom, int(pos_text), ref, alt


def classify_consequence(cdna: Any, protein: Any) -> str:
    """Assign a conservative HGVS consequence using anchored patterns.

    The variant list has HGVS but no consequence column, so the output is named a *derived*
    consequence and the raw HGVS strings are kept. Canonical splice offsets are separated from
    +5/+6 intronic variants.
    """
    c = clean_text(cdna)
    p = clean_text(protein)
    intronic = re.search(r"c\.\d+(?:_\d+)?([+-])(\d+)[ACGTN]*>[ACGTN]+", c)
    if intronic:
        offset = int(intronic.group(2))
        return "canonical_splice" if offset in (1, 2) else "intronic_near_splice"
    if re.fullmatch(r"p\.\(Met1[A-Z][a-z]{2}\)", p):
        return "start_loss"
    if "fs" in p:
        return "frameshift"
    if re.search(r"(?:del|ins|dup)", c):
        return "truncating_indel" if "Ter" in p else "indel_other"
    if "Ter" in p:
        return "nonsense"
    if re.fullmatch(r"p\.\([A-Z][a-z]{2}\d+[A-Z][a-z]{2}\)", p):
        return "missense"
    if p in ("", "p.?", "?"):
        return "unknown"
    return "other"


# Checks of the consequence rules on known cases.
assert classify_consequence("c.439A>G", "p.(Met147Val)") == "missense"
assert classify_consequence("c.2T>G", "p.(Met1Arg)") == "start_loss"
assert classify_consequence("c.919-2A>G", "p.?") == "canonical_splice"
assert classify_consequence("c.1707+5G>A", "p.?") == "intronic_near_splice"
assert classify_consequence("c.914_915insG", "p.(Ile305MetfsTer25)") == "frameshift"
assert classify_consequence("c.1520del", "p.(Leu507Ter)") == "truncating_indel"


PREDICTED_LOF = {
    "canonical_splice",
    "start_loss",
    "frameshift",
    "truncating_indel",
    "nonsense",
}


def as_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.size == 0:
        return ""
    return clean_text(value)


def as_array(patient: Any, field: str) -> np.ndarray:
    value = getattr(patient, field, None)
    if value is None or np.ndim(value) == 0:
        return np.asarray([], dtype=float)
    return np.asarray(value, dtype=float).ravel()


def array_stats(values: np.ndarray, names: tuple[str, ...] = VECTOR_STATS) -> dict[str, float]:
    if values.size == 0:
        return {name: math.nan for name in names}
    result: dict[str, float] = {}
    for name in names:
        if name == "mid":
            result[name] = float(values[values.size // 2])
        elif name == "median":
            result[name] = float(np.median(values))
        elif name == "mean":
            result[name] = float(np.mean(values))
        elif name == "std":
            result[name] = float(np.std(values))
        elif name == "max":
            result[name] = float(np.max(values))
        elif name == "min":
            result[name] = float(np.min(values))
        else:
            raise KeyError(name)
    return result


def gc_fraction(sequence: str) -> float:
    sequence = sequence.upper()
    valid = sum(base in "ACGT" for base in sequence)
    if not valid:
        return math.nan
    return sum(base in "GC" for base in sequence) / valid


def mat_payload_hash(patient: Any) -> str:
    digest = hashlib.sha256()
    for field in patient._fieldnames:
        digest.update(field.encode())
        value = getattr(patient, field)
        if isinstance(value, str):
            digest.update(value.encode())
        else:
            array = np.asarray(value)
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def target_edit_validation(patient: Any, slot: int, variant: pd.Series) -> dict[str, Any]:
    """Validate window, reference base(s), and the intended edit by alignment."""
    mat_window = window_string(getattr(patient, f"h{slot}"))
    source_window = window_string(variant["targeted variant ± 500bp"])
    chrom, start, end = parse_window(mat_window)
    locus_chrom, position, ref, alt = parse_locus(variant["Genomic loci - Ref - Alt (hg38)"])
    reference = as_string(getattr(patient, f"s0{slot}"))
    alternative = as_string(getattr(patient, f"s{slot}Alt"))
    offset = position - start

    window_exact = source_window == mat_window
    coordinate_in_window = chrom == locus_chrom and 0 <= offset <= len(reference)
    reference_length_valid = len(reference) == end - start + 1
    reference_target_valid = True if not ref else reference[offset : offset + len(ref)] == ref

    operations = [
        operation
        for operation in difflib.SequenceMatcher(
            None, reference, alternative, autojunk=False
        ).get_opcodes()
        if operation[0] != "equal"
    ]
    candidates: list[tuple[str, int]] = []
    for tag, i1, i2, j1, j2 in operations:
        ref_edit = reference[i1:i2]
        alt_edit = alternative[j1:j2]
        if ref and alt and tag == "replace" and i1 == offset and ref_edit == ref and alt_edit == alt:
            candidates.append(("substitution_exact", i1 - offset))
        elif ref and not alt and tag == "delete" and i1 == offset and ref_edit == ref:
            candidates.append(("deletion_exact", i1 - offset))
        elif not ref and alt and tag == "insert" and alt_edit == alt and abs(i1 - offset) <= 2:
            # The empty-ref insertion of the variant list lies two bases from its stated
            # coordinate in all three occurrences; the offset is recorded, not treated as exact.
            mode = "insertion_exact" if i1 == offset else "insertion_near_coordinate"
            candidates.append((mode, i1 - offset))

    target_edit_unique = len(candidates) == 1
    mode, edit_offset = candidates[0] if target_edit_unique else ("unresolved", math.nan)
    passed = all(
        (
            window_exact,
            coordinate_in_window,
            reference_length_valid,
            reference_target_valid,
            target_edit_unique,
        )
    )
    return {
        "mat_window": mat_window,
        "source_window": source_window,
        "window_exact": bool(window_exact),
        "coordinate_in_window": bool(coordinate_in_window),
        "reference_length_valid": bool(reference_length_valid),
        "reference_target_valid": bool(reference_target_valid),
        "target_edit_valid": bool(target_edit_unique),
        "target_validation_mode": mode,
        "target_alignment_offset": edit_offset,
        "sequence_validation_pass": bool(passed),
        "ref_sequence_length": len(reference),
        "alt_sequence_length": len(alternative),
    }


def variant_cgna_features(patient: Any, slot: int) -> tuple[dict[str, float], dict[str, Any]]:
    """Return per-variant cgNA descriptors before any patient-level aggregation."""
    features: dict[str, float] = {}
    checks: dict[str, Any] = {}
    modality_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for modality in ("u", "e", "p"):
        explicit_ref = as_array(patient, f"{modality}{slot}Ref")
        fallback_ref = as_array(patient, f"{modality}0{slot}")
        if explicit_ref.size:
            reference = explicit_ref
            reference_source = f"{modality}{slot}Ref"
        else:
            reference = fallback_ref
            reference_source = f"{modality}0{slot}_fallback"
        alternative = as_array(patient, f"{modality}{slot}Alt")
        if not reference.size or not alternative.size:
            raise AssertionError(f"Missing {modality} energy vector in slot {slot} of the DNA mechanics file")
        modality_arrays[modality] = (reference, alternative)
        checks[f"{modality}_reference_source"] = reference_source
        checks[f"{modality}_reference_length"] = int(reference.size)
        checks[f"{modality}_alternative_length"] = int(alternative.size)
        checks[f"{modality}_pointwise_aligned"] = bool(reference.size == alternative.size)

        point_delta = alternative - reference if reference.size == alternative.size else np.asarray([])
        for stat, value in array_stats(point_delta).items():
            features[f"d{modality}_point_{stat}"] = value

        ref_summary = array_stats(reference, DIST_STATS)
        alt_summary = array_stats(alternative, DIST_STATS)
        for stat in DIST_STATS:
            features[f"d{modality}_distribution_{stat}_shift"] = alt_summary[stat] - ref_summary[stat]

    # Alternative-state energy and occupancy-score families of the mapped variant.
    for family, vector in (
        ("u_alt", modality_arrays["u"][1]),
        ("bs", as_array(patient, f"b{slot}s")),
        ("by", as_array(patient, f"b{slot}y")),
    ):
        if not vector.size:
            raise AssertionError(f"Missing {family} vector in slot {slot} of the DNA mechanics file")
        for stat, value in array_stats(vector).items():
            features[f"{family}_{stat}"] = value

    reference_sequence = as_string(getattr(patient, f"s0{slot}"))
    alternative_sequence = as_string(getattr(patient, f"s{slot}Alt"))
    checks["reference_gc"] = gc_fraction(reference_sequence)
    checks["alternative_gc"] = gc_fraction(alternative_sequence)
    checks["any_energy_reference_fallback"] = any(
        str(value).endswith("_fallback")
        for key, value in checks.items()
        if key.endswith("_reference_source")
    )
    return features, checks


def strict_two_allele_aggregate(values: list[float]) -> tuple[float, float, float]:
    """Mean/min/max only when both biological alleles have finite values."""
    array = np.asarray(values, dtype=float)
    if array.shape != (2,) or not np.all(np.isfinite(array)):
        return math.nan, math.nan, math.nan
    return float(np.mean(array)), float(np.min(array)), float(np.max(array))


def resolve_ids(record_ids: set[str]) -> tuple[dict[str, str], Any]:
    alias_frame = pd.read_csv(config.ID_ALIASES)
    assert {"internal_id", "alt_id"}.issubset(alias_frame.columns)
    alt_to_internal = {
        clean_text(row.alt_id): clean_text(row.internal_id)
        for _, row in alias_frame.iterrows()
    }

    def resolve(source_id: Any) -> tuple[str | None, str]:
        original = clean_text(source_id)
        stripped = re.sub(r"-[A-Za-z]+$", "", original)
        if original in record_ids:
            return original, "direct"
        if stripped in record_ids:
            return stripped, "suffix_stripped_direct"
        if original in alt_to_internal and alt_to_internal[original] in record_ids:
            return alt_to_internal[original], "alias"
        if stripped in alt_to_internal and alt_to_internal[stripped] in record_ids:
            return alt_to_internal[stripped], "suffix_stripped_alias"
        return None, "unresolved"

    return alt_to_internal, resolve


def main() -> None:
    config.ensure_dirs()
    raw = pd.read_excel(config.VARIANT_LIST)
    required = {
        "File ID",
        "Patient_ID",
        "Variants",
        "Genomic loci - Ref - Alt (hg38)",
        "Genotype",
        " Gene (MANE or MANE Plus Clinical select transcript)",
        "HGVS (cDNA)",
        "HGVS (protein)",
        "targeted variant ± 500bp",
    }
    missing = required.difference(raw.columns)
    assert not missing, f"Variant list columns missing: {sorted(missing)}"

    raw[["File ID", "Patient_ID"]] = raw[["File ID", "Patient_ID"]].ffill()
    annotated = raw[raw["HGVS (cDNA)"].notna()].copy()
    annotated["pnum"] = pd.to_numeric(
        annotated["File ID"].str.extract(r"SLC_P(\d+)")[0], errors="raise"
    ).astype(int)
    duplicate_key = [
        "pnum",
        "Variants",
        "Genomic loci - Ref - Alt (hg38)",
        "HGVS (cDNA)",
        "HGVS (protein)",
        "Genotype",
    ]
    duplicate_annotation_rows = int(annotated.duplicated(duplicate_key).sum())
    annotated = annotated.drop_duplicates(duplicate_key).copy()
    assert set(annotated.pnum) == set(range(1, 61)), "Expected variant-list records SLC_P1..P60, as in the DNA mechanics file"
    transcript_values = annotated[
        " Gene (MANE or MANE Plus Clinical select transcript)"
    ].dropna().astype(str)
    assert transcript_values.str.contains(re.escape(TRANSCRIPT), regex=True).all()

    mat = sio.loadmat(config.DNA_MECHANICS, squeeze_me=True, struct_as_record=False)
    mat_patients = np.atleast_1d(mat["patient"])
    assert len(mat_patients) == 60

    patient_records = pd.read_parquet(config.PATIENT_RECORDS).copy()
    assert patient_records.patient_id.notna().all() and patient_records.patient_id.is_unique
    patient_records["patient_id"] = patient_records.patient_id.astype(str)
    patient_records = patient_records.set_index("patient_id")
    _, resolve = resolve_ids(set(patient_records.index))

    source_records: list[dict[str, Any]] = []
    source_variants: dict[int, list[dict[str, Any]]] = {}

    for pnum, group in annotated.groupby("pnum", sort=True):
        group = group.copy().reset_index(drop=True)
        source_id = clean_text(group.loc[0, "Patient_ID"])
        internal_id, resolution_mode = resolve(source_id)
        genotypes = set(group.Genotype.map(clean_text))
        if len(group) == 1:
            assert genotypes == {"1/1 (Homo)"}, (pnum, genotypes)
            zygosity = "homozygous"
        else:
            assert len(group) == 2 and genotypes == {"0/1 (Het)"}, (pnum, genotypes, len(group))
            zygosity = "compound_heterozygous"

        patient = mat_patients[pnum - 1]
        mat_slots = {
            slot: window_string(getattr(patient, f"h{slot}"))
            for slot in (1, 2)
            if as_string(getattr(patient, f"h{slot}", ""))
        }
        workbook_windows = [window_string(v) for v in group["targeted variant ± 500bp"]]

        if len(group) == 1:
            mapping_mode = "single_direct" if workbook_windows[0] == mat_slots.get(1) else "unresolved"
        elif workbook_windows == [mat_slots.get(1), mat_slots.get(2)]:
            mapping_mode = "direct"
        elif workbook_windows == [mat_slots.get(2), mat_slots.get(1)]:
            mapping_mode = "reversed"
        else:
            mapping_mode = "coordinate_fallback"

        mapped: list[dict[str, Any]] = []
        used_slots: set[int] = set()
        for workbook_index, variant in group.iterrows():
            source_window = window_string(variant["targeted variant ± 500bp"])
            candidates = [slot for slot, value in mat_slots.items() if value == source_window]
            if not candidates and mapping_mode == "coordinate_fallback":
                chrom, position, _, _ = parse_locus(variant["Genomic loci - Ref - Alt (hg38)"])
                candidates = [
                    slot
                    for slot, value in mat_slots.items()
                    if (lambda w: w[0] == chrom and w[1] <= position <= w[2])(parse_window(value))
                ]
            candidates = [slot for slot in candidates if slot not in used_slots]
            if len(candidates) != 1:
                raise AssertionError(
                    f"Variant not matched to a slot of the DNA mechanics file: pnum={pnum}, "
                    f"HGVS={variant['HGVS (cDNA)']}, workbook_window={source_window}, "
                    f"file_slots={mat_slots}, candidates={candidates}"
                )
            slot = candidates[0]
            used_slots.add(slot)
            sequence_checks = target_edit_validation(patient, slot, variant)
            assert sequence_checks["sequence_validation_pass"], (pnum, slot, sequence_checks)
            cgna, vector_checks = variant_cgna_features(patient, slot)
            chrom, position, ref, alt = parse_locus(variant["Genomic loci - Ref - Alt (hg38)"])
            consequence = classify_consequence(variant["HGVS (cDNA)"], variant["HGVS (protein)"])
            record: dict[str, Any] = {
                "pnum": int(pnum),
                "source_patient_id": source_id,
                "patient_id": internal_id,
                "workbook_allele_index": int(workbook_index + 1),
                "workbook_variant_label": clean_text(variant["Variants"]),
                "mat_slot": int(slot),
                "mapping_mode": mapping_mode,
                "transcript": TRANSCRIPT,
                "genome_build": GENOME_BUILD,
                "hgvs_cdna": clean_text(variant["HGVS (cDNA)"]),
                "hgvs_protein": clean_text(variant["HGVS (protein)"]),
                "chromosome": chrom,
                "position_hg38": position,
                "reference_allele": ref or "-",
                "alternate_allele": alt or "-",
                "source_genotype": clean_text(variant["Genotype"]),
                "zygosity": zygosity,
                "derived_consequence": consequence,
                "predicted_lof": int(consequence in PREDICTED_LOF),
                **sequence_checks,
                **vector_checks,
            }
            record.update({f"cgna_var_{name}": value for name, value in cgna.items()})
            mapped.append(record)

        expected_slots = {1} if zygosity == "homozygous" else {1, 2}
        assert used_slots == expected_slots, (pnum, used_slots, expected_slots)
        source_variants[int(pnum)] = mapped
        signature = "|".join(
            sorted(f"{row['hgvs_cdna']}@{row['position_hg38']}" for row in mapped)
        )
        source_records.append(
            {
                "pnum": int(pnum),
                "source_patient_id": source_id,
                "patient_id": internal_id,
                "id_resolution_mode": resolution_mode,
                "has_patient_record": internal_id is not None,
                "zygosity": zygosity,
                "genotype_signature": signature,
                "mapping_mode": mapping_mode,
                "mat_payload_sha256": mat_payload_hash(patient),
                "n_workbook_variants": int(len(group)),
                "phase_status": (
                    "homozygous_explicit_in_workbook"
                    if zygosity == "homozygous"
                    else "trans_phase_not_explicitly_encoded"
                ),
            }
        )

    source_frame = pd.DataFrame(source_records).sort_values("pnum").reset_index(drop=True)
    assert not (source_frame.mapping_mode == "unresolved").any()

    # Resolve duplicate records after checking that genotype and DNA mechanics data are identical.
    matched = source_frame[source_frame.has_patient_record].copy()
    duplicate_resolution: list[dict[str, Any]] = []
    selected_pnums: list[int] = []
    for patient_id, group in matched.groupby("patient_id", sort=True):
        if len(group) == 1:
            selected_pnums.append(int(group.pnum.iloc[0]))
            continue
        assert group.genotype_signature.nunique() == 1
        assert group.mat_payload_sha256.nunique() == 1
        # The only person listed twice; counted once.
        assert patient_id == "<PATIENT_ID_43>" and set(group.pnum) == {2, 37}
        direct = group[group.source_patient_id == patient_id]
        chosen = int(direct.pnum.iloc[0]) if len(direct) == 1 else int(group.pnum.min())
        selected_pnums.append(chosen)
        duplicate_resolution.append(
            {
                "patient_id": patient_id,
                "source_pnums": sorted(group.pnum.astype(int).tolist()),
                "source_patient_ids": sorted(group.source_patient_id.tolist()),
                "selected_pnum": chosen,
                "genotype_identical": True,
                "mat_payload_identical": True,
            }
        )

    selected = source_frame[source_frame.pnum.isin(selected_pnums)].copy()
    assert selected.patient_id.notna().all() and selected.patient_id.is_unique
    assert len(selected) == 35, f"Expected 35 unique patients, got {len(selected)}"
    source_frame["selected_for_analysis"] = source_frame.pnum.isin(selected_pnums)
    source_frame["duplicate_internal_id"] = source_frame.patient_id.duplicated(False) & source_frame.patient_id.notna()

    variant_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    swap_failures: list[str] = []

    for source in selected.itertuples(index=False):
        base_variants = source_variants[int(source.pnum)]
        if source.zygosity == "homozygous":
            assert len(base_variants) == 1
            biological_variants = [dict(base_variants[0]), dict(base_variants[0])]
            biological_variants[0]["homozygous_feature_replicated"] = False
            biological_variants[1]["homozygous_feature_replicated"] = True
        else:
            assert len(base_variants) == 2
            biological_variants = [dict(base_variants[0]), dict(base_variants[1])]
            for row in biological_variants:
                row["homozygous_feature_replicated"] = False

        for allele_index, row in enumerate(biological_variants, start=1):
            row["allele_index"] = allele_index
            variant_rows.append(row)

        patient_id = str(source.patient_id)
        phenotype = patient_records.loc[patient_id]
        row: dict[str, Any] = {
            "patient_id": patient_id,
            "canonical_pnum": int(source.pnum),
            "source_pnums": ";".join(
                map(
                    str,
                    sorted(
                        matched.loc[matched.patient_id == patient_id, "pnum"].astype(int).tolist()
                    ),
                )
            ),
            "source_patient_ids": ";".join(
                sorted(matched.loc[matched.patient_id == patient_id, "source_patient_id"].tolist())
            ),
            "zygosity": source.zygosity,
            "phase_status": source.phase_status,
        }
        cdnas = [variant["hgvs_cdna"] for variant in biological_variants]
        consequences = [variant["derived_consequence"] for variant in biological_variants]
        positions = [float(variant["position_hg38"]) for variant in biological_variants]
        ref_gc = [float(variant["reference_gc"]) for variant in biological_variants]
        alt_gc = [float(variant["alternative_gc"]) for variant in biological_variants]
        founder_dosage = sum(cdna == FOUNDER for cdna in cdnas)
        if founder_dosage == 2:
            allelic_configuration = "c919_homozygous"
        elif founder_dosage == 1:
            allelic_configuration = "c919_compound"
        elif source.zygosity == "homozygous":
            allelic_configuration = "non_c919_homozygous"
        else:
            allelic_configuration = "non_c919_compound"
        row.update(
            {
                "allele1_cdna": cdnas[0],
                "allele2_cdna": cdnas[1],
                "allele1_consequence": consequences[0],
                "allele2_consequence": consequences[1],
                "genotype_pair": "|".join(sorted(cdnas)),
                "consequence_pair": "|".join(sorted(consequences)),
                "c919_dosage": founder_dosage,
                "allelic_configuration": allelic_configuration,
                "predicted_lof_allele_count": sum(c in PREDICTED_LOF for c in consequences),
                "position_mean": float(np.mean(positions)),
                "position_min": float(np.min(positions)),
                "position_max": float(np.max(positions)),
                "position_range": float(np.ptp(positions)),
                "reference_gc_mean": float(np.mean(ref_gc)),
                "reference_gc_range": float(np.ptp(ref_gc)),
                "alternative_gc_mean": float(np.mean(alt_gc)),
                "alternative_gc_range": float(np.ptp(alt_gc)),
            }
        )
        nonfounder = [
            variant["derived_consequence"]
            for variant in biological_variants
            if variant["hgvs_cdna"] != FOUNDER
        ]
        row["c919_partner_consequence"] = nonfounder[0] if len(nonfounder) == 1 else ""

        for structure in STRUCTURES:
            grades = phenotype[[f"{structure}_R", f"{structure}_L"]].astype(float)
            row[f"{structure}_worst"] = float(grades.max())
        row["composite_sum_sensitivity"] = float(
            sum(row[f"{structure}_worst"] for structure in STRUCTURES)
        )
        row["vestibule_enlarged"] = int(row["vestibule_worst"] == 3)
        row["cochlea_abnormal"] = int(row["cochlea_worst"] > 0)

        feature_names = sorted(
            key.removeprefix("cgna_var_")
            for key in biological_variants[0]
            if key.startswith("cgna_var_")
        )
        for feature in feature_names:
            original_values = [
                float(variant[f"cgna_var_{feature}"]) for variant in biological_variants
            ]
            reversed_values = list(reversed(original_values))
            aggregate = strict_two_allele_aggregate(original_values)
            swapped = strict_two_allele_aggregate(reversed_values)
            if not all(np.isclose(a, b, equal_nan=True) for a, b in zip(aggregate, swapped)):
                swap_failures.append(f"{patient_id}:{feature}")
            for suffix, value in zip(("mean", "min", "max"), aggregate):
                row[f"cgna_{feature}_allele_{suffix}"] = value
        patient_rows.append(row)

    variants = pd.DataFrame(variant_rows).sort_values(["patient_id", "allele_index"]).reset_index(drop=True)
    patients = pd.DataFrame(patient_rows).sort_values("patient_id").reset_index(drop=True)

    # Checks of the analysis population and endpoint.
    assert patients.patient_id.is_unique and len(patients) == 35
    assert len(variants) == 70
    assert (variants.groupby("patient_id").size() == 2).all()
    assert not swap_failures
    assert set(patients.vestibule_worst.unique()).issubset({0.0, 3.0})
    assert int((patients.allelic_configuration == "c919_homozygous").sum()) == 9
    assert int((patients.zygosity == "compound_heterozygous").sum()) == 26
    assert variants.sequence_validation_pass.all()

    cgna_columns = [column for column in patients if column.startswith("cgna_")]
    mapping_counts_all = source_frame.mapping_mode.value_counts().to_dict()
    mapping_counts_analyzed = source_frame.loc[
        source_frame.selected_for_analysis, "mapping_mode"
    ].value_counts().to_dict()
    validation_modes = variants.target_validation_mode.value_counts().to_dict()
    source_variant_rows = [row for rows in source_variants.values() for row in rows]
    source_validation_modes = Counter(
        row["target_validation_mode"] for row in source_variant_rows
    )
    assert len(source_variant_rows) == 108
    assert all(row["sequence_validation_pass"] for row in source_variant_rows)
    consequence_counts = variants.derived_consequence.value_counts().to_dict()
    fallback_counts = {
        modality: int(variants[f"{modality}_reference_source"].str.endswith("_fallback").sum())
        for modality in ("u", "e", "p")
    }
    validation = {
        "analysis": "variant_table",
        "source_files": {
            "workbook": str(config.VARIANT_LIST),
            "dna_mechanics": str(config.DNA_MECHANICS),
            "patient_records": str(config.PATIENT_RECORDS),
            "aliases": str(config.ID_ALIASES),
        },
        "transcript": TRANSCRIPT,
        "genome_build": GENOME_BUILD,
        "source_counts": {
            "raw_workbook_rows": int(len(raw)),
            "annotated_rows_before_exact_dedup": int(len(raw[raw["HGVS (cDNA)"].notna()])),
            "duplicate_annotation_rows_removed": duplicate_annotation_rows,
            "records_in_dna_mechanics_file": int(len(source_frame)),
            "records_matched_to_patient_records": int(source_frame.has_patient_record.sum()),
            "unique_patients_analyzed": int(len(patients)),
            "biological_allele_rows_analyzed": int(len(variants)),
        },
        "source_duplicate_resolution": duplicate_resolution,
        "allele_order_mapping_counts_all_60": {
            "direct": int(mapping_counts_all.get("direct", 0)),
            "reversed": int(mapping_counts_all.get("reversed", 0)),
            "single_direct": int(mapping_counts_all.get("single_direct", 0)),
            "coordinate_fallback": int(mapping_counts_all.get("coordinate_fallback", 0)),
            "unresolved": int(mapping_counts_all.get("unresolved", 0)),
        },
        "allele_order_mapping_counts_analyzed": {
            "direct": int(mapping_counts_analyzed.get("direct", 0)),
            "reversed": int(mapping_counts_analyzed.get("reversed", 0)),
            "single_direct": int(mapping_counts_analyzed.get("single_direct", 0)),
            "coordinate_fallback": int(mapping_counts_analyzed.get("coordinate_fallback", 0)),
            "unresolved": int(mapping_counts_analyzed.get("unresolved", 0)),
        },
        "sequence_target_validation": {
            "all_pass": bool(variants.sequence_validation_pass.all()),
            "analyzed_70_biological_alleles_mode_counts": validation_modes,
            "all_108_unique_source_variant_occurrences": {
                "all_pass": all(row["sequence_validation_pass"] for row in source_variant_rows),
                "mode_counts": dict(source_validation_modes),
            },
            "near_coordinate_insertion_offsets": sorted(
                variants.loc[
                    variants.target_validation_mode == "insertion_near_coordinate",
                    "target_alignment_offset",
                ].astype(int).unique().tolist()
            ),
        },
        "energy_reference_vector_fallback_allele_counts": fallback_counts,
        "consequence_counts_across_70_biological_alleles": consequence_counts,
        "patient_counts": {
            "c919_homozygous": int((patients.allelic_configuration == "c919_homozygous").sum()),
            "c919_compound": int((patients.allelic_configuration == "c919_compound").sum()),
            "non_c919_compound": int((patients.allelic_configuration == "non_c919_compound").sum()),
            "compound_heterozygous_total": int((patients.zygosity == "compound_heterozygous").sum()),
        },
        "cgna": {
            "per_variant_descriptors": int(len([c for c in variants if c.startswith("cgna_var_")])),
            "swap_invariant_patient_descriptors": int(len(cgna_columns)),
            "strict_two_allele_rule": "mean/min/max are missing unless both biological allele values are finite",
            "patients_checked_for_swap_invariance": int(len(patients)),
            "swap_invariance_failures": swap_failures,
            "all_analyzed_records_swap_invariant": not swap_failures,
            "complete_n_by_descriptor": {
                column: int(patients[column].notna().sum()) for column in cgna_columns
            },
        },
        "source_limitations": {
            "phase": (
                "The variant list explicitly marks homozygotes but gives each variant in two-variant records as 0/1; "
                "it does not provide a separate trans-phase field. The compound designation follows the case list "
                "and is not independently phased here."
            ),
            "pathogenicity": (
                "The variant list has no pathogenicity/classification column; no external classification was imputed."
            ),
            "consequence": (
                "Consequence categories are derived from the HGVS cDNA/protein strings of the variant list, not "
                "taken from a consequence field."
            ),
        },
    }

    source_frame.to_csv(SOURCE_OUT, index=False)
    variants.to_csv(config.VARIANTS, index=False)
    patients.to_csv(config.VARIANT_TABLE, index=False)
    with VALIDATION_OUT.open("w") as handle:
        json.dump(jsonable(validation), handle, indent=2, allow_nan=False)

    print(f"  records in the DNA mechanics file: {len(source_frame)}")
    print(f"  records matched to a patient record: {source_frame.has_patient_record.sum()} -> unique patients: {len(patients)}")
    print(f"  duplicate resolution: {duplicate_resolution}")
    print(f"  allele order in the DNA mechanics file (all): {validation['allele_order_mapping_counts_all_60']}")
    print(f"  target validation: {validation_modes}")
    homozygous = patients.allelic_configuration == "c919_homozygous"
    compound = patients.zygosity == "compound_heterozygous"
    print(
        "  endpoint: c.919 homo "
        f"{int(patients.loc[homozygous, 'vestibule_enlarged'].sum())}/{int(homozygous.sum())} vs compounds "
        f"{int(patients.loc[compound, 'vestibule_enlarged'].sum())}/{int(compound.sum())} vestibular enlargement"
    )
    print(f"  swap-invariant cgNA descriptors: {len(cgna_columns)}; failures={len(swap_failures)}")
    print(f"  saved {config.VARIANT_TABLE}")
    print(f"  saved {config.VARIANTS}")
    print(f"  saved {SOURCE_OUT}")
    print(f"  saved {VALIDATION_OUT}")


if __name__ == "__main__":
    main()
