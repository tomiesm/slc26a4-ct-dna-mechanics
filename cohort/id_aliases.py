"""Write the table that maps identifiers of the variant list to the identifiers of the clinical dataset.

Each row maps an identifier used for a person in the variant list (`alt_id`) to the identifier
used for that person in the clinical dataset (`internal_id`). The table is used to match the
patients of the variant list to the patient records.

Usage: python -m cohort.id_aliases
"""

import csv
from pathlib import Path

import config

SOURCE_DATA_PROVIDER = "data_provider"

# (internal_id, alt_id, source)
KNOWN_ALIASES: list[tuple[str, str, str]] = [
    ("<PATIENT_ID_45>", "<PATIENT_ID_59>", SOURCE_DATA_PROVIDER),
    ("<PATIENT_ID_02>", "<PATIENT_ID_60>", SOURCE_DATA_PROVIDER),
    ("<PATIENT_ID_61>", "<PATIENT_ID_62>", SOURCE_DATA_PROVIDER),
    ("<PATIENT_ID_63>", "<PATIENT_ID_64>", SOURCE_DATA_PROVIDER),
    ("<PATIENT_ID_43>", "<PATIENT_ID_44>", SOURCE_DATA_PROVIDER),
    ("<PATIENT_ID_65>", "<PATIENT_ID_66>", SOURCE_DATA_PROVIDER),
    ("<PATIENT_ID_67>", "<PATIENT_ID_68>", SOURCE_DATA_PROVIDER),
    ("<PATIENT_ID_69>", "<PATIENT_ID_70>", SOURCE_DATA_PROVIDER),
]

CSV_COLUMNS = ["internal_id", "alt_id", "source"]


def _write_alias_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Write the alias rows to CSV at the given path."""
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for internal_id, alt_id, source in rows:
            writer.writerow([internal_id, alt_id, source])


def _validate_aliases(rows: list[tuple[str, str, str]]) -> None:
    """Check the alias list before writing.

    - alt_id values must be unique (each alias resolves to exactly one internal id)
    - no row may map an id to itself
    - no empty strings
    """
    alt_ids = [alt for _, alt, _ in rows]
    assert len(alt_ids) == len(set(alt_ids)), (
        f"Duplicate alt_id detected in KNOWN_ALIASES: {alt_ids}"
    )
    for internal_id, alt_id, source in rows:
        assert internal_id and alt_id and source, (
            f"Empty field in alias row: ({internal_id!r}, {alt_id!r}, {source!r})"
        )
        assert internal_id != alt_id, (
            f"Self-mapping is not allowed: {internal_id} -> {alt_id}"
        )


def main() -> None:
    config.ensure_dirs()
    print(f"[id_aliases] output CSV = {config.ID_ALIASES}")

    print("[id_aliases] validating aliases ...")
    _validate_aliases(KNOWN_ALIASES)

    print(f"[id_aliases] writing {len(KNOWN_ALIASES)} alias rows ...")
    _write_alias_csv(config.ID_ALIASES, KNOWN_ALIASES)

    print(f"[id_aliases] DONE: wrote {len(KNOWN_ALIASES)} aliases to {config.ID_ALIASES}")


if __name__ == "__main__":
    main()
