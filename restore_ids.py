"""Put the patient identifiers back into the code before running it on the restricted data.

In this public copy, the patient identifiers written in the code (exclusion lists, duplicate records, identifier
aliases and a few named patients) are replaced by numbered placeholders of the form <PATIENT_ID_nn>. The numbers
follow the mapping file and are not contiguous. The mapping file is kept with the restricted data; it is a JSON
object from each placeholder to the identifier used in the clinical dataset or the variant list.

The script replaces the placeholders in config.py and in the .py files of the analysis folders, in place. It checks
the whole mapping before writing and stops without changing anything if a placeholder is missing. The restored copy
contains patient identifiers and must not be shared.

Usage: python restore_ids.py [mapping.json]    (default: DATA_DIR/id_mapping.json)
"""

import json
import os
import re
import sys
from pathlib import Path

import config

PLACEHOLDER = re.compile(r"<PATIENT_ID_\d+>")
FOLDERS = ("cohort", "imaging", "grade_models", "cnn", "variants")


def main(mapping_path):
    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    if not all(isinstance(v, str) and v for v in mapping.values()):
        sys.exit(f"{mapping_path}: every identifier must be a non-empty string")

    root = Path(__file__).resolve().parent
    files = [root / "config.py"] + [p for folder in FOLDERS for p in sorted((root / folder).glob("*.py"))]
    texts = {p: p.read_text(encoding="utf-8") for p in files}
    used = {m for text in texts.values() for m in PLACEHOLDER.findall(text)}
    missing = sorted(used - mapping.keys())
    if missing:
        sys.exit(f"Not in {mapping_path}: {', '.join(missing)}")

    for path, text in texts.items():
        found = PLACEHOLDER.findall(text)
        if found:
            tmp = path.with_suffix(".py.tmp")
            tmp.write_text(PLACEHOLDER.sub(lambda m: mapping[m.group(0)], text), encoding="utf-8")
            os.replace(tmp, path)
            print(f"{path.relative_to(root)}: {len(found)} replaced")
    print(f"{len(used)} identifiers restored")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else config.DATA_DIR / "id_mapping.json")
