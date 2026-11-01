"""Locations of the input data and of all results, and settings shared by several scripts.

Set DATA_DIR to the folder with the restricted input data and RESULTS_DIR to the folder for the
results (defaults: ./data and ./results). Every script reads and writes files only under the
folders defined here.
"""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))

# ── Inputs (restricted; not distributed) ─────────────────────────────────────────────────────────
CLINICAL_DATASET = DATA_DIR / "clinical_dataset.xlsx"  # sheets "malformation" and "normal", one row per ear
CT_AXIAL_DIR = DATA_DIR / "ct_axial"  # axial CT volumes <patient>_<YYYYMMDD>[ <implant note>].npz (file names used)
CT_PNG_DIR = DATA_DIR / "ct_png"  # exported CT series, <scan>/<series>/*.png
CORONAL_CROPS_DIR = DATA_DIR / "coronal_crops"  # cropped coronal series, one <patient>.npz each (imaging/crop_slices.py)
COHORT_LIST = DATA_DIR / "cohort_list.csv"  # records included in the analysis, with their SLC26A4 status
DATA_CORRECTIONS = DATA_DIR / "data_corrections.csv"  # documented corrections of recorded values
VARIANT_LIST = DATA_DIR / "variants" / "variant_list.xlsx"  # SLC26A4 variants of the patients with variant data
DNA_MECHANICS = DATA_DIR / "variants" / "dna_mechanics.mat"  # cgNA+ wrapping energy profiles and occupancy scores

# ── Cohort ───────────────────────────────────────────────────────────────────────────────────────
COHORT_DIR = RESULTS_DIR / "cohort"
RECORDS = COHORT_DIR / "records.parquet"  # one row per ear and CT date
CT_MANIFEST = COHORT_DIR / "ct_manifest.csv"  # axial CT files per patient and date
GENE_MAPPING = COHORT_DIR / "gene_mapping.json"
PATIENT_RECORDS = COHORT_DIR / "patient_records.parquet"  # one row per patient record
ID_ALIASES = COHORT_DIR / "id_aliases.csv"  # identifiers of the variant list mapped to those of the clinical dataset
PEOPLE = COHORT_DIR / "people.parquet"  # one row per person, duplicate records merged
DUPLICATE_RECORDS = COHORT_DIR / "duplicate_records.csv"
DUPLICATE_RECORD_CONFLICTS = COHORT_DIR / "duplicate_record_conflicts.csv"
DUPLICATE_RESOLUTION = COHORT_DIR / "duplicate_resolution.json"
FOLDS = COHORT_DIR / "folds.csv"  # analysis cohort (229 patients) with labels and cross-validation folds
IMAGE_FINGERPRINTS = COHORT_DIR / "image_fingerprints.csv"
COHORT_EXCLUSIONS = COHORT_DIR / "exclusions.csv"
COHORT_FLOW = COHORT_DIR / "flow.json"

# ── Images ───────────────────────────────────────────────────────────────────────────────────────
IMAGES_DIR = RESULTS_DIR / "images"
PNG_INVENTORY = IMAGES_DIR / "png_inventory.csv"
SLICE_DIR = IMAGES_DIR / "slices_300x768"  # resized coronal slices, one <patient>.npy each
SLICE_HEIGHT, SLICE_WIDTH = 300, 768

# ── Grade models ─────────────────────────────────────────────────────────────────────────────────
GRADE_DIR = RESULTS_DIR / "grade_models"
GRADE_PREDICTIONS = GRADE_DIR / "predictions.csv"  # held-out predictions of all grade models
GRADE_METRICS = GRADE_DIR / "metrics.json"
AGE_CORRECTION = GRADE_DIR / "age_correction.json"
AGE_AND_WITHOUT_EVA = GRADE_DIR / "age_and_without_eva.json"
FORWARD_SELECTION = GRADE_DIR / "forward_selection.json"
FOUR_STRUCTURE_DIR = GRADE_DIR / "four_structures"
FOUR_STRUCTURE_ERRORS_DIR = GRADE_DIR / "four_structure_errors"

# ── CNNs ─────────────────────────────────────────────────────────────────────────────────────────
CNN_DIR = RESULTS_DIR / "cnn"
CNN_CHECKPOINT_DIR = CNN_DIR / "checkpoints"
CNN_PREDICTION_DIR = CNN_DIR / "predictions"
CNN_SUMMARY = CNN_DIR / "summary.json"
CNN_SEED = 42
N_FOLDS = 5


def cnn_predictions(mode):
    """Held-out SLC26A4 scores of one CNN ("single" or "multitask"), all folds (written by cnn.train)."""
    return CNN_PREDICTION_DIR / f"{mode}_oof.csv"


def cnn_fold_predictions(mode, fold, seed_run=1):
    """Held-out SLC26A4 scores of one fold; seed run 1 is cnn.train, seed runs 2 and 3 are cnn.seed_runs."""
    suffix = "" if seed_run == 1 else f"_seed{seed_run}"
    return CNN_PREDICTION_DIR / f"{mode}_fold{fold}{suffix}.csv"


# ── Variant subset ───────────────────────────────────────────────────────────────────────────────
VARIANT_DIR = RESULTS_DIR / "variants"
VARIANT_TABLE = VARIANT_DIR / "variant_table.csv"  # one row per patient with variant data, with DNA descriptors
VARIANTS = VARIANT_DIR / "variants.csv"  # one row per variant
DESCRIPTOR_TESTS = VARIANT_DIR / "descriptor_tests.csv"
DESCRIPTOR_TESTS_WITHIN_GROUPS = VARIANT_DIR / "descriptor_tests_within_groups.csv"
DESCRIPTOR_PREDICTIONS = VARIANT_DIR / "descriptor_predictions.csv"
ANATOMY_BY_GROUP = VARIANT_DIR / "anatomy_by_group.json"


def ensure_dirs():
    """Create the results folders."""
    for folder in (COHORT_DIR, IMAGES_DIR, SLICE_DIR, GRADE_DIR, FOUR_STRUCTURE_DIR, FOUR_STRUCTURE_ERRORS_DIR,
                   CNN_DIR, CNN_CHECKPOINT_DIR, CNN_PREDICTION_DIR, VARIANT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
