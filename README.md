# SLC26A4 temporal bone CT: deep learning, expert grading and DNA mechanics

Analysis code for the article

> Iešmantas T, et al. *Evaluating deep learning, expert CT grading, and variant DNA mechanics in SLC26A4-related hearing loss.* Submitted to PLOS Digital Health.

The study used 229 genetically tested patients from National Taiwan University Hospital. It compares two ResNet-18 convolutional neural networks trained on coronal temporal bone CT with a logistic regression on otologists' grades of six inner ear structures, for identifying *SLC26A4*-positive patients. In 35 patients with detailed variant data, it relates genotype groups and calculated DNA mechanics descriptors of the variants to vestibular enlargement.

## Layout

| Folder | Contents |
| --- | --- |
| `config.py` | Locations of all inputs and results, and shared settings |
| `restore_ids.py` | Puts the patient identifiers back into the code (see Data) |
| `cohort/` | Reading the clinical dataset, genetic results, patient table, duplicate records, analysis cohort and folds |
| `imaging/` | Coronal CT series: inventory, cropping and resized slices |
| `grade_models/` | Logistic regression on the CT grades, sensitivity analyses, four-structure model and its error review |
| `cnn/` | Training of the two CNNs and the analyses of their scores |
| `variants/` | Variant table, DNA descriptors, genotype groups, protein consequence and published transport measurements |

## Data

No patient data are included. The clinical, imaging and genetic data cannot be shared publicly; the data availability statement of the article explains how to request access.

Patient identifiers written in the code (exclusion lists, duplicate records, identifier aliases and a few named patients) are replaced by numbered placeholders `<PATIENT_ID_nn>`. The numbers follow the mapping file and are not contiguous. The mapping file is kept with the restricted data as `DATA_DIR/id_mapping.json`. Before running the analysis, run `python restore_ids.py` once in a local copy of the repository; it replaces the placeholders in the `.py` files, so that copy must not be shared.

The scripts read the inputs from `DATA_DIR` and write the results to `RESULTS_DIR` (environment variables; defaults `./data` and `./results`):

```
DATA_DIR/
  clinical_dataset.xlsx       clinical records, one row per ear (sheets "malformation" and "normal")
  ct_axial/                   axial CT volumes, <patient>_<YYYYMMDD>[ <implant note>].npz (only the file names are read)
  ct_png/<scan>/<series>/     exported CT series as PNG files; <scan> is the stem of the axial file name
  coronal_crops/              cropped coronal series, one <patient>.npz each (made by imaging.crop_slices)
  cohort_list.csv             records in the analysis cohort, with SLC26A4 status
  data_corrections.csv        documented corrections of recorded values
  variants/variant_list.xlsx  SLC26A4 variants of the patients with variant data
  variants/dna_mechanics.mat  cgNA+ nucleosome wrapping energy profiles and occupancy scores
  id_mapping.json             placeholder -> patient identifier (used by restore_ids.py)
```

- `cohort_list.csv` has the columns `patient_id`, `binary_label` (0 = *SLC26A4*-positive, 1 = other tested patient) and `original_class` (`SLC26A4`, `other_known_gene` or `negative`).
- `data_corrections.csv` has the columns `patient_id`, `field` (only `birthday`), `original_value`, `corrected_value`, `evidence` and `status`.
- Scan folders in `ct_png/` whose name contains `CI` (scans after cochlear implantation) are not cropped.

The DNA mechanics calculations themselves were done with the cgNA+ model before this analysis. The crops in `coronal_crops/` are supplied with the data; `imaging.crop_slices` is the script that made them and the only script that writes into `DATA_DIR`.

## Software

Python 3.12, with two sets of package versions:

- `requirements.txt`: all steps except the variant subset, including the CNNs (PyTorch 2.6.0 with CUDA 12.4).
- `requirements-variants.txt`: the variant subset (`variants/`).

With these versions the scripts reproduce the reported values. Other versions can change some values in the third decimal, because the penalty chosen by cross-validation and the logistic predictions depend on the scikit-learn and NumPy versions. All steps were run with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

The CNNs were trained on NVIDIA RTX A6000 and RTX 4000 Ada GPUs; one complete run took about 4.4 hours on four GPUs. CNN results depend on GPU nondeterminism and mixed precision, so a rerun should select the same epochs but may not reproduce every digit.

## Running

Run each step from the repository root, for example `python -m cohort.read_dataset`, in this order:

1. **Cohort:** `cohort.read_dataset`, `cohort.classify_genetics`, `cohort.patient_table`, `cohort.id_aliases`.
2. **Images:** `imaging.png_inventory`, then `imaging.slice_cache`. To remake the crops from the PNG series, run `imaging.crop_slices` between these two and delete `RESULTS_DIR/images/slices_300x768/` first, because `imaging.slice_cache` skips patients that already have resized slices.
3. **Analysis cohort and folds:** `cohort.build_cohort`.
4. **Grade models:** `grade_models.analysis` (it also writes the cohort flow to `RESULTS_DIR/cohort/`), `grade_models.alternative_grades`, `grade_models.sensitivity`, `grade_models.encoding`, `grade_models.tuning`, then `grade_models.four_structures freeze`, `run` and `verify` (see `grade_models/four_structures_protocol.md`).
5. **CNNs (GPU):** `cnn.train`, `cnn.seed_runs`, `cnn.anatomy_outputs`. `python -m cnn.train --onejob MODE FOLD GPU` trains a single fold, for testing; it overwrites that fold's checkpoint and predictions, and the pooled prediction files that later steps read are written only by the full run.
6. **Variant subset:** `variants.variant_table`, `variants.analysis`, `variants.anatomy_by_group`, `variants.descriptor_prediction`, `variants.descriptor_ranking`, `variants.protein_consequence`, `variants.pendrin_transport`.
7. **Analyses that combine the above:** `grade_models.four_structure_errors`, `cnn.added_value`, `cnn.added_value_seeds`, `cnn.related_pair_and_cnn_comparison`, `cnn.by_genotype_group`.

`grade_models.four_structures` and `grade_models.four_structure_errors` do not overwrite finished results; delete `RESULTS_DIR/grade_models/four_structures/` and `four_structure_errors/` before running them again.

`python -m grade_models.test_four_structures` checks the four-structure model (its last test needs the results of steps 3 and 4).

## Main results

- Grade model with six structures: AUC 0.978 (95% CI 0.954 to 0.995).
- CNNs: AUC 0.681 (single-task) and 0.713 (multitask); neither added information to the grades.
- Variant subset: vestibular enlargement in 1 of 9 c.919-2A>G homozygotes and 19 of 26 patients with two different variants.

## Licence

MIT; see `LICENSE`.

## Contact

Tomas Iešmantas, Department of Applied Mathematics, Kaunas University of Technology, Lithuania (tomas.iesmantas@ktu.lt; ORCID 0000-0002-1237-2448).
