# Grade model without the EVA and cochlear grades

Post-hoc sensitivity analysis with a design fixed before fitting. It does not change the patients,
folds or results of the six-structure grade model, and it does not involve the CNNs.

## Question and design

How much information on recorded SLC26A4 status remains in the vestibule, semicircular canal,
internal auditory canal and cochlear aperture grades when the EVA and cochlear grades are left
out? Input fields are removed, not patients. The analysis does not evaluate a population without
EVA or cochlear abnormalities.

The cohort is the 229 patients of the grade model (88 recorded SLC26A4, 141 other tested
patients), with its five outer folds and its CV groups, which keep the possible family pair
together. The outcome is recorded SLC26A4 (label 0 in the fold table). The model uses the
grades of both ears with the one-hot grade levels of the grade model, missing grades encoded as
grade 0, and the same fold-local imputation and scaling, L2 logistic regression, balanced class
weights, liblinear solver, seed, seven-value C grid, three inner folds and one-standard-error
selection rule. It has no demographic or image inputs.

Three models are fitted:

1. All six structures: reproduction control.
2. Five structures without EVA: reproduction control.
3. Four structures without EVA and cochlea: the model of interest (18 binary columns).

The script uses the encoding, tuning, fitting and scoring functions of `grade_models.analysis`,
but not its data-correction loader. It reads the person table directly: the birthday correction
has no effect here because age and other demographic variables are not model inputs.

## Checks before the results are interpreted

- The 229 patients, their labels and fold order match the saved predictions of the grade model;
  the grouping agrees between the fold table and the person table.
- No group spans an outer or inner split. All five outer test sets contain both classes.
- The four-structure model contains only the eight retained left/right grade fields (18 indicators).
- The six-structure probabilities are reproduced within a maximum absolute error of 1e-7, and the
  pooled AUCs of both controls within 1e-9; log loss and Brier score within 1e-8. The model
  without EVA has saved metrics but no saved patient scores.
- The selected C agrees with the saved controls in all folds.
- No convergence warnings, nonfinite scores, incomplete predictions or missing folds.
- The hashes of the protected inputs, code and grade-model results are unchanged.

A failed check stops the analysis; the tolerances and the training procedure are not changed.

## Estimation and comparisons

Main description: pooled held-out AUC of the four-structure model with a 2,000-sample patient
bootstrap percentile 95% CI, plus fold-wise AUC, log loss, Brier score, precision-recall summary
and calibration. The reproduction controls use the bootstrap seed labels of the saved results.

Two fixed paired contrasts, always **four-structure model minus reference**:

1. Versus five structures without EVA (effect of also removing the cochlear grades).
2. Versus all six structures.

Both are reported with 2,000-sample paired bootstrap 95% CIs and paired DeLong tests; the two
DeLong p values are Holm-adjusted together. The CIs are individual, not simultaneous. These are
conditional analyses of fixed held-out predictions; they do not cover model selection, training
variability or generalization to other populations. Paired differences in log loss and Brier score
are descriptive, with negative values favoring the four-structure model.

As a fixed secondary check, the CV groups are resampled (2,000 draws), so that both members of
the possible family pair are drawn together. This gives the AUC of the four-structure model and
the same two AUC differences. It does not address unknown relatives.

The observed abnormality rates of the four retained structures and the number of distinct encoded
grade profiles are described. No subgroups are selected, no further models are tuned, no
thresholds are searched, no localization is inferred, and coefficient signs are not read as
causal findings. The saved vestibule-only model is quoted as context, not as a further formal
comparison.

## Outputs and interpretation

All outputs are written to `config.FOUR_STRUCTURE_DIR`: the design (`design.json`, with input
hashes, environment, exact splits and feature columns), held-out scores (`predictions.csv`),
fitted coefficients (`coefficients.csv`), metrics, selected C and paired contrasts
(`results.json`), group-bootstrap draws (`group_bootstrap.csv`), profile counts
(`profile_summary.csv`), checks (`checks.json`, `verification.json`), a comparison plot
(`comparison.pdf/png`) and a short report (`report.md`). Patient-level files are not for
distribution; the report contains aggregate results only.

The result can support an association between the remaining anatomical pattern and SLC26A4 status
in this cohort. It cannot establish performance in patients without EVA or cochlear abnormalities,
an independent biological mechanism, generalization to an external population, or information
beyond EVA and cochlea in a combined model.

## Commands

From the repository root:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
python -m grade_models.test_four_structures
python -m grade_models.four_structures freeze
python -m grade_models.four_structures run
python -m grade_models.four_structures verify
```
