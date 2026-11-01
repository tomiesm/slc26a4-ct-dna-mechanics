"""Review of the four-structure model's errors from its fixed held-out predictions.

Describes the missed and detected SLC26A4 patients and the false positives at fixed score cutoffs,
the grade profiles, and how each patient contributes to the pooled AUC. No model is fitted.

Usage: python -m grade_models.four_structure_errors
"""
import hashlib
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from statsmodels.stats.multitest import multipletests

import config

MODEL_DIR = config.FOUR_STRUCTURE_DIR
OUT = config.FOUR_STRUCTURE_ERRORS_DIR
STRUCTS = ['cochlea', 'vestibule', 'scc', 'iac', 'eva', 'aperture']
KEEP = ['vestibule', 'scc', 'iac', 'aperture']
MODELS = ['full', 'without_eva', 'without_eva_cochlea']
NAMES = {'full': 'All six structures', 'without_eva': 'Without EVA',
         'without_eva_cochlea': 'Without EVA and cochlea'}
COLS = [f'{s}_{ear}' for s in KEEP for ear in ['R', 'L']]
INPUTS = [MODEL_DIR / p for p in ['predictions.csv', 'results.json', 'design.json']]
INPUTS += [config.FOLDS, config.PEOPLE, config.DATA_CORRECTIONS, config.VARIANT_TABLE, config.RECORDS]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def input_name(path):
    """Path of an input file under the results folder or the data folder."""
    root = config.RESULTS_DIR if path.is_relative_to(config.RESULTS_DIR) else config.DATA_DIR
    return str(path.relative_to(root))


def count_outcomes(y, p, cutoff):
    positive = p >= cutoff
    return {name: int(mask.sum()) for name, mask in {
        'TP': (y == 1) & positive, 'FN': (y == 1) & ~positive,
        'FP': (y == 0) & positive, 'TN': (y == 0) & ~positive}.items()}


def pair_credits(y, p):
    """Contribution to pooled AUC, including exact tie credits, for each class."""
    pos, neg = p[y == 1], p[y == 0]
    credit = (pos[:, None] > neg).astype(float) + 0.5 * (pos[:, None] == neg)
    out = np.empty(len(y))
    out[y == 1], out[y == 0] = credit.mean(axis=1), credit.mean(axis=0)
    return out


def describe(g):
    ages = g.age_at_first_ct.dropna()
    return {'n': len(g), 'age_n': len(ages),
            'age_median': float(ages.median()) if len(ages) else None,
            'age_q1': float(ages.quantile(.25)) if len(ages) else None,
            'age_q3': float(ages.quantile(.75)) if len(ages) else None,
            'age_missing': int(g.age_at_first_ct.isna().sum()),
            'sex_counts': {str(k): int(v) for k, v in g.gender.value_counts(dropna=False).items()},
            'retained_complete_normal_n': int(g.retained_complete_normal.sum()),
            'retained_encoded_normal_n': int(g.retained_encoded_normal.sum()),
            'retained_missing_n': int(g.retained_missing.sum()),
            'profile_not_in_training_n': int(g.profile_unseen_in_training.sum()),
            'mixed_class_profile_n': int(g.mixed_class_profile.sum()),
            'abnormal_counts': {s: int(g[s + '_abnormal'].sum()) for s in STRUCTS},
            'grade_missing_counts': {s: int(g[s + '_missing'].sum()) for s in STRUCTS},
            'gene_counts': {str(k): int(v) for k, v in g.gene_category.value_counts().items()},
            'mean_full_minus_reduced_score': float((g.p_full - g.p_without_eva_cochlea).mean()),
            'mean_auc_credit_full': float(g.auc_credit_full.mean()),
            'mean_auc_credit_reduced': float(g.auc_credit_without_eva_cochlea.mean())}


def main():
    config.ensure_dirs()
    assert not (OUT / 'results.json').exists(), 'Do not overwrite completed results'
    hashes = {input_name(p): sha(p) for p in INPUTS}
    checks = []
    def check(condition, msg):
        assert condition, msg
        checks.append(msg)
    man = pd.read_csv(config.FOLDS).set_index('patient_id')
    oof = pd.read_csv(MODEL_DIR / 'predictions.csv').set_index('patient_id')
    anatomy = pd.read_parquet(config.PEOPLE).set_index('patient_id')
    check(oof.index.is_unique and set(oof.index) == set(man.index) and len(oof) == 229,
          'Exact unique 229-patient membership')
    man = man.loc[oof.index]
    check(np.array_equal(oof.y_SLC26A4, (man.binary_label == 0).astype(int)), 'Fold-table labels')
    check(np.array_equal(oof.fold, man.fold), 'Outer folds')
    check(np.array_equal(oof.cv_group_id, man.cv_group_id), 'CV groups')
    d = oof.join(anatomy.drop(columns=['cv_group_id']), validate='one_to_one')
    check(np.array_equal(d.y_SLC26A4 == 1, d.gene_category == 'SLC26A4'), 'Clinical labels')
    check(d.groupby('cv_group_id').fold.nunique().max() == 1, 'No CV group crosses folds')
    master = pd.read_parquet(config.RECORDS)
    corrections = pd.read_csv(config.DATA_CORRECTIONS)
    for row in corrections.itertuples():
        check(row.field == 'birthday', 'Only documented birthday correction applied')
        if row.patient_id not in d.index:
            continue
        check(pd.Timestamp(d.loc[row.patient_id, 'birthday']).date().isoformat() == row.original_value,
              'Birthday correction source precondition')
        first = pd.to_datetime(master.loc[master.patient_id == row.patient_id, 'ct_date']).min()
        d.loc[row.patient_id, 'age_at_first_ct'] = (first - pd.Timestamp(row.corrected_value)).days / 365.25
    check((d.age_at_first_ct.dropna() >= 0).all(), 'No negative corrected ages')
    saved = json.loads((MODEL_DIR / 'results.json').read_text())
    y = d.y_SLC26A4.to_numpy()
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    thresholds = []
    for model in MODELS:
        p = d['p_' + model].to_numpy()
        check(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), model + ' valid scores')
        for name, fun in [('auroc', roc_auc_score), ('log_loss', log_loss), ('brier', brier_score_loss)]:
            check(abs(fun(y, p) - saved['models'][model][name]) < 1e-12, model + ' reproduces ' + name)
        d['auc_credit_' + model] = pair_credits(y, p)
        check(abs(d.loc[y == 1, 'auc_credit_' + model].mean() - roc_auc_score(y, p)) < 1e-12,
              model + ' positive AUC credits reproduce AUC')
        check(abs(d.loc[y == 0, 'auc_credit_' + model].mean() - roc_auc_score(y, p)) < 1e-12,
              model + ' negative AUC credits reproduce AUC')
        for cutoff in [.4, .5, .6]:
            row = {'model': model, 'threshold': cutoff, **count_outcomes(y, p, cutoff)}
            row.update(sensitivity=row['TP'] / n_pos, specificity=row['TN'] / n_neg)
            thresholds.append(row)
        d['outcome_' + model] = np.where(y == 1, np.where(p >= .5, 'TP', 'FN'), np.where(p >= .5, 'FP', 'TN'))
    d['outcome'] = d.outcome_without_eva_cochlea
    for s in STRUCTS:
        raw = d[[s + '_R', s + '_L']]
        d[s + '_missing'] = raw.isna().any(axis=1)
        d[s + '_abnormal'] = raw.max(axis=1) > 0
    d['retained_missing'] = d[COLS].isna().any(axis=1)
    d['retained_encoded_normal'] = d[COLS].fillna(0).eq(0).all(axis=1)
    d['retained_complete_normal'] = d.retained_encoded_normal & ~d.retained_missing
    encoded = d[COLS].fillna(0).round().astype(int)
    # These integer profiles are one-to-one with the one-hot encoding of the model.
    levels = {'vestibule': 3, 'scc': 3, 'iac': 2, 'aperture': 1}
    check(all(encoded[c].between(0, levels[c.split('_')[0]]).all() for c in COLS),
          'All grades in the fixed categorical encoding')
    d['profile'] = encoded.astype(str).agg('|'.join, axis=1)
    profiles = d.groupby('profile').agg(n=('y_SLC26A4', 'size'), slc_n=('y_SLC26A4', 'sum'),
        missing_n=('retained_missing', 'sum'), n_folds=('fold', 'nunique'))
    profiles['other_n'] = profiles.n - profiles.slc_n
    check(len(profiles) == saved['distinct_encoded_profiles'], 'Distinct profiles match the four-structure results')
    d['mixed_class_profile'] = d.profile.map((profiles.slc_n > 0) & (profiles.other_n > 0))
    d['profile_training_n'] = 0
    d['profile_training_slc_n'] = 0
    for pid, row in d.iterrows():
        peers = d[(d.fold != row.fold) & (d.profile == row.profile)]
        d.loc[pid, 'profile_training_n'] = len(peers)
        d.loc[pid, 'profile_training_slc_n'] = peers.y_SLC26A4.sum()
    d['profile_unseen_in_training'] = d.profile_training_n == 0
    check(d.groupby(['fold', 'profile']).p_without_eva_cochlea.nunique().max() == 1,
          'Identical features within a fold receive identical scores')
    check(int(d.retained_missing.sum()) == saved['patients_missing_retained_grade'], 'Retained missingness matches')
    groups = {k: describe(d[d.outcome == k]) for k in ['FN', 'TP', 'FP', 'TN']}
    changes = {str(label): pd.crosstab(d.loc[y == label, 'outcome_full'],
                                     d.loc[y == label, 'outcome']).to_dict()
               for label in [0, 1]}
    cases = d[y == 1]
    fn, tp = cases[cases.outcome == 'FN'], cases[cases.outcome == 'TP']
    age_test = mannwhitneyu(fn.age_at_first_ct.dropna(), tp.age_at_first_ct.dropna(),
                            alternative='two-sided', method='asymptotic')
    check(set(cases.gender.dropna()) <= {'M', 'F'}, 'Recognized recorded sex coding')
    sex_table = [[int(g.gender.eq('M').sum()), int(g.gender.eq('F').sum())] for g in [fn, tp]]
    sex_test = fisher_exact(sex_table, alternative='two-sided')
    holm = multipletests([age_test.pvalue, sex_test.pvalue], method='holm')[1]
    tests = {'age_FN_vs_TP': {'p': float(age_test.pvalue), 'holm_p_two_tests': float(holm[0]),
                              'method': 'Two-sided asymptotic Mann–Whitney; tie correction and default continuity correction'},
             'sex_FN_vs_TP': {'p': float(sex_test.pvalue), 'holm_p_two_tests': float(holm[1]),
                              'table_rows_FN_TP_columns_M_F': sex_table,
                              'missing_n': int(cases.gender.isna().sum())}}
    variant = pd.read_csv(config.VARIANT_TABLE).set_index('patient_id')
    check(variant.index.is_unique and len(variant) == 35, 'Unique patients in the variant subset')
    joined = d[['y_SLC26A4', 'outcome']].join(variant[['allelic_configuration', 'c919_dosage']], how='inner')
    check(len(joined) == 34 and joined.y_SLC26A4.eq(1).all(), 'Variant overlap is 34 SLC26A4 cases')
    variant_table = pd.crosstab(joined.allelic_configuration, joined.outcome).to_dict()
    normal = d[d.retained_encoded_normal]
    normal_summary = {'n': len(normal), 'slc_n': int(normal.y_SLC26A4.sum()),
                      'other_n': int((normal.y_SLC26A4 == 0).sum()),
                      'missing_n': int(normal.retained_missing.sum()),
                      'score_min': float(normal.p_without_eva_cochlea.min()),
                      'score_max': float(normal.p_without_eva_cochlea.max()),
                      'training_slc_prevalence_min': float((normal.profile_training_slc_n / normal.profile_training_n).min()),
                      'training_slc_prevalence_max': float((normal.profile_training_slc_n / normal.profile_training_n).max())}
    output = {'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Post-hoc fixed-score descriptive error review; no model fitting',
              'n': len(d), 'threshold': .5, 'threshold_counts': thresholds,
              'groups': groups, 'full_to_reduced': changes, 'exploratory_tests': tests,
              'normal_retained_profile': normal_summary, 'variant_overlap': variant_table,
              'input_sha256': hashes, 'checks': checks}
    check(all(sha(p) == hashes[input_name(p)] for p in INPUTS), 'All input hashes unchanged')
    (OUT / 'results.json').write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    profiles.to_csv(OUT / 'profiles.csv')
    pd.DataFrame(thresholds).to_csv(OUT / 'threshold_counts.csv', index=False)
    fields = list(oof.columns) + ['outcome', 'outcome_full', 'outcome_without_eva', 'gene_category', 'gender',
        'age_at_first_ct', *[f'{s}_{ear}' for s in STRUCTS for ear in ['R', 'L']],
        'profile', 'retained_missing', 'retained_complete_normal', 'mixed_class_profile',
        'profile_training_n', 'profile_training_slc_n', 'profile_unseen_in_training',
        *['auc_credit_' + m for m in MODELS]]
    d[fields].to_csv(OUT / 'patient_review.csv')
    lines = ['# Errors of the four-structure model (without EVA and cochlear grades)', '',
             'Review of the saved held-out predictions. A patient is counted as missed when the recorded status is '
             'SLC26A4-positive and the held-out score is below 0.5; this cut-off is used for the review only.', '',
             '## Classification at 0.5', '',
             f'| Model | Detected / {n_pos} | Missed / {n_pos} | False positives / {n_neg} | Correct negatives / {n_neg} |',
             '|---|---:|---:|---:|---:|']
    for row in thresholds:
        if row['threshold'] == .5:
            lines.append(f"| {NAMES[row['model']]} | {row['TP']} | {row['FN']} | {row['FP']} | {row['TN']} |")
    lines += ['', '## Missed and detected SLC26A4-positive patients (four-structure model)', '',
              '| Recorded finding | Missed | Detected |', '|---|---:|---:|']
    for name, key in [('Number of patients', 'n'), ('All four retained structures normal, complete grades', 'retained_complete_normal_n'),
                      ('Any missing retained grade', 'retained_missing_n'), ('Profile absent from outer training fold', 'profile_not_in_training_n')]:
        lines.append(f"| {name} | {groups['FN'][key]} | {groups['TP'][key]} |")
    for s in STRUCTS:
        lines.append(f"| {s}: at least one abnormal ear | {groups['FN']['abnormal_counts'][s]} | {groups['TP']['abnormal_counts'][s]} |")
    lines += ['', f"The all-normal encoded profile contains {normal_summary['slc_n']} SLC26A4-positive and {normal_summary['other_n']} other tested patients; {normal_summary['missing_n']} have a missing retained grade. Its held-out scores range from {normal_summary['score_min']:.3f} to {normal_summary['score_max']:.3f} across folds. The proportion of SLC26A4-positive patients among training patients with the same profile ranges from {normal_summary['training_slc_prevalence_min']:.1%} to {normal_summary['training_slc_prevalence_max']:.1%}.", '',
              '## Age and sex', '']
    for k in ['FN', 'TP']:
        g = groups[k]
        lines.append(f"- {k}: age median {g['age_median']:.2f} y (IQR {g['age_q1']:.2f}–{g['age_q3']:.2f}; n={g['age_n']}); recorded sex {g['sex_counts']}.")
    for k, v in tests.items():
        lines.append(f"- {k}: unadjusted p={v['p']:.6g}; Holm p={v['holm_p_two_tests']:.6g} (family of two).")
    lines += ['', '## False positives', '', f"Other tested patients with a positive four-structure prediction: {groups['FP']['n']}. Recorded gene categories: {groups['FP']['gene_counts']}. Abnormalities (a patient can have several): {groups['FP']['abnormal_counts']}. {groups['FP']['profile_not_in_training_n']} have a retained profile absent from their training fold.", '',
              '## Contributions to the AUC', '',
              'Each SLC26A4-positive patient contributes the proportion of other tested patients it outranks, with half credit for ties; the mean of these contributions is the AUC.', '',
              f"Mean contribution of missed patients: {groups['FN']['mean_auc_credit_reduced']:.3f} with four structures and {groups['FN']['mean_auc_credit_full']:.3f} with all six. Detected patients: {groups['TP']['mean_auc_credit_reduced']:.3f} and {groups['TP']['mean_auc_credit_full']:.3f}.", '',
              '## Other cut-offs', '', f'| Cut-off | Detected / {n_pos} | Missed / {n_pos} | False positives / {n_neg} |', '|---|---:|---:|---:|']
    for row in thresholds:
        if row['model'] == MODELS[-1]:
            lines.append(f"| {row['threshold']:.1f} | {row['TP']} | {row['FN']} | {row['FP']} |")
    lines += ['', '## Variant subset', '', f"{len(joined)} of the {n_pos} SLC26A4-positive patients are in the variant subset. Counts by model outcome and genotype group: `{variant_table}`.", '',
              '## Checks and files', '',
              f"{len(checks)} checks passed (cohort, folds and labels; saved AUCs, log losses and Brier scores; AUC contributions; birthday correction; input hashes).", '',
              'Values: `results.json`; cut-off counts: `threshold_counts.csv`; profile counts: `profiles.csv`; patient-level review: `patient_review.csv`; figure: `error_review.pdf` / `.png`.', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    colors = {'FN': '#c85b45', 'TP': '#267b72', 'FP': '#c85b45', 'TN': '#267b72'}
    for ax, label in zip(axes, [1, 0]):
        sub = d[y == label]
        # Group exactly coincident points; circle area denotes patient count.
        pts = sub.groupby(['p_without_eva_cochlea', 'p_full', 'outcome']).size().reset_index(name='n')
        for kind, g in pts.groupby('outcome'):
            ax.scatter(g.p_without_eva_cochlea, g.p_full, s=22*g.n, c=colors[kind], alpha=.65,
                       label=kind + f' (n={int(g.n.sum())})', edgecolors='white', linewidth=.5)
        ax.plot([0, 1], [0, 1], ':', c='gray', lw=1)
        ax.axhline(.5, color='gray', lw=.7)
        ax.axvline(.5, color='gray', lw=.7)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Four-structure held-out score', ylabel='Full-model held-out score',
               title=f'Recorded SLC26A4 (n={n_pos})' if label else f'Other tested patients (n={n_neg})')
        ax.legend(fontsize=8, loc='lower right')
    fig.suptitle('Circle area = number of patients; lines at the 0.5 review cut-off', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'error_review.pdf')
    fig.savefig(OUT / 'error_review.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'groups': groups, 'tests': tests, 'checks': len(checks)}, indent=2))


if __name__ == '__main__':
    main()
