"""Grade model of four structures (without the EVA and cochlear grades), with a design fixed before fitting.

The model uses the right and left grades of the vestibule, semicircular canals, internal auditory
canal and cochlear aperture (18 one-hot indicators) with the folds, tuning, fitting and scoring of
the grade model in grade_models.analysis. The six-structure model and the model without EVA are
refit as controls and must reproduce their saved results. ``freeze`` saves the design (feature
columns, splits, settings and hashes of the protected files), ``run`` fits the models and
``verify`` checks the saved results. See four_structures_protocol.md.

Usage: python -m grade_models.four_structures {freeze,run,verify}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from statsmodels.stats.multitest import multipletests

import config
from grade_models.analysis import (
    B_BOOT, C_GRID, SEED, STRUCTS, delong_auc_test, derived_seed, dump_json, make_pipeline,
    nested_oof, onehot_block, paired_bootstrap_auc, score_metrics,
)

HERE = Path(__file__).resolve().parent  # code folder; the code files below are protected by hash
CODE_FILES = ('analysis.py', 'four_structures.py', 'four_structures_protocol.md', 'test_four_structures.py')
OUT = config.FOUR_STRUCTURE_DIR

MODELS = {
    'full': tuple(STRUCTS),
    'without_eva': tuple(s for s in STRUCTS if s != 'eva'),
    'without_eva_cochlea': tuple(s for s in STRUCTS if s not in {'eva', 'cochlea'}),
}
DISPLAY = {'full': 'All six structures', 'without_eva': 'Without EVA',
           'without_eva_cochlea': 'Without EVA and cochlea'}
# score_metrics derives the bootstrap seed from these labels; the first two are those of the saved results.
LABELS = {'full': 'ladder:4_anatomy_PRIMARY', 'without_eva': 'anatomy_without_EVA',
          'without_eva_cochlea': 'four_structure_no_eva_no_cochlea_v1'}
FOUR = 'without_eva_cochlea'  # key of the four-structure model


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(name, obj):
    dump_json(OUT / name, obj)


def protected_paths():
    files = [HERE / name for name in CODE_FILES]
    files += [config.FOLDS, config.PEOPLE, config.GRADE_PREDICTIONS, config.GRADE_METRICS,
              config.AGE_AND_WITHOUT_EVA, config.FORWARD_SELECTION]
    return files


def protected_name(p):
    """Code files by name; result files by their path under the results folder."""
    p = Path(p)
    return f'code/{p.name}' if p.parent == HERE else str(p.relative_to(config.RESULTS_DIR))


def resolve(name):
    return HERE / name[5:] if name.startswith('code/') else config.RESULTS_DIR / name


def verify_hashes():
    design = read(OUT / 'design.json')
    changed = [p for p, digest in design['protected_sha256'].items() if sha(resolve(p)) != digest]
    if changed:
        raise RuntimeError(f'Protected files changed: {changed}')
    return len(design['protected_sha256'])


def load_data():
    manifest = pd.read_csv(config.FOLDS)
    anatomy = pd.read_parquet(config.PEOPLE)
    d = manifest[['patient_id', 'fold', 'binary_label']].merge(
        anatomy, on='patient_id', how='left', validate='one_to_one').set_index('patient_id')
    y = (d.binary_label == 0).astype(int).to_numpy()
    folds = d.fold.to_numpy(dtype=int)
    groups = d.cv_group_id.astype(str).to_numpy()
    assert len(d) == 229 and d.index.is_unique and int(y.sum()) == 88
    assert d.index.tolist() == manifest.patient_id.tolist()
    assert np.array_equal(groups, manifest.cv_group_id.astype(str).to_numpy())
    assert set(folds) == set(range(5))
    assert d.groupby('cv_group_id').fold.nunique().max() == 1
    saved = pd.read_csv(config.GRADE_PREDICTIONS).set_index('patient_id')
    assert set(saved.index) == set(d.index) and saved.index.is_unique
    saved = saved.loc[d.index]
    assert np.array_equal(saved.y_SLC26A4.to_numpy(), y)
    assert np.array_equal(saved.fold.to_numpy(), folds)
    return d, y, folds, groups, saved


def check_splits(d, y, folds, groups):
    """Check the outer and inner splits and return their patient lists."""
    result = {}
    for fold in range(5):
        tr, te = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        assert not set(groups[tr]) & set(groups[te])
        assert np.unique(y[te]).size == 2
        if np.unique(groups[tr]).size < len(tr):
            splitter = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
            inner = splitter.split(np.zeros((len(tr), 1)), y[tr], groups[tr])
        else:
            splitter = StratifiedKFold(3, shuffle=True, random_state=SEED)
            inner = splitter.split(np.zeros((len(tr), 1)), y[tr])
        rows = []
        for it, iv in inner:
            a, b = tr[it], tr[iv]
            assert not set(groups[a]) & set(groups[b])
            assert not set(te) & (set(a) | set(b))
            assert set(a) | set(b) == set(tr)
            rows.append({'train': d.index[a].tolist(), 'validation': d.index[b].tolist()})
        result[str(fold)] = {'train': d.index[tr].tolist(), 'test': d.index[te].tolist(), 'inner': rows}
    return result


def freeze():
    if (OUT / 'design.json').exists():
        raise RuntimeError('Design already saved; use verify, not overwrite.')
    d, y, folds, groups, _ = load_data()
    save('design.json', {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Grade fields removed, not patients; the six-structure grade-model results are not replaced',
        'models': MODELS, 'columns': {k: onehot_block(d, s).columns.tolist() for k, s in MODELS.items()},
        'n': len(d), 'positive_n': int(y.sum()), 'cv_groups': int(np.unique(groups).size),
        'seed': SEED, 'C_grid': C_GRID, 'bootstrap_B': B_BOOT,
        'environment': {'executable': sys.executable, 'python': platform.python_version(),
                        'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__,
                        'sklearn': sklearn.__version__, 'statsmodels': statsmodels.__version__},
        'protected_sha256': {protected_name(p): sha(p) for p in protected_paths()},
        'splits': check_splits(d, y, folds, groups),
    })
    print(f'Design saved: {len(d)} patients ({int(y.sum())} recorded SLC26A4), three fixed models, five outer folds '
          'and grouped inner splits.', flush=True)


def group_draws(groups, rng, n_draws):
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for _ in range(n_draws):
        yield np.concatenate([members[j] for j in rng.integers(0, len(members), len(members))])


def run():
    if (OUT / 'results.json').exists():
        raise RuntimeError('Completed results exist; use verify, not overwrite.')
    start = time.monotonic()
    verify_hashes()
    d, y, folds, groups, saved = load_data()
    assert check_splits(d, y, folds, groups) == read(OUT / 'design.json')['splits']
    ladder = read(config.GRADE_METRICS)
    noeva = read(config.AGE_AND_WITHOUT_EVA)['anatomy_without_EVA']
    references = {'full': ladder['ladder']['4_anatomy_PRIMARY'], 'without_eva': noeva}
    ref_tuning = {'full': ladder['outer_fold_tuning']['4_anatomy_PRIMARY'],
                  'without_eva': noeva['outer_fold_tuning']}
    checks = []

    def check(ok, label):
        checks.append({'passed': bool(ok), 'check': label})
        if not ok:
            save('checks.json', checks)
            raise AssertionError(label)

    predictions, metrics, coefficients, reproduction = {}, {}, [], {}
    for key, structs in MODELS.items():
        print(f'Fitting {key}: {structs}', flush=True)
        X = onehot_block(d, structs)
        check(X.columns.tolist() == read(OUT / 'design.json')['columns'][key], f'{key}: design columns')
        if key == FOUR:
            check(X.shape[1] == 18, 'Four structures: exactly 18 one-hot indicators')
            check(not any(c.startswith(('eva_', 'cochlea_')) for c in X), 'No EVA or cochlea indicators')
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            p, tuning = nested_oof(X, y, folds, groups=groups)
        check(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), f'{key}: finite complete probabilities')
        if key in references:
            auc = float(roc_auc_score(y, p))
            check(abs(auc - references[key]['auroc']) < 1e-9, f'{key}: saved AUC reproduced')
            for fold in range(5):
                check(tuning[str(fold)]['C'] == ref_tuning[key][str(fold)]['C'], f'{key}: fold {fold} selected C reproduced')
            if key == 'full':
                error = float(np.max(np.abs(p - saved.p_4_anatomy_PRIMARY.to_numpy())))
                check(error <= 1e-7, 'Full model: stored patient probabilities reproduced')
            else:
                error = None
            reproduction[key] = {'auc_difference': auc - references[key]['auroc'],
                                 'max_absolute_probability_difference': error}
        predictions[key] = p
        metrics[key] = score_metrics(y, p, LABELS[key])
        metrics[key]['structures'] = structs
        metrics[key]['n_encoded_features'] = X.shape[1]
        metrics[key]['outer_fold_tuning'] = tuning
        metrics[key]['fold_auc'] = {str(f): float(roc_auc_score(y[folds == f], p[folds == f])) for f in range(5)}
        if key in references:
            for field in ('log_loss', 'brier'):
                diff = metrics[key][field] - references[key][field]
                reproduction[key][field + '_difference'] = diff
                check(abs(diff) <= 1e-8, f'{key}: saved {field} reproduced')
            reproduction[key]['ci_difference'] = (np.array(metrics[key]['auroc_percentile_ci']) - references[key]['auroc_percentile_ci']).tolist()
        for f in range(5):
            train, test = folds != f, folds == f
            model = make_pipeline(tuning[str(f)]['C'])
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                model.fit(X.to_numpy()[train], y[train])
            check(np.max(np.abs(np.round(model.predict_proba(X.to_numpy()[test])[:, 1], 12) - p[test])) < 1e-10,
                  f'{key}: fold {f} fitted coefficient export matches held-out predictions')
            for column, coef, scale in zip(X.columns, model.named_steps['model'].coef_[0], model.named_steps['scale'].scale_):
                coefficients.append({'model': key, 'fold': f, 'feature': column,
                                     'standardized_coefficient': float(coef), 'input_scale': float(scale)})
        print(f"  AUC {metrics[key]['auroc']:.6f}; CI {metrics[key]['auroc_percentile_ci']}", flush=True)

    oof = pd.DataFrame({'patient_id': d.index, 'cv_group_id': groups, 'fold': folds, 'y_SLC26A4': y,
                        **{'p_' + k: v for k, v in predictions.items()}})
    oof.to_csv(OUT / 'predictions.csv', index=False)
    pd.DataFrame(coefficients).to_csv(OUT / 'coefficients.csv', index=False)
    comparisons = {}
    for ref in ('without_eva', 'full'):
        label = FOUR + '_minus_' + ref
        comparisons[ref] = {
            **paired_bootstrap_auc(y, predictions[FOUR], predictions[ref], label),
            'delong': delong_auc_test(y, predictions[FOUR], predictions[ref]),
            'delta_log_loss': metrics[FOUR]['log_loss'] - metrics[ref]['log_loss'],
            'delta_brier': metrics[FOUR]['brier'] - metrics[ref]['brier'],
        }
    holm = multipletests([c['delong']['p_two_sided'] for c in comparisons.values()], method='holm')[1]
    for c, p in zip(comparisons.values(), holm):
        c['delong_p_holm_two_comparisons'] = float(p)

    print('Computing CV-group bootstrap sensitivity...', flush=True)
    rows = []
    # derived_seed hashes this label, so the label fixes the bootstrap draws and must not be renamed.
    rng = np.random.default_rng(derived_seed('four_structure_cv_group_bootstrap_v1'))
    for idx in group_draws(groups, rng, B_BOOT):
        if np.unique(y[idx]).size != 2:
            continue
        auc = {k: roc_auc_score(y[idx], p[idx]) for k, p in predictions.items()}
        rows.append([auc[FOUR], auc[FOUR] - auc['without_eva'], auc[FOUR] - auc['full']])
    draws = pd.DataFrame(rows, columns=['four_structure_auc', 'delta_without_eva', 'delta_full'])
    draws.to_csv(OUT / 'group_bootstrap.csv', index=False)
    cluster = {k: np.percentile(draws[k], [2.5, 97.5]).tolist() for k in draws}

    raw_cols = [f'{s}_{ear}' for s in MODELS[FOUR] for ear in ('R', 'L')]
    rates = {}
    for s in MODELS[FOUR]:
        ab = d[[f'{s}_R', f'{s}_L']].max(axis=1).fillna(0).gt(0).to_numpy()
        rates[s] = {'slc_events': int(ab[y == 1].sum()), 'slc_n': int((y == 1).sum()),
                    'other_events': int(ab[y == 0].sum()), 'other_n': int((y == 0).sum())}
    X_four = onehot_block(d, MODELS[FOUR])
    profile = X_four.astype(str).agg(''.join, axis=1)
    table = pd.DataFrame({'profile': profile.to_numpy(), 'y': y, 'p': predictions[FOUR]}).groupby('profile').agg(
        n=('y', 'size'), slc_n=('y', 'sum'), mean_oof_probability=('p', 'mean'))
    table['other_n'] = table.n - table.slc_n
    table.to_csv(OUT / 'profile_summary.csv')
    result = {
        'n': len(d), 'slc_n': int(y.sum()), 'cv_groups': int(np.unique(groups).size),
        'models': metrics, 'reproduction': reproduction, 'comparisons_four_minus_reference': comparisons,
        'cv_group_bootstrap_sensitivity_ci95': cluster, 'valid_cluster_draws': len(draws),
        'retained_structure_abnormality_rates': rates,
        'distinct_encoded_profiles': len(table),
        'patients_missing_retained_grade': int(d[raw_cols].isna().any(axis=1).sum()),
        'vestibule_only_context': read(config.FORWARD_SELECTION)['single_structure_fixed_candidate']['vestibule'],
        'scope': 'Post-hoc fixed design; intervals conditional on fixed OOF scores. No external validation.',
    }
    check(verify_hashes() > 0, 'Protected inputs, code and grade-model results unchanged')
    result['elapsed_seconds'] = time.monotonic() - start
    save('results.json', result)
    save('checks.json', checks)
    make_report(result, predictions, y)
    verify()


def make_report(result, predictions, y):
    def interval(values):
        return f'[{values[0]:.3f}, {values[1]:.3f}]'
    lines = ['# Grade model of four structures (without EVA and cochlea)', '',
             'Post-hoc analysis with a fixed design.', '',
             '## Results', '', '| Inputs | AUC (95% patient-bootstrap CI) | Log loss | Brier score |',
             '|---|---:|---:|---:|']
    for key, row in result['models'].items():
        lines.append(f"| {DISPLAY[key]} | {row['auroc']:.3f} {interval(row['auroc_percentile_ci'])} | {row['log_loss']:.3f} | {row['brier']:.3f} |")
    lines += ['', 'The four-structure model uses left/right vestibule, semicircular canal, internal auditory canal,',
              'and cochlear aperture grades: 18 indicators, no age/sex or image inputs.', '',
              '## Fixed paired contrasts', '', 'Differences are four structures minus the reference; negative AUC differences favor the reference.', '',
              '| Reference | AUC difference (95% paired bootstrap CI) | DeLong p | Holm p (two tests) |',
              '|---|---:|---:|---:|']
    for ref, row in result['comparisons_four_minus_reference'].items():
        lines.append(f"| {DISPLAY[ref]} | {row['delta_auroc']:+.3f} {interval(row['percentile_ci'])} | {row['delong']['p_two_sided']:.6g} | {row['delong_p_holm_two_comparisons']:.6g} |")
    lines += ['', 'The CIs are individual intervals, not multiplicity-adjusted intervals.', '',
              '## Fold results', '', '| Outer fold | All six | Without EVA | Without EVA and cochlea | Selected C (four structures) |',
              '|---|---:|---:|---:|---:|']
    for f in map(str, range(5)):
        vals = [result['models'][k]['fold_auc'][f] for k in MODELS]
        lines.append(f"| {f} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {result['models'][FOUR]['outer_fold_tuning'][f]['C']} |")
    lines += ['', '## Group-bootstrap sensitivity', '',
              f"Resampled {result['cv_groups']} CV groups (the possible family pair is one group); {result['valid_cluster_draws']} valid draws."]
    for label, ci in result['cv_group_bootstrap_sensitivity_ci95'].items():
        lines.append(f'- {label}: 95% CI {interval(ci)}.')
    lines += ['', '## Retained anatomy: observed abnormality counts', '',
              '| Structure | Recorded SLC26A4 | Other tested patients |', '|---|---:|---:|']
    for s, r in result['retained_structure_abnormality_rates'].items():
        lines.append(f"| {s} | {r['slc_events']}/{r['slc_n']} | {r['other_events']}/{r['other_n']} |")
    lines += ['', f"There were {result['distinct_encoded_profiles']} distinct encoded four-structure grade profiles. "
              f"{result['patients_missing_retained_grade']} patients had a missing retained grade (handled as grade 0, as in the six-structure model).",
              '', '## Reproduction and limits', '',
              f"- Same {result['n']} patients, {result['slc_n']} recorded SLC26A4, the five outer folds and the group-aware inner selection of the grade model.",
              '- Full and EVA-excluded controls reproduced pooled AUC, log loss, Brier score and selected penalties within the fixed tolerances.',
              f"- Full-model maximum absolute probability difference: {result['reproduction']['full']['max_absolute_probability_difference']:.3g}.",
              '- No convergence warning was allowed. Fitted coefficient exports independently reproduced each fold\'s scores.',
              '- Protected inputs, code and grade-model results were unchanged (hash check).',
              '- Bootstrap intervals and DeLong tests condition on fixed OOF predictions; training/model-development uncertainty is not included.',
              '- Group bootstrap accounts only for known CV groups, not unknown relatives.', '',
              f"Elapsed computation: {result['elapsed_seconds']:.1f} seconds; CPU only. No CNN training.", '',
              'See `results.json`, `predictions.csv`, `coefficients.csv`, `checks.json`, `design.json` and `comparison.pdf`.',
              'Patient-level files are not for distribution.', '']
    (OUT / 'report.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for i, (k, p) in enumerate(predictions.items()):
        row = result['models'][k]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{DISPLAY[k]} ({row['auroc']:.3f})")
        lo, hi = row['auroc_percentile_ci']
        axes[1].errorbar(row['auroc'], 2-i, xerr=[[row['auroc']-lo], [hi-row['auroc']]], fmt='o', capsize=4)
    axes[0].plot([0, 1], [0, 1], ':', color='grey')
    axes[0].set(xlabel='False-positive rate', ylabel='True-positive rate', title='Held-out patient predictions')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[1].set(yticks=[2, 1, 0], yticklabels=[DISPLAY[k] for k in MODELS], xlabel='AUC (95% conditional CI)', xlim=(0.5, 1.01), ylim=(-0.5, 2.5))
    fig.suptitle(f"Same {result['n']} patients; EVA and cochlear grades removed, not patients")
    fig.tight_layout()
    fig.savefig(OUT / 'comparison.pdf', bbox_inches='tight')
    fig.savefig(OUT / 'comparison.png', dpi=160, bbox_inches='tight')
    plt.close(fig)


def verify():
    count = verify_hashes()
    if not (OUT / 'results.json').exists():
        print(f'{count} protected hashes verified; no computed result yet.')
        return
    result = read(OUT / 'results.json')
    oof = pd.read_csv(OUT / 'predictions.csv').set_index('patient_id')
    d, y, folds, groups, _ = load_data()
    assert oof.index.tolist() == d.index.tolist()
    assert np.array_equal(oof.y_SLC26A4, y) and np.array_equal(oof.fold, folds)
    assert np.array_equal(oof.cv_group_id.astype(str), groups)
    for k in MODELS:
        assert abs(roc_auc_score(y, oof['p_' + k]) - result['models'][k]['auroc']) < 1e-12
        assert len(result['models'][k]['fold_auc']) == 5
    checks = read(OUT / 'checks.json')
    assert checks and all(c['passed'] for c in checks)
    save('verification.json', {'passed': True, 'checks': len(checks), 'protected_hashes': count,
                               'oof_metrics_recomputed': True})
    print(f"PASS: {len(checks)} run checks; {count} protected hashes; saved held-out AUCs independently reproduced.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['freeze', 'run', 'verify'])
    args = parser.parse_args()
    config.ensure_dirs()
    {'freeze': freeze, 'run': run, 'verify': verify}[args.command]()


if __name__ == '__main__':
    main()
