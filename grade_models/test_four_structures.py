"""Tests of the encoding, grouping and splits of the four-structure grade model.

The last test reads the fold table, the person table and the saved predictions of the grade model
(run python -m grade_models.analysis first).

Usage: python -m grade_models.test_four_structures
"""
import unittest

import numpy as np
import pandas as pd

from grade_models import four_structures as E
from grade_models.analysis import STRUCTS, onehot_block


class EncodingAndGrouping(unittest.TestCase):
    def test_only_four_structures(self):
        self.assertEqual(E.MODELS[E.FOUR], ('vestibule', 'scc', 'iac', 'aperture'))

    def test_dimensions_and_missing_convention(self):
        d = pd.DataFrame({f'{s}_{ear}': [0., 1., np.nan] for s in STRUCTS for ear in ('R', 'L')})
        x = onehot_block(d, E.MODELS[E.FOUR])
        self.assertEqual(x.shape, (3, 18))
        self.assertEqual(int(x.iloc[0].sum()), 0)
        self.assertEqual(int(x.iloc[2].sum()), 0)
        self.assertEqual(int(x.iloc[1].sum()), 8)
        self.assertFalse(any(c.startswith(('eva_', 'cochlea_')) for c in x))

    def test_excluded_fields_cannot_change_encoding(self):
        d = pd.DataFrame({f'{s}_{ear}': [0, 1] for s in STRUCTS for ear in ('R', 'L')})
        before = onehot_block(d, E.MODELS[E.FOUR])
        d[['eva_R', 'eva_L', 'cochlea_R', 'cochlea_L']] = 6
        pd.testing.assert_frame_equal(before, onehot_block(d, E.MODELS[E.FOUR]))

    def test_group_bootstrap_keeps_pair(self):
        groups = np.array(['a', 'a', 'b', 'c'])
        draws = list(E.group_draws(groups, np.random.default_rng(4), 50))
        for idx in draws:
            self.assertEqual(int((idx == 0).sum()), int((idx == 1).sum()))

    def test_real_membership_and_splits(self):
        d, y, folds, groups, _ = E.load_data()
        splits = E.check_splits(d, y, folds, groups)
        self.assertEqual(len(splits), 5)
        self.assertTrue(all(len(r['inner']) == 3 for r in splits.values()))


if __name__ == '__main__':
    unittest.main()
