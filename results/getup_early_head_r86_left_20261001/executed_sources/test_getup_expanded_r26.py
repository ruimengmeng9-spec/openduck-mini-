import unittest
import numpy as np
from diagnostics.train_getup_expanded_r26 import extend_library, compare_paired


class ExpandedAtlasTest(unittest.TestCase):
    def previous(self):
        return dict(seed=np.array([1, 2]), obs=np.zeros((2, 50)),
                    knots=np.zeros((2, 2, 8)), rescue=np.array([False, True]))

    def scan(self, seed=3):
        return dict(seed=seed, success=False, initial_hash='same', initial_obs=[0.] * 50)

    def trial(self, long4=True):
        return dict(seed=3, baseline=dict(initial_hash='same'),
                    selected=dict(initial_hash='same', success=True),
                    selected_long=dict(initial_hash='same', success=True, success_4s=long4),
                    knots=np.full((2, 8), .1).tolist())

    def test_long_success_without_4s_is_not_rescue(self):
        library, anchors = extend_library(self.previous(), [self.scan()], [self.trial(False)])
        self.assertFalse(library['rescue'][-1])
        self.assertEqual(anchors[-1]['kind'], 'unsupported_abstention')
        self.assertTrue((library['knots'][-1] == 0).all())

    def test_only_verified_paired_rescue_is_added(self):
        library, anchors = extend_library(self.previous(), [self.scan()], [self.trial()])
        self.assertTrue(library['rescue'][-1])
        self.assertEqual(anchors[-1]['kind'], 'verified_rescue')

    def test_mismatched_initialization_rejected(self):
        trial = self.trial(); trial['selected_long']['initial_hash'] = 'different'
        with self.assertRaises(ValueError):
            extend_library(self.previous(), [self.scan()], [trial])

    def test_duplicate_training_seed_rejected(self):
        with self.assertRaises(ValueError):
            extend_library(self.previous(), [self.scan(1)], [])

    def test_unsupported_start_is_not_success_label(self):
        library, anchors = extend_library(self.previous(), [self.scan()], [])
        self.assertFalse(library['rescue'][-1])
        self.assertEqual(anchors[-1]['kind'], 'unsupported_abstention')

    def test_paired_comparison_checks_initial_hash(self):
        new = dict(results=[dict(seed=1, radius=.6, initial_hash='A', success=True)])
        old = dict(results=[dict(seed=1, radius=.3, initial_hash='B', success=False)])
        with self.assertRaises(ValueError):
            compare_paired(new, old, .6, .3)


if __name__ == '__main__':
    unittest.main()
