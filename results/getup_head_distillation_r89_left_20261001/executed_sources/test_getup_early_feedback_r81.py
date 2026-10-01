import unittest
from unittest.mock import patch
import numpy as np
from diagnostics.train_getup_early_feedback_r81 import (
    checked_matrix, early_envelope, combined_residual, feedback_rollout, CAP, TRAIN, HELDOUT)
from diagnostics.search_getup_reference_feedback_r64 import residual


class EarlyFeedbackTests(unittest.TestCase):
    def test_zero_reference_error_has_no_extra_feedback(self):
        np.testing.assert_array_equal(combined_residual(np.zeros(4), np.ones(8),
                                      np.ones((8, 2)), 2.), np.zeros(10))

    def test_outside_window_exactly_preserves_old_feedback(self):
        error = np.array([.12, -.25, .04, -.05]); gains = np.ones(8)
        for t in (0., .5, 3.5, 6., 30.):
            np.testing.assert_array_equal(combined_residual(error, gains, np.ones((8, 2)), t),
                                          residual(error, gains))
        self.assertEqual(early_envelope(1.5), 1.)
        self.assertEqual(early_envelope(2.5), 1.)

    def test_combined_cap_not_two_separate_budgets(self):
        out = combined_residual(np.ones(4), np.full(8, 2.), np.full((8, 2), 2.), 2.)
        self.assertLessEqual(np.abs(out).max(), CAP)
        np.testing.assert_array_equal(out[8:], np.zeros(2))

    def test_independent_leg_mapping(self):
        matrix = np.zeros((8, 2)); matrix[0, 0] = .3
        out = combined_residual(np.array([.2, 0., 0., 0.]), np.zeros(8), matrix, 2.)
        self.assertAlmostEqual(out[0], .06)
        np.testing.assert_array_equal(out[1:], np.zeros(9))

    def test_invalid_gains_rejected(self):
        for v in (np.zeros(15), np.full(16, np.nan), np.full(16, 2.001)):
            with self.assertRaises(ValueError): checked_matrix(v)

    def test_zero_gain_delegates_to_original_evaluator(self):
        with patch('diagnostics.train_getup_early_feedback_r81.rollout', return_value=('row', 'trace')) as original:
            result = feedback_rollout('sim', 'state', {}, True, 'targets', 'phases', 'ids',
                                      'ref', 'gains', np.zeros(16), True)
            self.assertEqual(result, ('row', 'trace'))
            original.assert_called_once_with('sim', 'state', {}, True, 'targets', 'phases', 'ids', 'ref', 'gains', True)

    def test_fresh_test_seeds_not_training_or_previous_test(self):
        self.assertFalse(set(TRAIN) & set(HELDOUT))
        self.assertFalse(set(range(780000, 780040)) & set(HELDOUT))


if __name__ == '__main__': unittest.main()
