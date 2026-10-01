"""Pure control-contract tests; these do not establish recovery success."""
import unittest
import numpy as np

from diagnostics.search_getup_reference_feedback_r64 import (
    features, residual, record_reference, targets_for)
from diagnostics.search_getup_sustained_bridge_r42 import DURATIONS, JOINTS
from diagnostics.getup_independent_native import DT


class FakeSim:
    def __init__(self):
        self.home = np.zeros(14)
        self.lower = np.full(14, -1.)
        self.upper = np.full(14, 1.)
        self.n = 0

    def restore(self, state):
        self.n = state

    def sensor(self, name):
        if name == 'upvector':
            return np.array([self.n, -self.n, 1.])
        return np.array([self.n + 1., self.n + 2., self.n + 3.])

    def step_target(self, target):
        self.n += 1

    def physical_valid(self):
        return True


class ReferenceFeedbackTest(unittest.TestCase):
    def test_reference_error_zero_means_zero_correction(self):
        for scale in (0., 1., -2.):
            np.testing.assert_array_equal(residual(np.zeros(4), np.full(8, scale)),
                                          np.zeros(len(JOINTS)))

    def test_bilateral_mapping_and_cap(self):
        out = residual(np.ones(4), np.full(8, 2.))
        self.assertLessEqual(np.abs(out).max(), .18)
        self.assertEqual(out[0], -out[4])
        self.assertEqual(out[1], -out[5])
        self.assertEqual(out[2], out[6])
        self.assertEqual(out[3], out[7])
        np.testing.assert_array_equal(out[8:], np.zeros(2))

    def test_body_imu_feature_order(self):
        np.testing.assert_allclose(features(FakeSim()), [0., 0., .3, .15])

    def test_reference_is_sampled_before_control(self):
        ref = record_reference(FakeSim(), 0, {}, True, np.zeros((3, 14)))
        np.testing.assert_array_equal(ref[:, 0], [0., 1., 2.])

    def test_long_hold_does_not_change_short_reference_prefix(self):
        sim = FakeSim()
        ids = np.arange(len(JOINTS))
        offsets = np.full((3, len(JOINTS)), 2.)
        short, phase = targets_for(sim, offsets, ids, 2.)
        long, long_phase = targets_for(sim, offsets, ids, 35.)
        np.testing.assert_array_equal(long[:len(short)], short)
        np.testing.assert_array_equal(long_phase[:len(phase)], phase)
        self.assertEqual(len(short), sum(round(t / DT) for t in (*DURATIONS, 2.)))
        self.assertLessEqual(np.abs(short).max(), 1.)
        np.testing.assert_array_equal(short[-1], sim.home)


if __name__ == '__main__':
    unittest.main()
