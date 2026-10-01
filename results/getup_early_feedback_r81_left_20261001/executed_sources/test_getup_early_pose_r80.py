import unittest
from types import SimpleNamespace
import numpy as np
from diagnostics.train_getup_early_pose_r80 import edited_early, BOUND, TRAIN, HELDOUT
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS


class EarlyPoseTests(unittest.TestCase):
    def setUp(self):
        ids = {name: i for i, name in enumerate(JOINTS)}
        self.sim = SimpleNamespace(lower=np.full(14, -1.), upper=np.full(14, 1.),
            model=SimpleNamespace(actuator=lambda name: SimpleNamespace(id=ids[name])))
        self.ck = {'prefix_targets': np.zeros((450, 14)),
                   'prefix_phases': np.r_[np.zeros(308, int), np.ones(100, int), np.full(42, 2)]}

    def test_zero_exactly_reproduces_baseline(self):
        np.testing.assert_array_equal(edited_early(self.sim, self.ck, np.zeros(16))['prefix_targets'],
                                      self.ck['prefix_targets'])

    def test_only_early_legs_change_without_touching_input(self):
        output = edited_early(self.sim, self.ck, np.full(16, BOUND))['prefix_targets']
        np.testing.assert_array_equal(output[:26], self.ck['prefix_targets'][:26])
        np.testing.assert_array_equal(output[175:], self.ck['prefix_targets'][175:])
        np.testing.assert_array_equal(output[:, 8:], self.ck['prefix_targets'][:, 8:])
        np.testing.assert_allclose(output[75, :8], BOUND)
        np.testing.assert_allclose(output[125, :8], BOUND)
        self.assertLessEqual(np.abs(np.diff(output[:, 0])).max(), .004 + 1e-10)
        np.testing.assert_array_equal(self.ck['prefix_targets'], np.zeros((450, 14)))

    def test_original_limits_preserved(self):
        self.ck['prefix_targets'][:, :8] = .95
        output = edited_early(self.sim, self.ck, np.full(16, BOUND))['prefix_targets']
        self.assertLessEqual(output.max(), 1.)
        np.testing.assert_array_equal(self.sim.upper, np.ones(14))

    def test_invalid_parameters_rejected(self):
        for value in (np.zeros(15), np.full(16, np.nan), np.full(16, BOUND + .001)):
            with self.assertRaises(ValueError): edited_early(self.sim, self.ck, value)

    def test_noncontiguous_phase_rejected(self):
        self.ck['prefix_phases'][50] = 1
        with self.assertRaises(ValueError): edited_early(self.sim, self.ck, np.zeros(16))

    def test_training_does_not_use_new_validation_seeds(self):
        self.assertFalse(set(TRAIN) & set(HELDOUT))
        self.assertEqual(len(HELDOUT), 40)


if __name__ == '__main__': unittest.main()
