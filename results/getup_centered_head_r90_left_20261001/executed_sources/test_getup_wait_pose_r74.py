import unittest
from types import SimpleNamespace
import numpy as np
from diagnostics.train_getup_wait_pose_r74 import edited_wait, BOUND
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS


class WaitPoseTests(unittest.TestCase):
    def setUp(self):
        ids = {name: i for i, name in enumerate(JOINTS)}
        self.sim = SimpleNamespace(lower=np.full(14, -1.), upper=np.full(14, 1.),
            model=SimpleNamespace(actuator=lambda name: SimpleNamespace(id=ids[name])))
        self.ck = {'prefix_targets': np.zeros((450, 14)),
                   'prefix_phases': np.r_[np.zeros(308, int), np.ones(100, int), np.full(42, 2)]}

    def test_zero_exactly_matches_baseline(self):
        np.testing.assert_array_equal(edited_wait(self.sim, self.ck, np.zeros(10))['prefix_targets'],
                                      self.ck['prefix_targets'])

    def test_only_wait_and_named_joints_change(self):
        target = edited_wait(self.sim, self.ck, np.full(10, BOUND))['prefix_targets']
        np.testing.assert_array_equal(target[:308], self.ck['prefix_targets'][:308])
        np.testing.assert_array_equal(target[407:], self.ck['prefix_targets'][407:])
        np.testing.assert_array_equal(target[:, 10:], self.ck['prefix_targets'][:, 10:])
        self.assertLessEqual(np.abs(np.diff(target[:, 0])).max(), .025 + 1e-10)
        np.testing.assert_allclose(target[350, :10], BOUND)
        np.testing.assert_array_equal(self.ck['prefix_targets'], np.zeros((450, 14)))

    def test_existing_limits_preserved(self):
        self.ck['prefix_targets'][:, :10] = .95
        result = edited_wait(self.sim, self.ck, np.full(10, BOUND))['prefix_targets']
        self.assertLessEqual(result.max(), 1.)
        np.testing.assert_array_equal(self.sim.upper, np.ones(14))

    def test_invalid_parameters_rejected(self):
        for v in [np.zeros(9), np.full(10, np.nan), np.full(10, BOUND + .001)]:
            with self.assertRaises(ValueError): edited_wait(self.sim, self.ck, v)

    def test_noncontiguous_phase_rejected(self):
        self.ck['prefix_phases'][350] = 0
        with self.assertRaises(ValueError): edited_wait(self.sim, self.ck, np.zeros(10))


if __name__ == '__main__':
    unittest.main()
