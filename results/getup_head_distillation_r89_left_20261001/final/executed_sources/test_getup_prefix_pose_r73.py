import unittest
from types import SimpleNamespace
import numpy as np

from diagnostics.train_getup_prefix_pose_r73 import edited_prefix, success
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS


class PrefixPoseTests(unittest.TestCase):
    def setUp(self):
        self.sim = SimpleNamespace(lower=np.full(14, -1.), upper=np.full(14, 1.))
        self.sim.model = SimpleNamespace(actuator=lambda name: SimpleNamespace(id=JOINTS.index(name)))
        self.ck = {'prefix_targets': np.zeros((450, 14)),
                   'prefix_phases': np.r_[np.zeros(308, dtype=int), np.ones(142, dtype=int)]}

    def test_zero_delta_exactly_reproduces_original(self):
        output = edited_prefix(self.sim, self.ck, np.zeros(32))
        np.testing.assert_array_equal(output['prefix_targets'], self.ck['prefix_targets'])

    def test_only_final_two_seconds_of_recovery_legs_change(self):
        output = edited_prefix(self.sim, self.ck, np.full(32, .35))['prefix_targets']
        np.testing.assert_array_equal(output[:208], self.ck['prefix_targets'][:208])
        np.testing.assert_array_equal(output[308:], self.ck['prefix_targets'][308:])
        np.testing.assert_array_equal(output[:, 8:], np.zeros((450, 6)))
        np.testing.assert_allclose(output[307, :8], .35)
        self.assertLessEqual(np.abs(np.diff(output[208:308, 0])).max(), .014 + 1e-8)
        np.testing.assert_array_equal(self.ck['prefix_targets'], np.zeros((450, 14)))

    def test_targets_respect_original_joint_limits(self):
        output = edited_prefix(self.sim, self.ck, np.full(32, 10.))['prefix_targets']
        self.assertLessEqual(output.max(), 1.)

    def test_gate_requires_physics_complete_and_continuous_standing(self):
        row = dict(valid=True, completed_steps=100, strict_tail_s=30.)
        self.assertTrue(success(row, 100, 30.))
        for patch in [dict(valid=False), dict(completed_steps=99), dict(strict_tail_s=29.98)]:
            self.assertFalse(success({**row, **patch}, 100, 30.))


if __name__ == '__main__':
    unittest.main()
