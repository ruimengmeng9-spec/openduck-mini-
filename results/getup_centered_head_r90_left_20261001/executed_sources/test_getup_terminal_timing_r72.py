import unittest
from types import SimpleNamespace

import numpy as np

from diagnostics.probe_getup_terminal_timing_r72 import timed_targets
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.sim = SimpleNamespace(home=np.zeros(14), lower=np.full(14, -1.),
                                   upper=np.full(14, 1.))
        self.sim.model = SimpleNamespace(actuator=lambda name: SimpleNamespace(id=JOINTS.index(name)))
        self.ck = {'prefix_targets': np.full((10, 14), .1),
                   'prefix_phases': np.zeros(10, dtype=int),
                   'offsets': np.full((3, len(JOINTS)), 2.)}

    def test_prefix_is_unchanged(self):
        targets, phases = timed_targets(self.sim, self.ck, (.4, .6, .8), 2.)
        np.testing.assert_array_equal(targets[:10], self.ck['prefix_targets'])
        np.testing.assert_array_equal(phases[:10], self.ck['prefix_phases'])
        self.assertEqual(len(targets), 10 + 20 + 30 + 40 + 100)

    def test_only_selected_dwell_changes(self):
        short, ps = timed_targets(self.sim, self.ck, (.4, .36, .8), 2.)
        long, pl = timed_targets(self.sim, self.ck, (.4, .9, .8), 2.)
        for phase in [0, 4, 6, 7]:
            np.testing.assert_array_equal(short[ps == phase], long[pl == phase])
        self.assertEqual(sum(ps == 5), 18)
        self.assertEqual(sum(pl == 5), 45)

    def test_actual_joint_limits_and_home_hold(self):
        targets, phases = timed_targets(self.sim, self.ck, (.4, .6, .8), 35.)
        self.assertLessEqual(np.abs(targets).max(), 1.)
        np.testing.assert_array_equal(targets[phases == 7], np.zeros((1750, 14)))


if __name__ == '__main__':
    unittest.main()
