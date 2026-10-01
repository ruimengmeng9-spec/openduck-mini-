import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from diagnostics.getup_independent_native import (
    canonical_quaternion, decode_full_range, standing_sample, sustained_tail, RecoverySim, SLEW, DT)
from diagnostics.getup_feedback_reference import ready_to_handoff


class IndependentRecoveryTests(unittest.TestCase):
    def test_four_distinct_orientations_and_correct_down_axis(self):
        qs = [canonical_quaternion(p) for p in ('prone', 'supine', 'left_side', 'right_side')]
        self.assertEqual(len({tuple(q.round(8)) for q in qs}), 4)
        for q in qs:
            self.assertAlmostEqual(float(np.linalg.norm(q)), 1.)
        # R_zx=2(xz-wy), R_zy=2(yz+wx).
        self.assertAlmostEqual(-2*qs[0][0]*qs[0][2], -1.)
        self.assertAlmostEqual(-2*qs[1][0]*qs[1][2], 1.)
        self.assertAlmostEqual(2*qs[2][0]*qs[2][1], -1.)
        self.assertAlmostEqual(2*qs[3][0]*qs[3][1], 1.)

    def test_full_decoder_does_not_expand_physical_range(self):
        lo, hi = np.array([-1., -.5]), np.array([1., 2.])
        np.testing.assert_array_equal(decode_full_range(np.array([-2., 3.]), lo, hi), [-1., 2.])
        with self.assertRaises(ValueError):
            decode_full_range(np.array([np.nan, 0.]), lo, hi)

    def test_transient_upright_and_airborne_are_not_recovery(self):
        self.assertFalse(standing_sample(1., .16, [False, False], False, 0., 0.))
        self.assertFalse(standing_sample(1., .16, [True, True], True, 0., 0.))
        self.assertFalse(standing_sample(1., .16, [True, True], False, 1., 0.))
        self.assertTrue(standing_sample(1., .16, [True, True], False, .01, .02))
        self.assertAlmostEqual(sustained_tail([False]*100+[True]), .02)
        self.assertAlmostEqual(sustained_tail([True]*100+[False]), 0.)
        self.assertAlmostEqual(sustained_tail([False]+[True]*100), 2.)

    def test_target_slew_and_three_distinct_history_frames(self):
        sim = RecoverySim.__new__(RecoverySim)
        sim.lower, sim.upper = np.array([-2.]), np.array([2.])
        sim.home, sim.prev = np.zeros(1), np.zeros(1)
        sim.history = np.zeros((3, 1), dtype=np.float32)
        sim.data, sim.model = SimpleNamespace(ctrl=np.zeros(1)), object()
        with patch('diagnostics.getup_independent_native.mujoco.mj_step'):
            for value in (.25, .50, .75):
                previous = sim.prev.copy()
                applied = sim.step_target(np.array([value]))
                self.assertLessEqual(float(abs(applied-previous)[0]), SLEW*DT+1e-8)
        np.testing.assert_array_equal(sim.history[:, 0], [3., 2., 1.])

    def test_sensor_address_not_sensor_id(self):
        sim = RecoverySim.__new__(RecoverySim)
        sim.sensor_indices = {'gyro': (5, 3)}
        sim.data = SimpleNamespace(sensordata=np.arange(12))
        np.testing.assert_array_equal(sim.sensor('gyro'), [5, 6, 7])

    def test_handoff_rejects_no_support_and_head_contact(self):
        m = dict(up_z=.99, height_m=.16, feet=[True, True], torso_contact=False,
                 angular_speed_rad_s=.1, linear_speed_mps=.02)
        self.assertTrue(ready_to_handoff(m))
        self.assertFalse(ready_to_handoff(dict(m, feet=[False, False])))
        self.assertFalse(ready_to_handoff(dict(m, torso_contact=True)))


if __name__ == '__main__':
    unittest.main()
