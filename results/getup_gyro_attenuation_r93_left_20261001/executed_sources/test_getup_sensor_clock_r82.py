import unittest
import numpy as np
from diagnostics.probe_getup_sensor_clock_r82 import (
    interpolate, clock_rate, phase0_end, advance, passed, TRAIN, HELDOUT)


class SensorClockTest(unittest.TestCase):
    def test_interpolation_and_bounds(self):
        np.testing.assert_allclose(interpolate(np.array([[0.,2.],[2.,4.]]), .5), [1.,3.])
        for cursor in (-1, 2, float('nan')):
            with self.assertRaises(ValueError): interpolate(np.zeros((2,4)), cursor)

    def test_identity_observation_is_identity_clock(self):
        ref = np.arange(400.)[:,None] * np.ones((1,4))
        for cursor in (0., 40., 40.5, 307.):
            self.assertEqual(clock_rate(interpolate(ref,cursor),ref,cursor,307,.6,20)[0],1.)

    def test_behind_and_ahead_bounded(self):
        ref = np.arange(400.)[:,None] * np.ones((1,4))
        self.assertEqual(clock_rate(ref[80],ref,100.,307,.6,20)[0],.25)
        self.assertEqual(clock_rate(ref[120],ref,100.,307,.6,20)[0],1.75)

    def test_cursor_never_rewinds_or_skips_phase_boundary(self):
        for rate in (.25,1.,1.75):
            self.assertGreater(advance(100.,307,rate),100.)
            self.assertEqual(advance(306.9,307,rate),307.)
            self.assertEqual(advance(307.,307,rate),308.)

    def test_phase_layout_rejected(self):
        self.assertEqual(phase0_end(np.r_[np.zeros(308),np.ones(100)]),307)
        with self.assertRaises(ValueError): phase0_end(np.r_[np.ones(2),np.zeros(308)])

    def test_continuous_gate_requires_complete_valid_path(self):
        row=dict(valid=True,path_completed=True,strict_tail_s=30.)
        self.assertTrue(passed(row,30.))
        for key,value in [('valid',False),('path_completed',False),('strict_tail_s',29.99)]:
            self.assertFalse(passed({**row,key:value},30.))

    def test_fresh_split(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT))
        self.assertFalse(set(range(781000,781040))&set(HELDOUT))


if __name__ == '__main__': unittest.main()
