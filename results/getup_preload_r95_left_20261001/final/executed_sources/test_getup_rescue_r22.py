import unittest
from types import SimpleNamespace
import numpy as np
from diagnostics.search_getup_rescue_r22 import target_at, BOUNDS, INDICES


class RescueTargetTest(unittest.TestCase):
    def setUp(self):
        self.sim = SimpleNamespace(home=np.zeros(14), lower=np.full(14,-.5), upper=np.full(14,.5))

    def test_zero_knots_exact_home(self):
        for t in (0., .2, .4, .6, 1., 1.2, 30.):
            np.testing.assert_array_equal(target_at(self.sim,np.zeros((2,8)),t),self.sim.home)

    def test_interpolation_returns_to_home(self):
        knots = np.array([np.full(8,.2), np.full(8,-.2)])
        for t,delta in ((0.,.2),(.4,0.),(.6,-.2),(.9,-.1),(1.2,0.)):
            q=target_at(self.sim,knots,t)
            np.testing.assert_allclose(q[INDICES],delta,atol=1e-12)
            np.testing.assert_array_equal(np.delete(q,INDICES),np.zeros(6))

    def test_clips_to_joint_ranges(self):
        q=target_at(self.sim,np.broadcast_to(BOUNDS,(2,8)),0.)
        self.assertLessEqual(q.max(),.5)

    def test_reject_invalid_knots(self):
        for knots in (np.zeros((8,)),np.full((2,8),np.nan),np.full((2,8),2.)):
            with self.assertRaises(ValueError):target_at(self.sim,knots,0.)


if __name__=='__main__':unittest.main()
