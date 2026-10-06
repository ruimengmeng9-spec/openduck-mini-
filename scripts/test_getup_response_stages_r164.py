import unittest
import numpy as np
from diagnostics.audit_getup_response_stages_r164 import gain_matrix,bounded_apply,projection_fraction,residual

class AuditTests(unittest.TestCase):
    def test_gain_matrix_original_mapping(self):
        rng=np.random.default_rng(264)
        for _ in range(100):
            g=rng.uniform(-2,2,8);e=rng.uniform(-.01,.01,4)
            np.testing.assert_allclose(gain_matrix(g)@e,residual(e,g),rtol=0,atol=1e-17)

    def test_float32_rounding_bound(self):
        rng=np.random.default_rng(264)
        for _ in range(1000):
            g=rng.uniform(-2,2,8);truth=rng.uniform(-4,4,4);e=truth.astype(np.float32)
            b=np.abs(gain_matrix(g))@(np.abs(np.spacing(e)).astype(float)/2)+2e-14
            assert np.all(np.abs(residual(truth,g)-residual(e.astype(float),g))<=b)

    def test_joint_slew_lipschitz(self):
        rng=np.random.default_rng(264)
        for _ in range(100):
            a=rng.normal(size=14);delta=rng.uniform(-1e-7,1e-7,14);prev=rng.normal(size=14)
            x=bounded_apply(a,prev,-np.ones(14),np.ones(14));y=bounded_apply(a+delta,prev,-np.ones(14),np.ones(14))
            assert np.all(np.abs(x-y)<=np.abs(delta)+1e-15)

    def test_fixed_projection_not_minimum_search(self):
        t=np.array([[1.,0.],[0.,0.]])
        self.assertEqual(projection_fraction(np.array([[0.,1.],[1.,0.]]),t),0.)
        self.assertEqual(projection_fraction(np.array([[1.,0.],[0.,0.]]),t),1.)
        self.assertIsNone(projection_fraction(np.zeros((2,2)),t))

if __name__=='__main__':unittest.main()
