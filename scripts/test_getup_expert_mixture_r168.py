import inspect
import unittest
import numpy as np
from diagnostics.train_getup_expert_mixture_r168 import mixture_feedback,features,local,IDS,PROBE

class MixtureTests(unittest.TestCase):
    def setUp(self):
        self.n=np.zeros(55,np.float32);self.x=self.n.copy();self.x[15:18]=[.02,-.03,.04]
        self.base=np.array([.2,-.3,.4,.1,.2,.3]);self.experts=np.stack([np.zeros(6),self.base,-self.base,2*self.base])
    def run_kernel(self,p=PROBE,x=None,initial=None,nominal=None):
        return mixture_feedback(p,self.x if x is None else x,self.n if nominal is None else nominal,
            self.n if initial is None else initial,self.n,self.base,self.experts)
    def test_causal_signature(self):
        self.assertEqual(list(inspect.signature(mixture_feedback).parameters),['parameters','current','nominal','initial','initial_nominal','base_gains','expert_gains'])
    def test_actual_scalar_nominal_exact_zero(self):
        x=np.random.default_rng(268).normal(size=55).astype(np.float32)
        values=mixture_feedback(PROBE,x,x,x,x,self.base,self.experts)
        for k in [0,1,2,3,5]:np.testing.assert_array_equal(values[k],np.zeros_like(values[k]))
        np.testing.assert_array_equal(values[4],[1,0,0,0,0])
    def test_zero_parameters_exact_base(self):
        m,b,e,g,w,phi=self.run_kernel(np.zeros((4,12)));np.testing.assert_array_equal(m,b)
        np.testing.assert_array_equal(b,local.local_feedback(self.x,self.n,IDS,self.base));np.testing.assert_array_equal(w,[1,0,0,0,0])
    def test_nonzero_initial_context_first_control(self):
        m,b,e,g,w,phi=self.run_kernel(initial=self.x);np.testing.assert_array_equal(m,b)
        np.testing.assert_array_equal(g,np.zeros(4));np.testing.assert_array_equal(phi,np.zeros(12))
    def test_weights_nonnegative_normalized_and_convex(self):
        rng=np.random.default_rng(268)
        for _ in range(100):
            x=rng.normal(size=55).astype(np.float32);m,b,e,g,w,_=self.run_kernel(rng.uniform(-1,1,(4,12)),x)
            self.assertTrue(np.all(w>=0));self.assertAlmostEqual(w.sum(),1)
            self.assertTrue(np.all(g>=0) and np.all(g<=1));v=np.vstack([b,e])
            self.assertTrue(np.all(m>=v.min(0)-1e-14) and np.all(m<=v.max(0)+1e-14));self.assertLessEqual(abs(m).max(),.18+1e-14)
    def test_native_velocity_and_position_units(self):
        x=self.n.copy();x[29]=.05;x[15]=.05;x[3]=.05;x[0]=1.
        v=features(x,self.n);self.assertAlmostEqual(v[0],1);self.assertAlmostEqual(v[3],float(np.float32(.05))/.05)
        self.assertAlmostEqual(v[6],float(np.float32(.05))/.05);self.assertAlmostEqual(v[9],float(np.float32(.05))/.05)
    def test_unused_sensors_not_input(self):
        x=self.x.copy();used=np.r_[np.arange(6),6+IDS,20+IDS];x[np.setdiff1d(np.arange(55),used)]=10
        for a,b in zip(self.run_kernel(),self.run_kernel(x=x)):np.testing.assert_array_equal(a,b)
    def test_independent_multiple_expert_rows(self):
        p=np.zeros((4,12));p[2,6]=.5;m,b,e,g,w,_=self.run_kernel(p)
        self.assertGreater(g[2],0);self.assertEqual(np.count_nonzero(g),1);self.assertFalse(np.array_equal(m,b))
    def test_expert_row_permutation(self):
        a=self.run_kernel();order=[3,1,0,2]
        b=mixture_feedback(PROBE[order],self.x,self.n,self.n,self.n,self.base,self.experts[order])
        np.testing.assert_allclose(a[0],b[0],rtol=0,atol=1e-16)
    def test_output_mixture_not_gain_interpolation(self):
        m,b,e,g,w,_=self.run_kernel();gains=w[0]*self.base+(w[1:,None]*self.experts).sum(0)
        self.assertGreater(np.abs(m-local.local_feedback(self.x,self.n,IDS,gains)).max(),1e-7)
    def test_finite_bounds_shapes(self):
        for p in [np.zeros((3,12)),np.full((4,12),np.nan),np.full((4,12),1.001)]:
            with self.assertRaises(ValueError):self.run_kernel(p)
        with self.assertRaises(ValueError):self.run_kernel(x=np.full(55,np.nan))
        with self.assertRaises(ValueError):self.run_kernel(x=np.zeros(54))
        with self.assertRaises(ValueError):mixture_feedback(PROBE,self.x,self.n,self.n,self.n,self.base,np.ones((4,6))*2.001)
    def test_total_cap_and_zero_merge_identity(self):
        target=np.zeros(14);self.assertIs(target,local.merge_target(target,target,np.zeros(3),IDS,-np.ones(14),np.ones(14)))
        m,*_=self.run_kernel();target[IDS]=.17
        adjusted=local.merge_target(target,np.zeros(14),m,IDS,-np.ones(14),np.ones(14));self.assertLessEqual(abs(adjusted).max(),.18)

if __name__=='__main__':unittest.main()
