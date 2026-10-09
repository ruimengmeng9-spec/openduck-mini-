"""Fixed causal local phase regression before any dynamic replay."""
import inspect
import unittest
import numpy as np
from diagnostics import probe_getup_phase_coordinate_r185b as run


class PhaseTests(unittest.TestCase):
    def setUp(self):
        self.zero=np.zeros(14,dtype=np.float32)
        self.reference=np.arange(2279)[:,None]*np.linspace(.001,.002,14)[None,:]
        self.tangent=np.full(14,.004)
    def call(self,**kw):
        defaults=dict(current=self.zero,nominal=self.zero,initial=self.zero,initial_nominal=self.zero,tangent=self.tangent,reference=self.reference,control=10,enabled=True)
        defaults.update(kw);return run.local_phase(**defaults)
    def test_worker_frozen_snapshot_loaded(self):
        run.local.WEIGHTS=None
        run.init_worker(run.ROOT/'outputs/getup_phase_coordinate_r185_smoke_20261010/frozen')
        self.assertIsNotNone(run.local.WEIGHTS);self.assertIn('feature_mode',run.local.WEIGHTS)
        self.assertEqual(run.local.NOMINAL.shape,(2279,55));self.assertEqual(run.TANGENT.shape,(529,14))
    def test_causal_signature(self):
        self.assertEqual(list(inspect.signature(run.local_phase).parameters),['current','nominal','initial','initial_nominal','tangent','reference','control','enabled'])
        source=inspect.getsource(run.local_phase)
        for forbidden in ('seed','case','qpos','qvel','observe','nearest','argmin'):self.assertNotIn(forbidden,source)
    def test_actual_nominal529_scalar_zero(self):
        with np.load(run.local.prior.OUTPUT/'snapshot/case_None/trajectory.npz',allow_pickle=False) as z:q=z['observations'][:529,6:20].copy()
        with np.load(run.local.prior.program.REFERENCE,allow_pickle=False) as z:ref=z['targets'].copy()
        v=np.gradient(q.astype(float),axis=0)
        for k in range(529):
            extra,s,d=run.local_phase(q[k],q[k],q[0],q[0],v[k],ref,k,True)
            np.testing.assert_array_equal(extra,np.zeros(14));np.testing.assert_array_equal(d,np.zeros(14));self.assertEqual(s,0.)
    def test_initial_anchor_nonzero_exact_zero(self):
        q=np.linspace(-.4,.4,14).astype(np.float32)
        extra,s,d=self.call(current=q,initial=q,control=0)
        for x in (extra,d):np.testing.assert_array_equal(x,np.zeros(14))
        self.assertEqual(s,0.)
    def test_zero_merge_original_object(self):
        target=np.arange(14,dtype=float)/10
        self.assertIs(run.local.merge_target(target,target,self.call()[0],np.arange(14),-np.ones(14)*3,np.ones(14)*3),target)
    def test_disabled_object_and_signals(self):
        e,s,d=self.call(current=np.ones(14),enabled=False)
        np.testing.assert_array_equal(e,self.zero);np.testing.assert_array_equal(d,self.zero);self.assertEqual(s,0.)
    def test_coordinate_known_units_and_positive_direction(self):
        q=np.full(14,.001,dtype=np.float32);e,s,d=self.call(current=q)
        expected=float(np.dot(self.tangent,q.astype(float))/(np.dot(self.tangent,self.tangent)+.02**2))
        self.assertEqual(s,expected);self.assertGreater(s,0.)
        np.testing.assert_allclose(e,s*np.linspace(.001,.002,14),atol=1e-17,rtol=1e-12)
    def test_negative_coordinate_and_no_clock_accumulator(self):
        q=np.full(14,-.001,dtype=np.float32)
        a=self.call(current=q);b=self.call(current=-q)
        self.assertEqual(a[1],-b[1]);self.assertLess(a[1],0.)
        for x,y in zip(a,self.call(current=q)):np.testing.assert_array_equal(x,y)
    def test_no_tangent_no_phase_action(self):
        e,s,_=self.call(current=np.ones(14),tangent=np.zeros(14))
        np.testing.assert_array_equal(e,self.zero);self.assertEqual(s,0.)
    def test_coordinate_and_curve_endpoint_bounds(self):
        e,s,_=self.call(current=np.ones(14)*100,control=528)
        self.assertEqual(s,1.);np.testing.assert_array_equal(e,self.zero)
        for value in (-100.,100.):self.assertLessEqual(abs(self.call(current=np.full(14,value))[1]),1.)
    def test_all14_curve_coordinates_participate(self):
        e,s,_=self.call(current=np.ones(14)*.001)
        self.assertEqual(np.count_nonzero(e),14);self.assertGreater(s,0.)
    def test_home_original(self):
        for k in (529,2278):
            e,s,d=self.call(current=np.ones(14),control=k)
            np.testing.assert_array_equal(e,self.zero);np.testing.assert_array_equal(d,self.zero);self.assertEqual(s,0.)
    def test_total_original_cap_joint_and_slew(self):
        rng=np.random.default_rng(285)
        for _ in range(100):
            ref=rng.uniform(-.4,.4,14);target=ref+rng.uniform(-.18,.18,14)
            extra=self.call(current=rng.normal(size=14).astype(np.float32))[0]
            final=run.local.merge_target(target,ref,extra,np.arange(14),-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(final-ref).max(),.18+1e-12)
            prev=np.zeros(14);planned=run.planner.apply_limits(final,prev,-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(planned).max(),5.24*.02)
    def test_invalid_shape_nonfinite_and_phase(self):
        for kw in ({'current':np.zeros(55)},{'initial':np.full(14,np.nan)},{'tangent':np.zeros(3)},{'reference':np.zeros((2,13))},{'control':-.1},{'control':2279}):
            with self.assertRaises(ValueError):self.call(**kw)


if __name__=='__main__':unittest.main()
