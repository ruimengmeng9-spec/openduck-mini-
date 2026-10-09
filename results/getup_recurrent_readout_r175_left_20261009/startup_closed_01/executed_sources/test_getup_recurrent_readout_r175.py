import inspect
import unittest
import numpy as np
from diagnostics import train_getup_recurrent_readout_r175 as run


class RecurrentReadoutTests(unittest.TestCase):
    def setUp(self):
        self.a,self.b,self.w,_=run.fixed_features();self.x=np.zeros(55,np.float32);self.h=np.zeros(16)
    def call(self,w=None,x=None,n=None,x0=None,n0=None,h=None):
        return run.recurrent_feedback(self.w if w is None else w,self.x if x is None else x,self.x if n is None else n,self.x if x0 is None else x0,self.x if n0 is None else n0,self.h if h is None else h,self.a,self.b)
    def test_causal_signature(self):
        self.assertEqual(list(inspect.signature(run.recurrent_feedback).parameters),['parameters','current','nominal','initial','initial_nominal','state','input_matrix','recurrent_matrix'])
    def test_nominal_actual_scalar_sequence_exact_zero(self):
        rng=np.random.default_rng(275)
        for _ in range(529):
            x=rng.normal(size=55).astype(np.float32)
            e,self.h,p,d=self.call(x=x,n=x)
            for v in (e,self.h,p,d):np.testing.assert_array_equal(v,np.zeros_like(v))
    def test_first_step_nonzero_initial_error_zero(self):
        x=np.random.default_rng(275).normal(size=55).astype(np.float32)
        for v in self.call(x=x,x0=x):np.testing.assert_array_equal(v,np.zeros_like(v))
    def test_zero_readout_action_and_merge_identity(self):
        x=self.x.copy();x[6]=.1
        e,h,_,_=self.call(w=np.zeros(run.SHAPE),x=x)
        np.testing.assert_array_equal(e,np.zeros(14));self.assertTrue(np.any(h))
        target=np.zeros(14)
        self.assertIs(run.local.merge_target(target,target,e,np.arange(14),-np.ones(14),np.ones(14)),target)
    def test_recurrent_history_changes_same_current_output(self):
        x=self.x.copy();x[3]=.03
        _,h,_,_=self.call(x=x)
        first=self.call(x=x);second=self.call(x=x,h=h)
        self.assertFalse(np.array_equal(first[0],second[0]))
    def test_all_fifty_causal_channels_covered(self):
        for i in range(50):
            x=self.x.copy();x[i]=float(run.SCALE[i]);e,h,p,d=self.call(x=x)
            self.assertEqual(np.count_nonzero(d),1);self.assertGreater(np.abs(h).max(),0)
    def test_unused_original_reference_error_and_phase_fields(self):
        x=self.x.copy();x[50:]=100
        for v in self.call(x=x):np.testing.assert_array_equal(v,np.zeros_like(v))
    def test_native_velocity_units(self):
        x=self.x.copy();x[20]=.05
        result=self.call(x=x)
        self.assertEqual(result[3][20],float(np.float32(.05))/.05)
    def test_signed_symmetric_network_without_biases(self):
        x=self.x.copy();x[:50]=np.linspace(-.1,.1,50)
        for a,b in zip(self.call(x=x),self.call(x=-x)):np.testing.assert_array_equal(a,-b)
    def test_cross_sensor_and_all_actuator_coordination(self):
        x=self.x.copy();x[0]=.1
        e,_,_,_=self.call(x=x)
        self.assertEqual(np.count_nonzero(e),14)
    def test_recurrence_fixed_contraction(self):
        self.assertLessEqual(np.abs(self.b).sum(1).max(),.75+1e-14)
        a,b,w,_=run.fixed_features()
        for x,y in [(a,self.a),(b,self.b),(w,self.w)]:np.testing.assert_array_equal(x,y)
    def test_hidden_and_output_bound_for_long_sequence(self):
        rng=np.random.default_rng(275);w=np.full(run.SHAPE,run.BOUND)
        for _ in range(1000):
            e,self.h,_,_=self.call(w=w,x=rng.normal(size=55).astype(np.float32),h=self.h)
            self.assertLessEqual(np.abs(self.h).max(),1.);self.assertLessEqual(np.abs(e).max(),.18)
    def test_original_total_target_cap_and_joint_bound(self):
        rng=np.random.default_rng(275)
        for _ in range(100):
            reference=rng.uniform(-.4,.4,14);target=reference+rng.uniform(-.18,.18,14)
            extra=self.call(x=rng.normal(size=55).astype(np.float32))[0]
            final=run.local.merge_target(target,reference,extra,np.arange(14),-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(final-reference).max(),.18+1e-12);self.assertLessEqual(np.abs(final).max(),1.)
    def test_invalid_or_nonfinite_input_rejected(self):
        for w in [np.zeros(3),np.full(run.SHAPE,np.nan),np.full(run.SHAPE,.020001)]:
            with self.assertRaises(ValueError):self.call(w=w)
        with self.assertRaises(ValueError):self.call(x=np.full(55,np.nan))
        with self.assertRaises(ValueError):self.call(h=np.full(16,1.01))
    def test_nonzero_initial_context_subtraction_is_causal(self):
        x=self.x.copy();x[0]=.2;x0=self.x.copy();x0[0]=.1
        self.assertAlmostEqual(self.call(x=x,x0=x0)[3][0],float(np.float32(.2))-float(np.float32(.1)))


if __name__=='__main__':unittest.main()
