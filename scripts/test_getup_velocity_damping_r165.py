import inspect
import unittest
import numpy as np
from diagnostics.train_getup_velocity_damping_r165 import damping_feedback,merge_damping,apply_limits

GROUPS=np.array([0]*5+[2]*4+[1]*5)

class DampingTests(unittest.TestCase):
    def test_causal_signature(self):
        self.assertEqual(list(inspect.signature(damping_feedback).parameters),['parameters','current','nominal','groups'])

    def test_actual_scalar_nominal_exact_zero(self):
        x=np.random.default_rng(265).normal(size=55).astype(np.float32)
        e,v=damping_feedback([.05,.03,.01],x,x,GROUPS)
        np.testing.assert_array_equal(e,np.zeros(14));np.testing.assert_array_equal(v,np.zeros(14))

    def test_zero_parameters(self):
        e,_=damping_feedback([0,0,0],np.ones(55),np.zeros(55),GROUPS);np.testing.assert_array_equal(e,np.zeros(14))

    def test_native_velocity_units(self):
        x=np.zeros(55,np.float32);x[20]=.05
        e,v=damping_feedback([.02,0,0],x,np.zeros(55),GROUPS)
        self.assertAlmostEqual(v[0],float(np.float32(.05))/.05);self.assertAlmostEqual(e[0],-.18*np.tanh(.02*v[0]))

    def test_no_position_imu_history_or_initial_input(self):
        x=np.zeros(55,np.float32);x[:20]=11.;x[34:]=22.
        e,v=damping_feedback([.05]*3,x,np.zeros(55),GROUPS)
        np.testing.assert_array_equal(e,np.zeros(14));np.testing.assert_array_equal(v,np.zeros(14))

    def test_fourteen_independent_channels(self):
        for i in range(14):
            x=np.zeros(55,np.float32);x[20+i]=.1;e,_=damping_feedback([.02]*3,x,np.zeros(55),GROUPS)
            self.assertLess(e[i],0);self.assertEqual(np.count_nonzero(e),1)

    def test_nonnegative_group_separation_and_odd_response(self):
        x=np.zeros(55,np.float32);x[20:34]=.1
        a,_=damping_feedback([.01,0,0],x,np.zeros(55),GROUPS)
        b,_=damping_feedback([.01,0,0],-x,np.zeros(55),GROUPS)
        np.testing.assert_array_equal(a,-b);self.assertEqual(np.count_nonzero(a),5)

    def test_finite_shapes_and_coefficients(self):
        for p in [[-.0001,0,0],[.05001,0,0],[np.nan,0,0],[0,0]]:
            with self.assertRaises(ValueError):damping_feedback(p,np.zeros(55),np.zeros(55),GROUPS)
        with self.assertRaises(ValueError):damping_feedback([0]*3,np.full(55,np.nan),np.zeros(55),GROUPS)
        with self.assertRaises(ValueError):damping_feedback([0]*3,np.zeros(54),np.zeros(55),GROUPS)

    def test_reference_cap_joint_and_slew_sign_invariant(self):
        rng=np.random.default_rng(265)
        for _ in range(100):
            reference=rng.uniform(-.5,.5,14);target=reference+rng.uniform(-.18,.18,14);err=rng.uniform(-10,10,14)
            extra=-.18*np.tanh(.05*err);lower=np.full(14,-1.);upper=np.ones(14)
            changed=merge_damping(target,reference,extra,err,lower,upper)
            self.assertLessEqual(np.abs(changed-reference).max(),.18+1e-12)
            prev=rng.uniform(-.7,.7,14)
            diff=apply_limits(changed,prev,lower,upper)-apply_limits(target,prev,lower,upper)
            self.assertTrue(np.all(diff*err<=1e-13))

    def test_zero_merge_identity(self):
        target=np.zeros(14);result=merge_damping(target,np.zeros(14),np.zeros(14),np.ones(14),-np.ones(14),np.ones(14))
        self.assertIs(target,result)

if __name__=='__main__':unittest.main()
