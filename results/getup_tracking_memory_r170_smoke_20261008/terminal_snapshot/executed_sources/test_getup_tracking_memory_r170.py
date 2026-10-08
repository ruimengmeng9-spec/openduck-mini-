import inspect
import unittest
import numpy as np
from diagnostics.train_getup_tracking_memory_r170 import tracking_memory,controlled_memory,LEAK,DT

GROUPS=np.array([0]*5+[2]*4+[1]*5)

class TrackingMemoryTests(unittest.TestCase):
    def setUp(self):self.zero=np.zeros(55,np.float32);self.state=np.zeros(14)
    def test_causal_signature(self):
        self.assertEqual(list(inspect.signature(tracking_memory).parameters),['parameters','current','nominal','initial','initial_nominal','state','groups'])
    def test_actual_scalar_nominal_exact_zero(self):
        x=np.random.default_rng(270).normal(size=55).astype(np.float32)
        for _ in range(529):
            extra,z,delta,drive,_=tracking_memory([.05,.03],x,x,x,x,self.state,GROUPS)
            for value in (extra,z,delta,drive):np.testing.assert_array_equal(value,np.zeros(14))
            self.state=z
    def test_nonzero_initial_error_first_target_zero(self):
        x=np.random.default_rng(270).normal(size=55).astype(np.float32)
        e,z,d,u,_=tracking_memory([.05,.05],x,self.zero,x,self.zero,self.state,GROUPS)
        for value in (e,z,d,u):np.testing.assert_array_equal(value,np.zeros(14))
    def test_tracking_radian_units_and_target_sign(self):
        x=self.zero.copy();x[34]=.05
        e,z,d,u,_=tracking_memory([.02,.02],x,self.zero,self.zero,self.zero,self.state,GROUPS)
        self.assertEqual(d[0],float(np.float32(.05)));self.assertAlmostEqual(u[0],np.tanh(float(np.float32(.05))/.05));self.assertGreater(e[0],0)
        x=self.zero.copy();x[6]=.05
        opposite=tracking_memory([.02,.02],x,self.zero,self.zero,self.zero,self.state,GROUPS)
        np.testing.assert_array_equal(e,-opposite[0])
    def test_unread_channels_no_effect(self):
        x=self.zero.copy();x[:6]=9;x[20:34]=9;x[48:]=9
        for value in tracking_memory([.05,.05],x,self.zero,self.zero,self.zero,self.state,GROUPS)[:4]:np.testing.assert_array_equal(value,np.zeros(14))
    def test_prior_memory_distinguishes_identical_current_sensors(self):
        z=self.state.copy();z[0]=.3
        a=tracking_memory([.02,.02],self.zero,self.zero,self.zero,self.zero,z,GROUPS)
        b=tracking_memory([.02,.02],self.zero,self.zero,self.zero,self.zero,self.state,GROUPS)
        self.assertGreater(a[0][0],b[0][0]);self.assertEqual(a[1][0],LEAK*.3)
    def test_bounded_recurrence(self):
        x=self.zero.copy();x[34:48]=100
        for _ in range(3000):
            _,self.state,_,_,_=tracking_memory([.05,.05],x,self.zero,self.zero,self.zero,self.state,GROUPS)
            self.assertLessEqual(np.abs(self.state).max(),1.)
        np.testing.assert_array_equal(self.state[GROUPS==2],np.zeros(4))
    def test_group_separation_no_head_neck_output(self):
        x=self.zero.copy();x[34:48]=.1
        a=tracking_memory([.02,0],x,self.zero,self.zero,self.zero,self.state,GROUPS)[0]
        self.assertEqual(np.count_nonzero(a),5);np.testing.assert_array_equal(a[GROUPS!=0],np.zeros(9))
    def test_zero_parameters_and_merge_object_identity(self):
        x=self.zero.copy();x[34:48]=.1;target=np.zeros(14)
        result=controlled_memory([0,0],x,self.zero,self.zero,self.zero,self.state,GROUPS,target,np.zeros(14),np.zeros(14),-np.ones(14),np.ones(14))
        self.assertIs(result[0],target);np.testing.assert_array_equal(result[1],np.zeros(14))
    def test_antiwindup_rejects_saturation_driven_accumulation(self):
        x=self.zero.copy();x[34]=.1;target=np.zeros(14);target[0]=.18
        r=controlled_memory([.05,.05],x,self.zero,self.zero,self.zero,self.state,GROUPS,target,np.zeros(14),target.copy(),-np.ones(14),np.ones(14))
        self.assertTrue(r[5][0]);self.assertEqual(r[2][0],0);self.assertGreater(r[6][0],0)
    def test_antiwindup_allows_opposite_unwinding(self):
        x=self.zero.copy();x[34]=-.1;z=self.state.copy();z[0]=.5;target=np.zeros(14);target[0]=.18
        r=controlled_memory([.05,.05],x,self.zero,self.zero,self.zero,z,GROUPS,target,np.zeros(14),target.copy(),-np.ones(14),np.ones(14))
        self.assertFalse(r[5][0]);self.assertLess(r[2][0],LEAK*z[0])
    def test_slew_blockage_is_current_planning_not_future(self):
        x=self.zero.copy();x[34]=.1;target=np.zeros(14);target[0]=.15
        r=controlled_memory([.05,.05],x,self.zero,self.zero,self.zero,self.state,GROUPS,target,np.zeros(14),np.zeros(14),-np.ones(14),np.ones(14))
        self.assertTrue(r[5][0]);self.assertEqual(r[2][0],0.)
    def test_combined_cap_and_joint_bounds(self):
        rng=np.random.default_rng(270)
        for _ in range(100):
            ref=rng.uniform(-.5,.5,14);target=ref+rng.uniform(-.18,.18,14);x=rng.normal(size=55).astype(np.float32)
            r=controlled_memory([.05,.05],x,self.zero,self.zero,self.zero,rng.uniform(-1,1,14),GROUPS,target,ref,rng.uniform(-.5,.5,14),-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(r[0]-ref).max(),.18+1e-12);self.assertLessEqual(np.abs(r[0]).max(),1.)
    def test_invalid_inputs_rejected(self):
        for p in [[-.01,0],[.050001,0],[np.nan,0],[0,0,0]]:
            with self.assertRaises(ValueError):tracking_memory(p,self.zero,self.zero,self.zero,self.zero,self.state,GROUPS)
        with self.assertRaises(ValueError):tracking_memory([0,0],np.full(55,np.nan),self.zero,self.zero,self.zero,self.state,GROUPS)
        with self.assertRaises(ValueError):tracking_memory([0,0],self.zero,self.zero,self.zero,self.zero,np.full(14,1.01),GROUPS)

if __name__=='__main__':unittest.main()
