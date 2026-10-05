import inspect
import unittest
import numpy as np
from diagnostics.train_getup_jointwise_dynamic_r151 import jointwise_gains

class JointwiseTests(unittest.TestCase):
    def setUp(self):
        self.x=np.zeros(55,np.float32);self.ids=np.array([9,10,11]);self.base=np.array([.1,-.2,.3,.4,-.5,.6]);self.p=np.full(6,.2)
    def call(self,x,initial=None,p=None,base=None):
        return jointwise_gains(self.base if base is None else base,self.p if p is None else p,x,self.x,self.x if initial is None else initial,self.x,self.ids)
    def test_initial_exact(self):
        x=self.x.copy();x[:34]=.1
        g,a=self.call(x,x);np.testing.assert_array_equal(g,self.base);self.assertFalse(np.any(a))
    def test_nominal_exact(self):
        g,a=self.call(self.x);np.testing.assert_array_equal(g,self.base);self.assertFalse(np.any(a))
    def test_zero_coefficients_exact(self):
        x=self.x.copy();x[6+self.ids]=.2
        g,_=self.call(x,p=np.zeros(6));np.testing.assert_array_equal(g,self.base)
    def test_yaw_position_does_not_change_other_gains(self):
        x=self.x.copy();x[6+self.ids[0]]=.1
        g,a=self.call(x);self.assertNotEqual(g[0],self.base[0]);np.testing.assert_array_equal(g[1:],self.base[1:]);self.assertEqual(np.count_nonzero(a),1)
    def test_roll_velocity_does_not_change_position_or_other_velocity(self):
        x=self.x.copy();x[20+self.ids[1]]=.05
        g,a=self.call(x);self.assertNotEqual(g[4],self.base[4]);np.testing.assert_array_equal(g[[0,1,2,3,5]],self.base[[0,1,2,3,5]]);self.assertEqual(np.count_nonzero(a),1)
    def test_gyro_up_other_joints_and_future_history_irrelevant(self):
        x=self.x.copy();x[:6]=3;x[34:]=5;x[6:15]=2;x[20:29]=4
        g,a=self.call(x);np.testing.assert_array_equal(g,self.base);self.assertFalse(np.any(a))
    def test_velocity_units(self):
        x=self.x.copy();x[20+self.ids[2]]=.05
        _,a=self.call(x);self.assertEqual(a[5],np.tanh(float(np.float32(.05))/.05))
    def test_bounds(self):
        x=self.x.copy();x[6+self.ids]=10;x[20+self.ids]=10
        g,_=self.call(x,p=np.ones(6),base=np.full(6,2.));self.assertLessEqual(np.abs(g).max(),2.)
        with self.assertRaises(ValueError):self.call(x,p=np.full(6,1.001))
    def test_invalid_input(self):
        with self.assertRaises(ValueError):self.call(np.zeros(56))
        x=self.x.copy();x[0]=np.nan
        with self.assertRaises(ValueError):self.call(x)
        with self.assertRaises(ValueError):self.call(self.x,p=np.zeros(7))
    def test_fixed_causal_signature(self):
        self.assertEqual(tuple(inspect.signature(jointwise_gains).parameters),('base','parameters','current','nominal','initial','nominal_initial','ids'))

if __name__=='__main__':unittest.main()
