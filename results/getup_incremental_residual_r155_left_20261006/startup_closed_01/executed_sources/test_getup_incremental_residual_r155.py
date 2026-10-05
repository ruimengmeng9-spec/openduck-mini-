import inspect
import unittest
import numpy as np
from diagnostics.train_getup_incremental_residual_r155 import incremental_feedback
from diagnostics.train_getup_local_hip_r130 import merge_target

class IncrementalTests(unittest.TestCase):
    def setUp(self):
        self.zero=np.zeros(55,np.float32);self.ids=np.array([9,10,11]);self.p=np.full(6,.1)
    def call(self,x,initial=None,p=None):
        return incremental_feedback(self.p if p is None else p,x,self.zero,self.zero if initial is None else initial,self.zero,self.ids)
    def test_initial_exact(self):
        x=self.zero.copy();x[:34]=.1;f,a=self.call(x,x);self.assertFalse(np.any(f));self.assertFalse(np.any(a))
    def test_nominal_scalar_exact(self):
        for k in range(25):
            x=np.full(55,k/100,np.float32);f,a=incremental_feedback(self.p,x,x,x,x,self.ids)
            self.assertFalse(np.any(f));self.assertFalse(np.any(a))
    def test_zero_parameters(self):
        x=self.zero.copy();x[15:18]=1;f,_=self.call(x,p=np.zeros(6));np.testing.assert_array_equal(f,np.zeros(3))
    def test_odd_position_response(self):
        x=self.zero.copy();x[15]=.05;positive,_=self.call(x);negative,_=self.call(-x)
        np.testing.assert_array_equal(negative,-positive);self.assertLess(positive[0],0)
    def test_joint_independence(self):
        x=self.zero.copy();x[16]=.1;f,a=self.call(x);self.assertEqual(np.count_nonzero(f),1);self.assertEqual(np.count_nonzero(a),1)
    def test_velocity_scale(self):
        x=self.zero.copy();x[31]=.05;f,a=self.call(x)
        self.assertEqual(a[5],np.tanh(float(np.float32(.05))/.05));self.assertLess(f[2],0)
    def test_unused_channels_irrelevant(self):
        x=self.zero.copy();x[:6]=3;x[34:]=4;x[6:15]=2;x[20:29]=3;f,a=self.call(x)
        self.assertFalse(np.any(f));self.assertFalse(np.any(a))
    def test_nonzero_initial_change_only(self):
        i=self.zero.copy();i[15]=.2;x=i.copy();x[15]+=.01;_,a=self.call(x,i)
        self.assertEqual(a[0],np.tanh(float(np.float32(x[15]-i[15]))/.05))
    def test_finite_and_bounds(self):
        with self.assertRaises(ValueError):self.call(np.zeros(56))
        x=self.zero.copy();x[0]=np.nan
        with self.assertRaises(ValueError):self.call(x)
        with self.assertRaises(ValueError):self.call(self.zero,p=np.full(6,1.001))
        with self.assertRaises(ValueError):self.call(self.zero,p=np.zeros(7))
    def test_residual_and_total_target_bounds(self):
        x=self.zero.copy();x[15:18]=10;x[29:32]=10;f,_=self.call(x,p=np.ones(6));self.assertLessEqual(np.abs(f).max(),.18)
        target=np.full(14,.17);base=np.zeros(14);out=merge_target(target,base,np.full(3,.18),self.ids,np.full(14,-2),np.full(14,2))
        self.assertLessEqual(np.abs(out[self.ids]).max(),.18);np.testing.assert_array_equal(out[:9],target[:9])
        self.assertIs(merge_target(target,base,np.zeros(3),self.ids,np.full(14,-2),np.full(14,2)),target)
    def test_causal_signature(self):
        self.assertEqual(tuple(inspect.signature(incremental_feedback).parameters),('parameters','current','nominal','initial','nominal_initial','ids'))

if __name__=='__main__':unittest.main()
