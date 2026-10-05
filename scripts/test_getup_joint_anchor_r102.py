import unittest
import numpy as np
from diagnostics.train_getup_joint_anchor_r102 import action_for,rank
from diagnostics.getup_reference_env_r100 import numpy_action


class AnchorTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(9);self.obs=rng.normal(size=55).astype(np.float32)
        self.weights={}
        for name,a,b in [('hidden0',55,64),('hidden1',64,32),('mean',32,10)]:
            self.weights[name+'_kernel']=(rng.normal(size=(a,b))*.1).astype(np.float32)
            self.weights[name+'_bias']=np.zeros(b,np.float32)
        self.anchor=numpy_action(self.weights,self.obs)

    def test_nominal_exact_zero(self):
        np.testing.assert_array_equal(action_for(self.weights,self.obs,self.anchor,np.linspace(-4,4,10)),np.zeros(10))

    def test_zero_gain_exact_zero(self):
        np.testing.assert_array_equal(action_for(self.weights,self.obs,self.anchor+1,np.zeros(10)),np.zeros(10))

    def test_bounds(self):
        self.assertLessEqual(np.max(np.abs(action_for(self.weights,self.obs,-self.anchor,np.ones(10)*4))),1.)

    def test_invalid_gains(self):
        for gains in (np.ones(9),np.ones(10)*4.1,np.ones(10)*np.nan):
            with self.assertRaises(ValueError):action_for(self.weights,self.obs,self.anchor,gains)

    def test_nominal_preservation_first(self):
        good=dict(nominal_success=True,successes=12,physical_failures=0,return_sum=0)
        bad=dict(nominal_success=False,successes=24,physical_failures=0,return_sum=100)
        self.assertGreater(rank(good),rank(bad))

    def test_success_before_return(self):
        good=dict(nominal_success=True,successes=13,physical_failures=0,return_sum=0)
        bad=dict(nominal_success=True,successes=12,physical_failures=0,return_sum=1000)
        self.assertGreater(rank(good),rank(bad))


if __name__=='__main__':unittest.main()
