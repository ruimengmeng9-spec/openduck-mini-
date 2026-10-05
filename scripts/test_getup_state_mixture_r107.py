import unittest
import numpy as np
from diagnostics.train_getup_state_mixture_r107 import scale_for


class MixtureTest(unittest.TestCase):
    def test_zero_and_one_profiles(self):
        obs=np.zeros(55);nominal=obs.copy();obs[50:54]=[10,-10,1,-1]
        self.assertEqual(scale_for(np.zeros(5),obs,nominal),0.)
        self.assertEqual(scale_for([1,0,0,0,0],obs,nominal),1.)

    def test_actual_deviation_is_used(self):
        nominal=np.ones(55);obs=nominal.copy();obs[50]+=.1
        self.assertAlmostEqual(scale_for([.5,2,0,0,0],obs,nominal),.7)
        self.assertAlmostEqual(scale_for([.5,2,0,0,0],nominal,nominal),.5)

    def test_clipped_profile_convexity(self):
        obs=np.zeros(55);obs[50]=1
        self.assertEqual(scale_for([.5,8,0,0,0],obs,np.zeros(55)),1.)
        self.assertEqual(scale_for([.5,-8,0,0,0],obs,np.zeros(55)),0.)

    def test_other_inputs_and_seed_not_used(self):
        obs=np.zeros(55);nominal=obs.copy();obs[:50]=10;obs[54]=1
        self.assertEqual(scale_for([.4,1,1,1,1],obs,nominal),.4)

    def test_bad_values_rejected(self):
        for p in ([0,0],[-1,0,0,0,0],[1,9,0,0,0],[0,float('nan'),0,0,0]):
            with self.assertRaises(ValueError):scale_for(p,np.zeros(55),np.zeros(55))
        with self.assertRaises(ValueError):scale_for(np.zeros(5),np.full(55,np.nan),np.zeros(55))


if __name__=='__main__':unittest.main()
