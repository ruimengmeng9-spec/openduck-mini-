import unittest
from unittest.mock import patch
import numpy as np
from diagnostics.train_getup_gyro_attenuation_r93 import modified_gains,run,TRAIN,HELDOUT,BIAS


class GyroAttenuationTest(unittest.TestCase):
    def test_only_gyro_columns_change(self):
        gains=np.arange(8,dtype=float);changed=modified_gains(gains,.5,.25,2.)
        np.testing.assert_array_equal(changed[[0,1,2,6]],gains[[0,1,2,6]])
        np.testing.assert_array_equal(changed[3:6],gains[3:6]*.5)
        self.assertEqual(changed[7],gains[7]*.25)
        np.testing.assert_array_equal(gains,np.arange(8))

    def test_inactive_window_identity(self):
        for elapsed in (0.,.5,3.5,4.):
            np.testing.assert_array_equal(modified_gains(np.arange(8.),0.,0.,elapsed),np.arange(8.))

    def test_identity_literal_delegation(self):
        with patch('diagnostics.train_getup_gyro_attenuation_r93.rollout',return_value=('base',[])) as fn:
            self.assertEqual(run(None,None,None,None,None,None,None,None,(1.,1.),(0.,0.),True),('base',[]))
            fn.assert_called_once()

    def test_invalid_amplification_rejected(self):
        for p,r in ((1.1,0.),(0.,-.1),(np.nan,0.)):
            with self.assertRaises(ValueError):modified_gains(np.ones(8),p,r,2.)

    def test_split_and_noise_budget(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT))
        for old in (787000,789000,790000):self.assertFalse(set(range(old,old+40))&set(HELDOUT))
        self.assertEqual(max(abs(x) for b in BIAS for x in b),.0005)


if __name__=='__main__':unittest.main()
