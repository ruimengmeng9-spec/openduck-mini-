import unittest
import numpy as np
from diagnostics.probe_getup_head_feedback_r128 import sensor_feedback


class SensorHeadFeedback(unittest.TestCase):
    def test_nominal_exact_zero(self):
        x=np.arange(55,dtype=np.float32)/10
        for g in (0.,1.,4.):np.testing.assert_array_equal(sensor_feedback(x,x,[6,7,8,9],g),np.zeros(4))

    def test_bounds_and_only_selected_sensor_joints(self):
        x=np.zeros(55,dtype=np.float32);x[12:16]=[1.,-1.,.01,-.01]
        np.testing.assert_allclose(sensor_feedback(x,np.zeros(55),[6,7,8,9],4.),[-.18,.18,-.04,.04],atol=1e-8)
        x[0:6]=1000.;x[20:]=1000.
        np.testing.assert_allclose(sensor_feedback(x,np.zeros(55),[6,7,8,9],4.),[-.18,.18,-.04,.04],atol=1e-8)

    def test_reject_invalid_sensor_or_gain(self):
        with self.assertRaises(ValueError):sensor_feedback(np.zeros(56),np.zeros(55),[6,7,8,9],1.)
        with self.assertRaises(ValueError):sensor_feedback(np.zeros(55),np.zeros(55),[6,7,8,9],8.)


if __name__=='__main__':unittest.main()
