import unittest
import numpy as np
from diagnostics.train_getup_dynamic_gain_r147 import state_features,dynamic_gains


class DynamicGainTests(unittest.TestCase):
    def setUp(self):
        self.x=np.zeros(55,np.float32);self.ids=np.array([9,10,11]);self.base=np.array([.1,-.2,.3,.4,-.5,.6])
        self.parameters=np.r_[np.full(6,.5),np.full(12,.3)]
    def test_initial_action_exact(self):
        x=self.x.copy();x[:34]=np.arange(34)/100
        gains,a=dynamic_gains(self.base,self.parameters,x,self.x,x,self.x,self.ids)
        np.testing.assert_array_equal(gains,self.base);self.assertEqual(a,0.)
    def test_standard_scalar_exact(self):
        x=self.x.copy();x[:34]=.5
        gains,a=dynamic_gains(self.base,self.parameters,x,x,self.x,self.x,self.ids)
        np.testing.assert_array_equal(gains,self.base);self.assertEqual(a,0.)
    def test_zero_parameters_exact(self):
        x=self.x.copy();x[6+self.ids]=.1
        gains,a=dynamic_gains(self.base,np.zeros(18),x,self.x,self.x,self.x,self.ids)
        np.testing.assert_array_equal(gains,self.base);self.assertEqual(a,0.)
    def test_actual_current_state_changes(self):
        x=self.x.copy();x[6+self.ids]=.1
        gains,a=dynamic_gains(self.base,self.parameters,x,self.x,self.x,self.x,self.ids)
        self.assertTrue(a!=0.);self.assertTrue(np.any(gains!=self.base));self.assertLessEqual(np.abs(gains).max(),2.)
    def test_unused_fields_do_not_change_state(self):
        x=self.x.copy();x[34:]=7.
        np.testing.assert_array_equal(state_features(x,self.x,self.ids),state_features(self.x,self.x,self.ids))
    def test_shape_and_metadata_rejected(self):
        with self.assertRaises(ValueError):state_features(np.zeros(56),self.x,self.ids)
        with self.assertRaises(ValueError):dynamic_gains(self.base,np.zeros(19),self.x,self.x,self.x,self.x,self.ids)
    def test_parameter_and_sensor_bounds(self):
        bad=self.parameters.copy();bad[0]=1.01
        with self.assertRaises(ValueError):dynamic_gains(self.base,bad,self.x,self.x,self.x,self.x,self.ids)
        bad=self.x.copy();bad[0]=np.nan
        with self.assertRaises(ValueError):state_features(bad,self.x,self.ids)
    def test_velocity_scale(self):
        x=self.x.copy();x[20+self.ids]=.05
        np.testing.assert_array_equal(state_features(x,self.x,self.ids)[-3:],np.full(3,float(np.float32(.05))/.05))
    def test_no_inference_context_library(self):
        # Inference is a fixed equation: no fitted initial-state table, case,
        # seed, outcomes, directories or future values accepted by interface.
        import inspect
        self.assertEqual(tuple(inspect.signature(dynamic_gains).parameters),('base','parameters','current','nominal','initial','nominal_initial','ids'))


if __name__=='__main__':unittest.main()
