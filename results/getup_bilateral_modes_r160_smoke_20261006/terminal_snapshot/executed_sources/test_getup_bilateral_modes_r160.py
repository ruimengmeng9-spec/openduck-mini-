import inspect
import unittest
import numpy as np
from diagnostics import train_getup_bilateral_modes_r160 as run

IDS=np.array([0,1,2,9,10,11]);ZERO=np.zeros(55,dtype=np.float32)
class TestBilateral(unittest.TestCase):
    def call(self,p,current,initial=ZERO):return run.bilateral_feedback(p,current,ZERO,initial,ZERO,IDS)
    def test_initial_context_exact_zero(self):
        current=np.arange(55,dtype=np.float32)/100;extra,activation=self.call(np.ones(12),current,current)
        np.testing.assert_array_equal(extra,np.zeros(6));np.testing.assert_array_equal(activation,np.zeros(12))
    def test_actual_nominal_scalar_zero(self):
        current=np.arange(55,dtype=np.float32)/100
        extra,a=run.bilateral_feedback(np.ones(12),current,current,ZERO,ZERO,IDS)
        np.testing.assert_array_equal(extra,np.zeros(6));np.testing.assert_array_equal(a,np.zeros(12))
    def test_zero_coefficients(self):
        extra,_=self.call(np.zeros(12),np.arange(55,dtype=np.float32));np.testing.assert_array_equal(extra,np.zeros(6))
    def test_pure_common_same_side_response(self):
        current=ZERO.copy();current[6+IDS[[0,3]]]=.05;p=np.zeros(12);p[0]=.1
        extra,_=self.call(p,current);self.assertLess(extra[0],0.);self.assertEqual(extra[0],extra[3]);self.assertEqual(np.count_nonzero(extra),2)
    def test_pure_differential_opposite_response(self):
        current=ZERO.copy();current[6+IDS[0]]=-.05;current[6+IDS[3]]=.05;p=np.zeros(12);p[6]=.1
        extra,_=self.call(p,current);self.assertGreater(extra[0],0.);self.assertEqual(extra[0],-extra[3]);self.assertEqual(np.count_nonzero(extra),2)
    def test_right_sensor_can_drive_left(self):
        current=ZERO.copy();current[6+IDS[3]]=.05;p=np.zeros(12);p[0]=.1
        extra,_=self.call(p,current);self.assertNotEqual(extra[0],0.);self.assertEqual(extra[0],extra[3])
    def test_native_velocity_units(self):
        current=ZERO.copy();current[20+IDS[[0,3]]]=.05
        feature=run.bilateral_features(current,ZERO,IDS);self.assertAlmostEqual(feature[3],1.,places=7)
    def test_unused_sensors_do_not_change_feedback(self):
        current=ZERO.copy();current[:6]=1.;current[34:]=10.
        extra,a=self.call(np.ones(12),current);np.testing.assert_array_equal(extra,np.zeros(6));np.testing.assert_array_equal(a,np.zeros(12))
    def test_odd_change_about_zero_initial(self):
        current=ZERO.copy();current[6+IDS]=np.array([.01,-.02,.03,.04,-.03,.02]);current[20+IDS]=.01
        a,_=self.call(np.full(12,.1),current);b,_=self.call(np.full(12,.1),-current);np.testing.assert_array_equal(a,-b)
    def test_finite_shapes_ids_coefficient_limits(self):
        for p in [np.ones(11),np.full(12,1.01),np.full(12,np.nan)]:
            with self.assertRaises(ValueError):self.call(p,ZERO)
        with self.assertRaises(ValueError):run.bilateral_features(np.full(55,np.nan),ZERO,IDS)
        with self.assertRaises(ValueError):run.bilateral_features(ZERO,ZERO,np.array([0,0,2,9,10,11]))
    def test_feedback_and_total_correction_bounds(self):
        current=ZERO.copy();current[6+IDS]=100.;current[20+IDS]=100.;extra,_=self.call(np.ones(12),current)
        self.assertLessEqual(np.abs(extra).max(),.18)
        base=np.zeros(14);target=np.full(14,.17);lower=np.full(14,-2.);upper=-lower
        result=run.prior.old.local.merge_target(target,base,extra,IDS,lower,upper);self.assertLessEqual(np.abs(result).max(),.18)
        self.assertIs(run.prior.old.local.merge_target(target,base,np.zeros(6),IDS,lower,upper),target)
    def test_causal_interface(self):
        self.assertEqual(list(inspect.signature(run.bilateral_feedback).parameters),['parameters','current','nominal','initial','nominal_initial','ids'])

if __name__=='__main__':unittest.main()
