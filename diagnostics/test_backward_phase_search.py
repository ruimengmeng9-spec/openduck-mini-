import unittest
import numpy as np
from diagnostics.backward_phase_search import phase_delta, balance_delta, balance_features, body_pitch, phase_pitch


class PhaseCorrectionTest(unittest.TestCase):
    def test_zero_identity_and_ramp(self):
        np.testing.assert_array_equal(phase_delta(np.zeros(18),[1,0],1.),np.zeros(14))
        np.testing.assert_array_equal(phase_delta(np.ones(18),[1,0],0.),np.zeros(14))

    def test_pitch_has_opposite_ankle_compensation(self):
        weights=np.zeros((6,3))
        weights[2,0]=np.arctanh(.5)
        result=phase_delta(weights,[1,0],1.)
        self.assertAlmostEqual(result[2],.03)
        self.assertAlmostEqual(result[4],-.03)
        self.assertEqual(np.count_nonzero(result),2)

    def test_independent_legs_and_bounded_feedback(self):
        result=phase_delta(np.zeros(18),[1,0],1.,[100,100,100,0,0,0],np.pi/2)
        np.testing.assert_allclose(result[[0,1,2]],[.06,.035,.06])
        np.testing.assert_array_equal(result[9:],np.zeros(5))
        self.assertAlmostEqual(result[4],-.06)

    def test_balance_deadband_and_compensation(self):
        np.testing.assert_array_equal(balance_delta([1,0],.09,0,1),np.zeros(14))
        delta=balance_delta([1,.2],-.3,-1,1)
        self.assertLess(delta[2],0)
        self.assertEqual(delta[2],delta[11])
        self.assertEqual(delta[4],-delta[2])
        self.assertEqual(delta[13],-delta[11])
        self.assertLessEqual(max(abs(delta)),.06)
        np.testing.assert_array_equal(balance_delta([1,1],.4,4,0),np.zeros(14))

    def test_pitch_and_rate_feature_units(self):
        q=[0,0,0,np.cos(-.15),0,np.sin(-.15),0]
        self.assertAlmostEqual(body_pitch(q),-.3)
        np.testing.assert_allclose(balance_features(-.3,-8),[-1,-5])

    def test_phase_pitch_template_and_configured_deadband(self):
        self.assertAlmostEqual(phase_pitch([.1,.2,0,0,0,0,0],[1,0]),.3)
        self.assertAlmostEqual(phase_pitch([.1,.2,0,0,0,0,0],[-1,0]),-.1)
        np.testing.assert_allclose(balance_features(.1,0,.04),[.3,0])


if __name__=='__main__':
    unittest.main()
