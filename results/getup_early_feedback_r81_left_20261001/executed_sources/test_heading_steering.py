import unittest
import numpy as np
from playground.open_duck_mini_v2.heading_steering import yaw_and_features, steering_delta


class SteeringTest(unittest.TestCase):
    def test_features_are_continuous_across_yaw_wrap(self):
        q = np.array([0,0,.15, 0,0,0,1.])
        a = yaw_and_features(q, -np.pi+.001, .5, 1)
        b = yaw_and_features(q, np.pi+.001, .5, 1)
        np.testing.assert_allclose(a, b, atol=1e-7)
        np.testing.assert_allclose(a[2:5], [0,0,1])
        self.assertEqual(a[5], .5)

    def test_only_four_hips_can_change(self):
        delta = steering_delta(np.array([10.,-10.]), .05, .025)
        np.testing.assert_allclose(delta[[0,9]], .05)
        np.testing.assert_allclose(delta[[1,10]], -.025)
        self.assertEqual(np.count_nonzero(delta), 4)
        np.testing.assert_array_equal(steering_delta(np.zeros(2), .05, .025), np.zeros(14))


if __name__ == "__main__":
    unittest.main()
