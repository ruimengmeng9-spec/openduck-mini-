import unittest
import numpy as np
from diagnostics.fit_getup_wait_selector_r78 import cross_validate, predict


class WaitSelectorTests(unittest.TestCase):
    def test_only_sensor_distance_selects_wait(self):
        x = np.array([[0., 0.], [2., 2.], [2.1, 2.1]])
        y = np.array([[1., 0.], [0., 1.], [0., 1.]])
        selected, _ = predict(np.array([2.05, 2.05]), x, y, [-5., -5.], 1, 0.)
        self.assertEqual(selected, 1)

    def test_nominal_and_ties_use_baseline(self):
        x, y = np.array([[1.], [2.]]), np.ones((2, 3))
        selected, _ = predict(np.array([-1.]), x, y, [-1.], 1, 0.)
        self.assertEqual(selected, 0)

    def test_margin_prevents_unsupported_change(self):
        selected, _ = predict(np.array([1.]), np.array([[1.], [1.1]]),
                              np.array([[0., 1.], [1., 1.]]), [-5.], 2, .5)
        self.assertEqual(selected, 0)

    def test_cross_validation_excludes_matching_training_row(self):
        x, y = np.array([[0.], [1.]]), np.array([[0., 1.], [1., 0.]])
        selected, passed = cross_validate(x, y, [-10.], 1, 0.)
        np.testing.assert_array_equal(selected, [0, 1])
        self.assertEqual(passed, 0)

    def test_nonfinite_sensor_rejected(self):
        with self.assertRaises(ValueError):
            predict([float('nan')], np.array([[1.]]), np.ones((1, 2)), [0.], 1, 0.)


if __name__ == '__main__':
    unittest.main()
