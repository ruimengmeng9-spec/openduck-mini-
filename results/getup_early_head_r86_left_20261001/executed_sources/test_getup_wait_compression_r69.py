import unittest
import numpy as np

from diagnostics.probe_getup_wait_compression_r69 import compress
from diagnostics.getup_independent_native import DT


class WaitCompressionTests(unittest.TestCase):
    def setUp(self):
        self.home = np.zeros(14)
        self.recovery = np.full((5, 14), .2)
        self.pulse = np.full((21, 14), -.3)
        self.original = np.concatenate([self.recovery, np.zeros((1550, 14)), self.pulse])

    def test_only_identical_wait_is_removed(self):
        result, phases = compress(self.original, 5, self.home, 2.)
        self.assertEqual(len(result), 5 + round(2. / DT) + 21)
        np.testing.assert_array_equal(result[:5], self.recovery)
        np.testing.assert_array_equal(result[-21:], self.pulse)
        np.testing.assert_array_equal(phases[:5], np.zeros(5))
        self.assertEqual(phases[-1], 3)

    def test_nonconstant_wait_is_rejected(self):
        broken = self.original.copy()
        broken[100, 0] = .001
        with self.assertRaises(ValueError):
            compress(broken, 5, self.home, 2.)

    def test_zero_wait_keeps_both_motion_segments(self):
        result, phases = compress(self.original, 5, self.home, 0.)
        np.testing.assert_array_equal(result, np.concatenate([self.recovery, self.pulse]))
        self.assertFalse(np.any(phases == 1))


if __name__ == '__main__':
    unittest.main()
