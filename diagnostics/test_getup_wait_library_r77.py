import unittest
import numpy as np
from diagnostics.probe_getup_wait_library_r77 import change_wait


class WaitLibraryTests(unittest.TestCase):
    def setUp(self):
        self.ck = {'prefix_targets': np.array([[1., 2.], [3., 4.], [5., 6.], [5., 6.],
                                               [5., 6.], [7., 8.], [9., 10.]]),
                   'prefix_phases': np.array([0, 0, 1, 1, 1, 2, 3]),
                   'prefix_gains': np.arange(32).reshape(4, 8)}

    def test_identity_bit_exact_and_does_not_mutate(self):
        result = change_wait(self.ck, .06)
        np.testing.assert_array_equal(result['prefix_targets'], self.ck['prefix_targets'])
        np.testing.assert_array_equal(result['prefix_phases'], self.ck['prefix_phases'])
        self.assertIs(result['prefix_gains'], self.ck['prefix_gains'])

    def test_zero_wait_preserves_earlier_and_later_commands(self):
        result = change_wait(self.ck, 0.)
        np.testing.assert_array_equal(result['prefix_targets'], self.ck['prefix_targets'][[0, 1, 5, 6]])
        np.testing.assert_array_equal(result['prefix_phases'], [0, 0, 2, 3])
        self.assertEqual(self.ck['prefix_targets'].shape, (7, 2))

    def test_longer_wait_repeats_identical_target_only(self):
        result = change_wait(self.ck, .1)
        np.testing.assert_array_equal(result['prefix_targets'][:2], self.ck['prefix_targets'][:2])
        np.testing.assert_array_equal(result['prefix_targets'][2:7], np.tile([5., 6.], (5, 1)))
        np.testing.assert_array_equal(result['prefix_targets'][7:], self.ck['prefix_targets'][5:])

    def test_nonconstant_noncontiguous_and_invalid_durations_rejected(self):
        for value in (-.1, 5.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                change_wait(self.ck, value)
        bad = {**self.ck, 'prefix_targets': self.ck['prefix_targets'].copy()}
        bad['prefix_targets'][3, 0] += .01
        with self.assertRaises(ValueError):
            change_wait(bad, 2.)
        bad = {**self.ck, 'prefix_phases': np.array([0, 1, 0, 1, 1, 2, 3])}
        with self.assertRaises(ValueError):
            change_wait(bad, 2.)


if __name__ == '__main__':
    unittest.main()
