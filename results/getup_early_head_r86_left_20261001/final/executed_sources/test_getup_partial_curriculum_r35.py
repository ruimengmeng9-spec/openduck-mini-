"""Numerical checks for the diagnostic tilt-only reset generator."""
import unittest

import numpy as np

from diagnostics.getup_independent_native import POSES, canonical_quaternion
from diagnostics.probe_getup_partial_curriculum_r35 import fractional_quaternion


class PartialCurriculumTest(unittest.TestCase):
    def test_endpoints_preserve_home_and_canonical_fall(self):
        for pose in POSES:
            with self.subTest(pose=pose):
                np.testing.assert_allclose(fractional_quaternion(pose, 0.), [1., 0., 0., 0.])
                np.testing.assert_allclose(fractional_quaternion(pose, 1.), canonical_quaternion(pose))

    def test_halfway_is_unit_quaternion(self):
        for pose in POSES:
            with self.subTest(pose=pose):
                self.assertAlmostEqual(np.linalg.norm(fractional_quaternion(pose, .5)), 1.)

    def test_invalid_fraction_and_pose_rejected(self):
        for pose, fraction in [('prone', -.01), ('supine', 1.01), ('other', .5)]:
            with self.subTest(pose=pose, fraction=fraction):
                with self.assertRaises(ValueError):
                    fractional_quaternion(pose, fraction)


if __name__ == '__main__':
    unittest.main()
