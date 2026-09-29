"""Contract tests for R35 reset staging and exact exploration scale."""
import unittest

import numpy as np

from diagnostics.getup_curriculum_exploration_r35 import effective_log_std
from diagnostics.getup_partial_curriculum_env_r35 import FRACTIONS
from diagnostics.getup_independent_native import POSES


class CurriculumContractTest(unittest.TestCase):
    def test_exploration_shape_and_scale(self):
        obs = np.zeros((4, 50), dtype=np.float32)
        obs[:, 5] = [1., .5, 0., -.5]
        log_std = effective_log_std(obs)
        self.assertEqual(log_std.shape, (4, 14))
        np.testing.assert_allclose(np.exp(log_std[:, 0]), [.02, .02, .26, .5], rtol=1e-6)
        self.assertTrue(np.isfinite(log_std).all())

    def test_bad_observation_rejected(self):
        with self.assertRaises(ValueError):
            effective_log_std(np.zeros((1, 49)))

    def test_curriculum_covers_every_pose_and_advances(self):
        for stage in (0, 1, 2):
            self.assertEqual(set(FRACTIONS[stage]), set(POSES))
            for pose in POSES:
                self.assertEqual(len(FRACTIONS[stage][pose]), 3)
                self.assertTrue(all(0. < f < 1. for f in FRACTIONS[stage][pose]))
                if stage:
                    self.assertGreaterEqual(max(FRACTIONS[stage][pose]),
                                            max(FRACTIONS[stage - 1][pose]))


if __name__ == '__main__':
    unittest.main()
