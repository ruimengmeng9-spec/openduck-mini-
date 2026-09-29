"""Contract tests for R35 reset staging and exact exploration scale."""
import unittest
from unittest.mock import patch

import numpy as np

from diagnostics.getup_curriculum_exploration_r35 import effective_log_std
from diagnostics.getup_partial_curriculum_env_r35 import CurriculumEpisode, FRACTIONS
from diagnostics.getup_independent_native import POSES


class CurriculumContractTest(unittest.TestCase):
    def test_full_fall_reset_clears_only_previous_audit(self):
        class FakeSim:
            def __init__(self, new_floor_peak):
                self.peaks = {'floor': .02}
                self.new_floor_peak = new_floor_peak

            def clear_audit(self):
                self.peaks = {'floor': 0.}

            def prepare(self, pose, seed, perturb):
                self.assertion = self.peaks['floor'] == 0.
                self.peaks['floor'] = self.new_floor_peak

            def measure(self):
                return {'up_z': 0., 'torso_contact': True}

            def physical_valid(self):
                return self.peaks['floor'] < .01

        def make_episode(new_floor_peak):
            episode = CurriculumEpisode.__new__(CurriculumEpisode)
            episode.stage = 3
            episode.rng = np.random.default_rng(1)
            episode.sim = FakeSim(new_floor_peak)
            episode.phi = lambda measurement: 0.
            return episode

        with patch('diagnostics.getup_partial_curriculum_env_r35.state_hash',
                   return_value='test-hash'), patch(
                       'diagnostics.getup_partial_curriculum_env_r35.observation',
                       return_value=np.zeros(50, dtype=np.float32)):
            recovered = make_episode(0.)
            recovered.reset('supine')
            self.assertTrue(recovered.sim.assertion)
            self.assertTrue(recovered.initial_valid)
            invalid = make_episode(.02)
            with self.assertRaisesRegex(RuntimeError, 'physical audit'):
                invalid.reset('supine')
            self.assertTrue(invalid.sim.assertion)

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
