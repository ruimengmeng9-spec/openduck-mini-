"""Pure adapter checks for the backward temporal-mirror diagnostic."""

import unittest

import numpy as np

from diagnostics.test_backward_temporal_mirror import TemporalMirrorPolicy


class RecordingPolicy:
    def __init__(self):
        self.last_obs = None

    def infer(self, obs):
        self.last_obs = np.asarray(obs).copy()
        return np.ones(14, dtype=np.float32)


class TemporalMirrorPolicyTest(unittest.TestCase):
    def test_negative_command_phase_and_ankle_flip(self):
        base = RecordingPolicy()
        adapter = TemporalMirrorPolicy(
            base, "command_flip_phase_reverse_flip_ankle", 1.0
        )
        obs = np.zeros(101, dtype=np.float32)
        obs[6] = -0.074
        obs[100] = 0.5
        action = adapter.infer(obs)
        self.assertAlmostEqual(float(base.last_obs[6]), 0.074)
        self.assertAlmostEqual(float(base.last_obs[100]), -0.5)
        np.testing.assert_allclose(action[[4, 13]], [-1.0, -1.0])
        self.assertEqual(float(action[3]), 1.0)
        self.assertAlmostEqual(float(obs[6]), -0.074, places=6)

    def test_nonnegative_command_passes_through(self):
        base = RecordingPolicy()
        adapter = TemporalMirrorPolicy(base, "command_flip_flip_all")
        obs = np.zeros(101, dtype=np.float32)
        obs[6] = 0.074
        np.testing.assert_allclose(adapter.infer(obs), np.ones(14))
        np.testing.assert_allclose(base.last_obs, obs)

    def test_pitch_guard_reduces_flip_strength(self):
        base = RecordingPolicy()
        pitch = [-10.0]
        adapter = TemporalMirrorPolicy(
            base, "command_flip_flip_ankle", 0.4,
            lambda: pitch[0], guard_start_deg=-5.0,
            guard_full_deg=-15.0, guard_strength=0.3,
        )
        obs = np.zeros(101, dtype=np.float32)
        obs[6] = -0.074
        self.assertAlmostEqual(float(adapter.infer(obs)[4]), 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
