"""Pure checks for the lateral-mirror diagnostic wrapper."""

import unittest

import numpy as np

from diagnostics.backward_lateral_mirror import (
    AlternatingMirrorPolicy, LateralMirrorBlendPolicy,
)


class ConstantPolicy:
    def infer(self, obs):
        return np.asarray(obs[13:27], dtype=np.float32)


class MirrorBlendTest(unittest.TestCase):
    def test_zero_weight_matches_original(self):
        obs = np.arange(101, dtype=np.float32) / 101.0
        wrapped = LateralMirrorBlendPolicy(
            ConstantPolicy(), np.zeros(14, dtype=np.float32), 0.0
        )
        np.testing.assert_array_equal(wrapped.infer(obs), obs[13:27])

    def test_half_weight_is_left_right_equivariant(self):
        obs = np.arange(101, dtype=np.float32) / 101.0
        default = np.zeros(14, dtype=np.float32)
        wrapped = LateralMirrorBlendPolicy(
            ConstantPolicy(), default, 0.5
        )
        from diagnostics.test_turn_mirror_policy import mirror_joint, mirror_obs
        np.testing.assert_allclose(
            wrapped.infer(mirror_obs(obs, default, False)),
            mirror_joint(wrapped.infer(obs)), atol=1e-6,
        )

    def test_alternation_selects_full_gaits_outside_transitions(self):
        clock = [0.0]
        obs = np.arange(101, dtype=np.float32) / 101.0
        default = np.zeros(14, dtype=np.float32)
        wrapped = AlternatingMirrorPolicy(
            ConstantPolicy(), default, lambda: clock[0],
            period_s=2.0, original_fraction=0.5, transition_s=0.1,
        )
        clock[0] = 0.5
        np.testing.assert_array_equal(wrapped.infer(obs), obs[13:27])
        clock[0] = 1.5
        from diagnostics.test_turn_mirror_policy import mirror_joint, mirror_obs
        expected = mirror_joint(ConstantPolicy().infer(mirror_obs(obs, default, False)))
        np.testing.assert_array_equal(wrapped.infer(obs), expected)


if __name__ == "__main__":
    unittest.main()
