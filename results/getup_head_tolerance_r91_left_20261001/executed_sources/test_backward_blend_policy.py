import unittest

import numpy as np

from diagnostics.backward_blend_policy import (
    BlendedBackwardPolicy, PitchGuardBackwardPolicy,
)


class ConstantPolicy:
    def __init__(self, value):
        self.value = value

    def infer(self, obs):
        return np.full(14, self.value, dtype=np.float32)


class BackwardBlendPolicyTest(unittest.TestCase):
    def test_reverse_blend_and_nonreverse_passthrough(self):
        policy = BlendedBackwardPolicy(ConstantPolicy(0.0), ConstantPolicy(1.0), 0.25)
        obs = np.zeros(101, dtype=np.float32)
        obs[6] = -0.074
        np.testing.assert_allclose(policy.infer(obs), np.full(14, 0.25))
        obs[6] = 0.074
        np.testing.assert_allclose(policy.infer(obs), np.zeros(14))

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            BlendedBackwardPolicy(ConstantPolicy(0.0), ConstantPolicy(1.0), 1.1)
        policy = BlendedBackwardPolicy(ConstantPolicy(0.0), ConstantPolicy(1.0), 0.5)
        with self.assertRaises(ValueError):
            policy.infer(np.zeros(100, dtype=np.float32))

    def test_pitch_guard_interpolates_without_retaining_previous_weight(self):
        pitch = [-12.0]
        policy = PitchGuardBackwardPolicy(
            ConstantPolicy(0.0), ConstantPolicy(1.0), lambda: pitch[0],
            full_reverse_deg=-12.0, full_guard_deg=-18.0,
            guard_reverse_weight=0.9,
        )
        obs = np.zeros(101, dtype=np.float32)
        obs[6] = -0.074
        for value, expected in [(-12.0, 1.0), (-15.0, 0.95), (-18.0, 0.9),
                                (-12.0, 1.0)]:
            pitch[0] = value
            np.testing.assert_allclose(policy.infer(obs), np.full(14, expected), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
