import unittest

import numpy as np

from diagnostics.backward_blend_policy import BlendedBackwardPolicy


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


if __name__ == "__main__":
    unittest.main()
