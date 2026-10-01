"""Decoder contract tests; no robot, GPU, or model weights are required."""

from types import SimpleNamespace
import unittest

import numpy as np

from diagnostics.reference_residual_policy import ReferenceResidualPolicy


class FakeReference:
    nb_steps_in_period = 27
    value = 0.4

    def get_reference_motion(self, vx, vy, yaw, phase):
        return np.full(16, self.value, dtype=np.float32)


class ConstantActor:
    def infer(self, obs):
        return np.ones(14, dtype=np.float32)


class ReferenceResidualTest(unittest.TestCase):
    def setUp(self):
        self.sim = SimpleNamespace(
            data=SimpleNamespace(time=3.0),
            model=SimpleNamespace(
                actuator_trnid=np.column_stack([np.arange(14), np.zeros(14)]).astype(int),
                jnt_range=np.column_stack([-np.full(14, 0.45), np.full(14, 0.45)]),
            ),
            default_actuator=np.zeros(14), action_scale=0.25,
            PRM=FakeReference(),
        )
        self.obs = np.zeros(101, dtype=np.float32)
        self.obs[99] = 1.0

    def test_ramp_starts_from_home_with_small_residual(self):
        policy = ReferenceResidualPolicy(ConstantActor(), self.sim, 0.08, 1.0)
        np.testing.assert_allclose(policy.infer(self.obs), 0.08 / 0.25)

    def test_full_reference_is_not_actor_clipped_but_joint_clamped(self):
        policy = ReferenceResidualPolicy(ConstantActor(), self.sim, 0.08, 1.0)
        self.sim.data.time = 4.5
        np.testing.assert_allclose(policy.infer(self.obs), 0.45 / 0.25)

    def test_rejects_excessive_residual_gain(self):
        with self.assertRaises(ValueError):
            ReferenceResidualPolicy(ConstantActor(), self.sim, 0.3, 1.0)

    def test_feedback_retains_authority_at_unreachable_reference(self):
        class NegativeActor:
            def infer(self, obs):
                return -np.ones(14, dtype=np.float32)
        self.sim.PRM.value = 0.8
        policy = ReferenceResidualPolicy(NegativeActor(), self.sim, 0.08, 0.0)
        np.testing.assert_allclose(policy.infer(self.obs), (0.45 - 0.08) / 0.25)


if __name__ == "__main__":
    unittest.main()
