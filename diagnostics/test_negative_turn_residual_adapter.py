"""Offline contract checks for the experimental residual adapter."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import types

import numpy as np

try:
    import onnxruntime  # noqa: F401
except ModuleNotFoundError:
    # The fake session below lets this contract test run on a workstation
    # without installing the full inference runtime.
    stub = types.ModuleType("onnxruntime")
    stub.InferenceSession = lambda *args, **kwargs: None
    sys.modules["onnxruntime"] = stub

from diagnostics.negative_turn_residual import ResidualNegativePolicy


class FakeBase:
    def infer(self, obs: np.ndarray) -> np.ndarray:
        return np.full(14, 0.2, dtype=np.float32)


class FakeSession:
    calls = 0

    def get_inputs(self):
        return [type("Input", (), {"name": "features"})()]

    def run(self, _outputs, feed):
        self.calls += 1
        assert feed["features"].shape == (1, 115)
        return [np.full((1, 14), 0.1, dtype=np.float32)]


class ResidualAdapterTest(unittest.TestCase):
    def setUp(self):
        self.session = FakeSession()
        patcher = patch(
            "diagnostics.negative_turn_residual.ort.InferenceSession",
            return_value=self.session,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.policy = ResidualNegativePolicy(FakeBase(), Path("unused.onnx"), "extended")

    def observation(self, yaw: float) -> np.ndarray:
        obs = np.zeros(101, dtype=np.float32)
        obs[8] = yaw
        return obs

    def test_nonnegative_commands_leave_base_untouched(self):
        action = self.policy.infer(self.observation(0.15))
        np.testing.assert_array_equal(action, np.full(14, 0.2, dtype=np.float32))
        self.assertEqual(self.session.calls, 0)

    def test_extended_gate_and_output_contract(self):
        self.assertAlmostEqual(self.policy._weight(-0.08), 0.0)
        self.assertAlmostEqual(self.policy._weight(-0.09), 0.5)
        self.assertAlmostEqual(self.policy._weight(-0.10), 1.0)
        self.assertAlmostEqual(self.policy._weight(-0.17), 1.0)
        self.assertAlmostEqual(self.policy._weight(-0.20), 0.0)
        action = self.policy.infer(self.observation(-0.15))
        np.testing.assert_allclose(action, np.full(14, 0.3, dtype=np.float32))
        self.assertEqual(self.session.calls, 1)

    def test_rejects_nonfinite_observation(self):
        obs = self.observation(-0.15)
        obs[10] = np.nan
        with self.assertRaises(ValueError):
            self.policy.infer(obs)
        self.assertEqual(self.session.calls, 0)


if __name__ == "__main__":
    unittest.main()
