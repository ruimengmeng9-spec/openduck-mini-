"""Verify JAX-training and native-MuJoCo reference interpolation agree."""

import os
from unittest import mock
import unittest

import numpy as np

from diagnostics.reference_residual_policy import ContinuousDxReference
from playground.common.poly_reference_motion import PolyReferenceMotion as JaxReference
from playground.common.poly_reference_motion_numpy import PolyReferenceMotion as NumpyReference


class ReferenceParityTest(unittest.TestCase):
    def test_continuous_dx_matches_every_phase(self):
        path = "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
        jax_ref = JaxReference(path)
        numpy_ref = NumpyReference(path)
        max_error = 0.0
        max_joint_error = 0.0
        for dx in (-0.148, -0.12, -0.0925, -0.074, 0.0):
            adapter = ContinuousDxReference(numpy_ref, dx=dx, interpolate=True)
            with mock.patch.dict(os.environ, {
                "REFERENCE_DX": str(dx), "REFERENCE_DX_INTERPOLATION": "1"
            }):
                for phase in range(jax_ref.nb_steps_in_period):
                    expected = np.asarray(jax_ref.get_reference_motion(dx, 0.0, 0.0, phase))
                    actual = adapter.get_reference_motion(dx, 0.0, 0.0, phase)
                    max_error = max(max_error, float(np.max(np.abs(expected - actual))))
                    max_joint_error = max(max_joint_error, float(np.max(np.abs(expected[:16] - actual[:16]))))
        print("JAX/NUMPY REFERENCE MAX ERROR:", max_error, flush=True)
        print("JAX/NUMPY JOINT TARGET MAX ERROR RAD:", max_joint_error, flush=True)
        # Only the first 16 channels are decoded into joint targets. Derivative
        # channels are reported but are not consumed by the native decoder.
        self.assertLess(max_joint_error, 1e-3)


if __name__ == "__main__":
    unittest.main()
