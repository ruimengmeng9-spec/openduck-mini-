import unittest
import numpy as np
from diagnostics.probe_residual_history_timing import LaggedHistoryActor


class EchoActor:
    def infer(self, obs):
        return np.asarray(obs).copy()


class HistoryTimingTest(unittest.TestCase):
    def test_delays_only_history_and_does_not_mutate_input(self):
        shim = LaggedHistoryActor(EchoActor())
        first = np.arange(101, dtype=np.float32)
        original = first.copy()
        shim.infer(first)
        second = first + 100
        output = shim.infer(second)
        np.testing.assert_array_equal(output[41:83], original[41:83])
        np.testing.assert_array_equal(output[:41], second[:41])
        np.testing.assert_array_equal(output[83:], second[83:])
        np.testing.assert_array_equal(first, original)


if __name__ == "__main__":
    unittest.main()
