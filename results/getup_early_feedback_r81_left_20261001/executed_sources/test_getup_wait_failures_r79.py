import unittest
import numpy as np
from diagnostics.audit_getup_wait_failures_r79 import commands, first_run, longest_run, up_z


class OfflineFailureAuditTests(unittest.TestCase):
    def test_orientation_projection(self):
        qpos = np.zeros((3, 7))
        qpos[:, 3:7] = [[1., 0., 0., 0.], [2 ** -.5, 2 ** -.5, 0., 0.], [0., 1., 0., 0.]]
        np.testing.assert_allclose(up_z(qpos), [1., 0., -1.], atol=1e-14)

    def test_continuous_not_cumulative_runs(self):
        self.assertEqual(longest_run([1, 1, 0, 1, 1, 1, 0]), 3)
        self.assertEqual(first_run([0, 1, 1, 0, 1, 1, 1], 3), 4)
        self.assertIsNone(first_run([1, 0, 1], 2))

    def test_original_slew_and_joint_clipping(self):
        targets = np.array([[1., -1.], [1., -1.]])
        desired, applied = commands(targets, np.array([[.18], [.18]]), [0],
                                    np.zeros(2), np.array([-.5, -.5]), np.array([.5, .5]))
        np.testing.assert_allclose(desired, [[.5, -.5], [.5, -.5]])
        np.testing.assert_allclose(applied, [[.1048, -.1048], [.2096, -.2096]])
        np.testing.assert_array_equal(targets, [[1., -1.], [1., -1.]])


if __name__ == '__main__':
    unittest.main()
