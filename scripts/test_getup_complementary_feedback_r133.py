import unittest
import numpy as np
from diagnostics.probe_getup_complementary_feedback_r133 import local


class ComplementaryTests(unittest.TestCase):
    def test_global_program_input_contract(self):
        obs=np.zeros(55,np.float32)
        np.testing.assert_array_equal(local.local_feedback(obs,obs,[9,10,11],np.ones(6)),np.zeros(3))
        with self.assertRaises(ValueError):local.local_feedback(np.zeros(56),obs,[9,10,11],np.ones(6))

    def test_zero_exec_identity(self):
        target=np.zeros(14)
        self.assertIs(local.merge_target(target,target,np.zeros(3),[9,10,11],target-2,target+2),target)


if __name__=='__main__':unittest.main()
