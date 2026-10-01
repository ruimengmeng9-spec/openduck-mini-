import unittest
from types import SimpleNamespace
import numpy as np
from diagnostics.bridge_getup_contact_prefixes_r31 import teacher_targets
from diagnostics.search_getup_rescue_r22 import target_at


class BridgeTests(unittest.TestCase):
    def test_teacher_sampling_is_exact_50hz_and_ends_home(self):
        sim=SimpleNamespace(home=np.zeros(14),lower=np.full(14,-1.57),upper=np.full(14,1.57))
        teacher=np.full((2,8),.1)
        q,d=teacher_targets(sim,teacher)
        self.assertEqual(q.shape,(61,14))
        np.testing.assert_array_equal(d[:60],np.full(60,.02))
        for i in (0,10,29,45,59):
            np.testing.assert_array_equal(q[i],target_at(sim,teacher,i*.02))
        np.testing.assert_array_equal(q[-1],sim.home)


if __name__=='__main__':
    unittest.main()
