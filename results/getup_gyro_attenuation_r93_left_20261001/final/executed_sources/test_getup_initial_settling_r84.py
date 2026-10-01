import unittest
import numpy as np
from diagnostics.probe_getup_initial_settling_r84 import prepend,TRAIN,HELDOUT
from diagnostics.audit_getup_clock_selector_r83 import choose


class InitialSettlingTest(unittest.TestCase):
    def test_zero_identity(self):
        a=np.arange(42.).reshape(3,14);p=np.array([0,1,2])
        t,q=prepend(a,p,np.zeros(14),0)
        np.testing.assert_array_equal(t,a);np.testing.assert_array_equal(q,p)

    def test_prefix_preserves_every_original_command_and_phase(self):
        a=np.arange(42.).reshape(3,14);p=np.array([0,1,2]);home=np.ones(14)
        t,q=prepend(a,p,home,25)
        np.testing.assert_array_equal(t[:25],np.tile(home,(25,1)))
        np.testing.assert_array_equal(t[25:],a);np.testing.assert_array_equal(q[25:],p)
        np.testing.assert_array_equal(q[:25],np.full(25,8))

    def test_invalid_wait_rejected(self):
        for n in (-1,11,2.5):
            with self.assertRaises(ValueError):prepend(np.zeros((3,14)),np.zeros(3),np.zeros(14),n)

    def test_tie_prefers_baseline(self):
        self.assertEqual(choose(np.zeros(4),np.zeros((3,4)),np.ones((3,9)),3),0)

    def test_known_neighbor_arm(self):
        x=np.array([[0]*4,[1]*4,[2]*4],dtype=float)
        y=np.array([[0,1],[1,0],[1,0]],dtype=bool)
        self.assertEqual(choose(np.zeros(4),x,y,1),1)

    def test_disjoint_splits(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT))
        self.assertFalse(set(range(781000,783040))&set(HELDOUT))


if __name__=='__main__':unittest.main()
