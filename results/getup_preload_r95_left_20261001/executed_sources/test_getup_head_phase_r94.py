import unittest
import numpy as np
from diagnostics.probe_getup_head_phase_r94 import warped_head_prefix,LEVELS,TRAIN,HELDOUT,WINDOW,DT


class HeadPhaseTest(unittest.TestCase):
    def setUp(self):
        self.targets=np.tile(np.arange(120.)[:,None],(1,14))/100
        self.ck={'prefix_targets':self.targets,'prefix_phases':np.zeros(120,dtype=int)}

    def test_identity_exact_and_no_mutation(self):
        result=warped_head_prefix(self.ck,[5,6],[0.,0.])
        np.testing.assert_array_equal(result['prefix_targets'],self.targets)
        self.assertIsNot(result['prefix_targets'],self.targets)

    def test_only_head_and_early_targets_change(self):
        result=warped_head_prefix(self.ck,[5,6],[.12,-.12])['prefix_targets']
        others=[i for i in range(14) if i not in (5,6)]
        np.testing.assert_array_equal(result[:,others],self.targets[:,others])
        np.testing.assert_array_equal(result[60:],self.targets[60:])
        np.testing.assert_array_equal(result[0],self.targets[0])
        self.assertGreater(result[30,5],self.targets[30,5])
        self.assertLess(result[30,6],self.targets[30,6])
        np.testing.assert_array_equal(self.ck['prefix_targets'],self.targets)

    def test_clock_monotonic_and_bounded(self):
        elapsed=np.arange(61)*DT;weight=np.interp(elapsed,WINDOW,(0,1,1,0))
        for shift in LEVELS:
            cursor=elapsed+shift*weight
            self.assertTrue((np.diff(cursor)>0).all())
            self.assertGreaterEqual(cursor.min(),0.)
            self.assertLessEqual(cursor.max(),1.2)
            self.assertTrue((np.diff(cursor)/DT>=.6-1e-10).all())
            self.assertTrue((np.diff(cursor)/DT<=1.4+1e-10).all())

    def test_invalid_parameters_and_phase_rejected(self):
        for lead in ([.121,0],[np.nan,0],[0]):
            with self.assertRaises(ValueError):warped_head_prefix(self.ck,[5,6],lead)
        phases=self.ck['prefix_phases'].copy();phases[10]=1
        with self.assertRaises(ValueError):warped_head_prefix({**self.ck,'prefix_phases':phases},[5,6],[.06,0])

    def test_fresh_split(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT))
        self.assertEqual((HELDOUT[0],HELDOUT[-1]),(794000,794039))


if __name__=='__main__':unittest.main()
