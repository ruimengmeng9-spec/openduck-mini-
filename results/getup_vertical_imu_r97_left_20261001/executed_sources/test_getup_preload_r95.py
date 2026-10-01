import unittest
from types import SimpleNamespace
import numpy as np
from diagnostics.train_getup_preload_r95 import edited_preload,fitness,feasible_rank,TRAIN,HELDOUT


class PreloadTest(unittest.TestCase):
    def setUp(self):
        self.sim=SimpleNamespace(lower=np.full(14,-2.),upper=np.full(14,2.))
        self.ck={'prefix_targets':np.zeros((100,14)),'prefix_phases':np.zeros(100,dtype=int)}
        self.ids=np.arange(10)

    def test_zero_is_exact(self):
        out=edited_preload(self.sim,self.ck,self.ids,np.zeros(10))
        np.testing.assert_array_equal(out['prefix_targets'],self.ck['prefix_targets'])

    def test_start_end_later_and_other_actuators_unchanged(self):
        out=edited_preload(self.sim,self.ck,self.ids,np.full(10,.1))['prefix_targets']
        np.testing.assert_array_equal(out[0],np.zeros(14))
        np.testing.assert_array_equal(out[45:],np.zeros((55,14)))
        np.testing.assert_array_equal(out[:,10:],np.zeros((100,4)))
        np.testing.assert_allclose(out[15,:10],.1)
        np.testing.assert_array_equal(self.ck['prefix_targets'],np.zeros((100,14)))

    def test_bounds_and_phase_rejected(self):
        for flat in ([0]*9,[.121]*10,[np.nan]*10):
            with self.assertRaises(ValueError):edited_preload(self.sim,self.ck,self.ids,flat)
        ck={**self.ck,'prefix_phases':np.ones(100,dtype=int)}
        with self.assertRaises(ValueError):edited_preload(self.sim,ck,self.ids,np.zeros(10))

    def test_search_can_use_failed_nominal_but_selection_cannot(self):
        nominal={'success':False,'score':10.};cases=[{'success':True,'score':20.}]*24
        score=fitness(nominal,cases,np.zeros(10));self.assertTrue(np.isfinite(score))
        bad={'nominal_success':False,'successes':24,'fitness':score}
        good={'nominal_success':True,'successes':13,'fitness':0.}
        self.assertGreater(feasible_rank(good),feasible_rank(bad))

    def test_split_fresh(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT))
        self.assertEqual((HELDOUT[0],HELDOUT[-1]),(795000,795039))


if __name__=='__main__':unittest.main()
