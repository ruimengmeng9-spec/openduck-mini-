import unittest
import numpy as np
from diagnostics.train_getup_history_program_r122 import sensor_features,fit_ridge,scalar_program,RIDGES


class ProgramTest(unittest.TestCase):
    def test_feature_boundary(self):
        h=np.arange(2000).reshape(40,50)
        np.testing.assert_array_equal(sensor_features(h,'snapshot'),h[-1])
        self.assertEqual(sensor_features(h,'history').shape,(200,))
        for bad in (np.zeros((39,50)),np.full((40,50),np.nan)):
            with self.assertRaises(ValueError):sensor_features(bad,'history')

    def test_earlier_history_changes_history_only(self):
        a=np.zeros((40,50));b=a.copy();b[0,0]=1.
        np.testing.assert_array_equal(sensor_features(a,'snapshot'),sensor_features(b,'snapshot'))
        self.assertFalse(np.array_equal(sensor_features(a,'history'),sensor_features(b,'history')))

    def test_nominal_scalar_exact_zero_and_bounded(self):
        rng=np.random.default_rng(222);histories=rng.normal(size=(8,40,50))
        for mode in ('snapshot','history'):
            x=np.stack([sensor_features(h,mode) for h in histories]);nodes=rng.uniform(-1,1,(8,6,10));nodes[0]=0
            w=fit_ridge(x,np.ones((8,2)),nodes,x[0],10.)
            w['feature_mode']=np.array(mode)
            np.testing.assert_array_equal(scalar_program(w,histories[0])[1],np.zeros((6,10)))
            self.assertLessEqual(np.abs(scalar_program(w,histories[1]*1e3)[1]).max(),1.)
            self.assertEqual(set(w),{'mean','std','nominal_feature','flags','nodes','feature_mode'})

    def test_deterministic_regularized_fit(self):
        x=np.arange(80).reshape(8,10);flags=np.zeros((8,2));nodes=np.zeros((8,6,10))
        for r in RIDGES:
            a=fit_ridge(x,flags,nodes,x[0],r);b=fit_ridge(x,flags,nodes,x[0],r)
            for key in a:np.testing.assert_array_equal(a[key],b[key])


if __name__=='__main__':unittest.main()
