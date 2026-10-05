import unittest
import numpy as np
from diagnostics.train_getup_aggregate_r112 import dataset,WARM,base


class AggregationTest(unittest.TestCase):
    def test_relabel_shapes_bounds_and_frozen_statistics(self):
        original,labels,anchors,mean,std,nominal,x,y,t,teachers,manifest=dataset()
        self.assertEqual(x.shape[1],110);self.assertEqual(y.shape,(len(x),10));self.assertEqual(len(manifest),96)
        self.assertTrue(np.isfinite(x).all() and np.abs(y).max()<=1.);self.assertTrue(np.all(t<529))
        with np.load(WARM) as w:
            np.testing.assert_array_equal(w['input_mean'],mean);np.testing.assert_array_equal(w['input_std'],std)
            np.testing.assert_array_equal(w['nominal_observations'],nominal)
        self.assertTrue(all(r['case_seed'] in base.TRAIN for r in manifest))

    def test_warm_actor_scalar_nominal_zero(self):
        with np.load(WARM) as data:w={k:data[k].copy() for k in data.files}
        for k in range(529):np.testing.assert_array_equal(base.policy_action(w,w['nominal_observations'][k],w['nominal_observations'][0],k),np.zeros(10))


if __name__=='__main__':unittest.main()
