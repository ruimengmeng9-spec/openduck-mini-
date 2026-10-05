import unittest
import numpy as np
from diagnostics.probe_getup_teacher_neighborhood_r118 import nearest_pairs


class NeighborTest(unittest.TestCase):
    def test_sensor_only_nearest_two_deterministic(self):
        x=np.random.default_rng(218).normal(size=(65,55))
        a=nearest_pairs(x);self.assertEqual(a,nearest_pairs(x));self.assertEqual(len(a),130)
        for i in range(65):
            pair=[r for r in a if r[0]==i];self.assertEqual(len(pair),2)
            self.assertTrue(all(r[1]!=i and r[2]>0 and np.isfinite(r[2]) for r in pair))
        with self.assertRaises(AssertionError):nearest_pairs(np.full((65,55),np.nan))


if __name__=='__main__':unittest.main()
