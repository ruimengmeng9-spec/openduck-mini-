import unittest
import numpy as np
from diagnostics.probe_getup_head_tolerance_r91 import grid,constant_network
from diagnostics.train_getup_head_distillation_r89 import predict


class ToleranceTest(unittest.TestCase):
    def test_prescribed_grid(self):
        rows=grid();self.assertEqual(len(rows),17)
        self.assertEqual(rows[0]['bias'],[0.,0.])
        self.assertTrue(all(max(map(abs,r['bias']))<=.005 for r in rows))

    def test_constant_head_only_output(self):
        net=constant_network([.001,-.002])
        np.testing.assert_array_equal(predict(net,np.arange(35.)),[.001,-.002])

    def test_bounds(self):
        for bias in ([.1,0.],[np.nan,0.],[0.]):
            with self.assertRaises(ValueError):constant_network(bias)


if __name__=='__main__':unittest.main()
