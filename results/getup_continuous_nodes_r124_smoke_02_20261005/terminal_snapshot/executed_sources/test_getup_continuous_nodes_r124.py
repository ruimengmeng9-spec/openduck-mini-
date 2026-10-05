import unittest
import numpy as np
from diagnostics.probe_getup_continuous_nodes_r124 import predict


class GateTest(unittest.TestCase):
    def model(self):
        w=dict(feature_mode=np.array('snapshot'),mean=np.zeros(50),std=np.ones(50),nominal_feature=np.zeros(50),
            flags=np.zeros((51,2)),nodes=np.ones((50,60))*.01)
        w['flags'][-1]=[1.,-1.]
        return w

    def test_only_gate_changes(self):
        w=self.model();h=np.ones((40,50))
        p,k,info=predict(w,h,'original');q,v,other=predict(w,h,'continuous')
        self.assertEqual(p,q);self.assertEqual(info,other)
        np.testing.assert_array_equal(k,np.zeros((6,10)));self.assertTrue(np.all(v>0.))

    def test_enabled_and_nominal_exact(self):
        w=self.model();w['flags'][-1,1]=1.
        h=np.ones((40,50))
        np.testing.assert_array_equal(predict(w,h,'original')[1],predict(w,h,'continuous')[1])
        np.testing.assert_array_equal(predict(w,np.zeros((40,50)),'continuous')[1],np.zeros((6,10)))

    def test_bounds_and_invalid_inputs(self):
        w=self.model();self.assertLessEqual(np.abs(predict(w,np.ones((40,50))*1e8,'continuous')[1]).max(),1.)
        with self.assertRaises(ValueError):predict(w,np.zeros((39,50)),'continuous')
        with self.assertRaises(ValueError):predict(w,np.zeros((40,50)),'other')


if __name__=='__main__':unittest.main()
