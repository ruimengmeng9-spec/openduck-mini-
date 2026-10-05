import unittest
import numpy as np
from diagnostics.train_getup_program_r113 import load_program_data,checked_context,predict_program,execute_program,interpolate_nodes
from diagnostics.search_getup_case_teachers_r109 import knot_action


class ProgramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.x,cls.flags,cls.nodes,cls.frozen,cls.records=load_program_data()

    def encoder(self):
        rng=np.random.default_rng(213);w=self.frozen.copy()
        for name,shape in [('hidden0',(55,64)),('hidden1',(64,32)),('flags',(32,2)),('knots',(32,60))]:
            w[name+'_kernel']=rng.normal(0,.1,size=shape).astype(np.float32);w[name+'_bias']=rng.normal(0,.1,size=shape[1]).astype(np.float32)
        return w

    def test_causal_input_and_payload_no_case_library(self):
        self.assertEqual(self.x.shape,(25,55));self.assertEqual(self.nodes.shape,(25,6,10))
        w=self.encoder();self.assertFalse(any(k in w for k in ('case_seed','teacher_parameters','initial_context_library','contexts')))
        with self.assertRaises(ValueError):checked_context(np.ones(56))
        with self.assertRaises(ValueError):checked_context(np.full(55,np.nan))

    def test_scalar_nominal_nodes_exact_zero(self):
        w=self.encoder();profile,knots,meta=predict_program(w,w['nominal_context'])
        np.testing.assert_array_equal(knots,np.zeros((6,10)))
        self.assertIn(profile,(0,1));self.assertTrue(np.abs(knots).max()<=1.)

    def test_node_interpolation_exact_and_bounds(self):
        knots=self.nodes[-1].astype(float)
        for k in range(2279):np.testing.assert_array_equal(interpolate_nodes(knots,k),knot_action(knots,k))
        np.testing.assert_array_equal(interpolate_nodes(knots,0),np.zeros(10))
        with self.assertRaises(ValueError):interpolate_nodes(np.ones((6,10))*2,1)
        with self.assertRaises(ValueError):interpolate_nodes(knots,2279)

    def test_home_and_zero_program(self):
        w=self.encoder();obs=self.x[1]
        np.testing.assert_array_equal(execute_program(w,obs,1,0,np.zeros((6,10))),np.zeros(10))
        np.testing.assert_array_equal(execute_program(w,obs,529,1,np.ones((6,10))*.1),np.zeros(10))
        self.assertTrue(np.abs(execute_program(w,obs,100,1,np.ones((6,10)))).max()<=1.)


if __name__=='__main__':unittest.main()
