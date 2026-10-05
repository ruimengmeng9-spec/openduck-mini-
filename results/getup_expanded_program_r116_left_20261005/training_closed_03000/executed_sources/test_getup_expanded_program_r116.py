import unittest
import numpy as np
from diagnostics.train_getup_expanded_program_r116 import load_data,calibrate_decoder
from diagnostics import train_getup_program_r113 as program


class ExpandedProgramTest(unittest.TestCase):
    def test_data_causal_and_decoder_precision(self):
        x,y,nodes,frozen,records=load_data()
        self.assertEqual(x.shape,(65,55));self.assertEqual(nodes.shape,(65,6,10));self.assertEqual(len(records),65)
        rng=np.random.default_rng(216);w=frozen.copy()
        for name,shape in [('hidden0',(55,128)),('hidden1',(128,128)),('flags',(128,2)),('knots',(128,60))]:
            w[name+'_kernel']=rng.normal(0,.1,size=shape).astype(np.float32);w[name+'_bias']=np.zeros(shape[1],dtype=np.float32)
        calibrated,report,design,target=calibrate_decoder(w,x,nodes)
        self.assertEqual(report['rank'],65);self.assertLess(report['ungated_node_max_error'],1e-10)
        self.assertNotIn('design',calibrated);self.assertNotIn('target',calibrated);self.assertNotIn('case_seed',calibrated)
        np.testing.assert_array_equal(program.predict_program(calibrated,x[0])[1],np.zeros((6,10)))
        unseen=x[1].copy();unseen[0]+=.001
        p,k,m=program.predict_program(calibrated,unseen)
        self.assertTrue(np.isfinite(k).all());self.assertLessEqual(np.abs(k).max(),1.)
        self.assertLessEqual(np.abs(program.execute_program(calibrated,unseen,100,p,k)).max(),1.)


if __name__=='__main__':unittest.main()
