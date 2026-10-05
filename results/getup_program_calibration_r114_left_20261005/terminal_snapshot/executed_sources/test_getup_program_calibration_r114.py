import unittest
import numpy as np
from diagnostics import train_getup_program_r113 as program
from diagnostics.train_getup_program_calibration_r114 import calibrate,SOURCE


class CalibrationTest(unittest.TestCase):
    def test_fixed_decoder_causal_scalar_and_preserved_flags(self):
        x,flags,nodes,frozen,records=program.load_program_data()
        with np.load(SOURCE/'training/update_01500.npz') as data:w={k:data[k].copy() for k in data.files}
        labels=[]
        for case in (None,*program.TRAIN):
            with np.load(program.TEACHERS/f'case_{case}/teacher_parameters.npz') as data:labels.append(data['knots'].copy())
        trained,report=calibrate(w,x,np.stack(labels))
        self.assertEqual(report['rank'],25);self.assertLess(report['effective_node_max_error'],1e-10)
        self.assertFalse(any(k in trained for k in ('design','targets','case_seed','context_library')))
        np.testing.assert_array_equal(program.predict_program(trained,x[0])[1],np.zeros((6,10)))
        unseen=x[1].copy();unseen[0]+=.001
        profile,knots,meta=program.predict_program(trained,unseen)
        self.assertIn(profile,(0,1));self.assertTrue(np.isfinite(knots).all());self.assertLessEqual(np.abs(knots).max(),1.)
        for control in (0,50,100,200,400,529,2278):
            action=program.execute_program(trained,unseen,control,profile,knots)
            self.assertTrue(np.isfinite(action).all());self.assertLessEqual(np.abs(action).max(),1.)


if __name__=='__main__':unittest.main()
