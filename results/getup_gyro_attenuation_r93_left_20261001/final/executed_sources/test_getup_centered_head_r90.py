import unittest
from unittest.mock import patch
import numpy as np
from diagnostics import train_getup_head_distillation_r89 as student
from diagnostics.probe_getup_centered_head_r90 import centered_prediction,run_centered,TRAIN,HELDOUT


class CenteredHeadTest(unittest.TestCase):
    def network(self):
        return {'center':np.zeros(35),'scale':np.ones(35),'w1':np.zeros((35,64)),
                'b1':np.zeros(64),'w2':np.zeros((64,2)),'b2':np.array([.01,-.02])}

    def test_nominal_prediction_exact_zero(self):
        x=np.zeros(35);x[-1]=.5/3.5
        np.testing.assert_array_equal(centered_prediction(self.network(),x,np.tile([.01,-.02],(30,1))),[0.,0.])

    def test_current_reference_time_only(self):
        x=np.zeros(35);x[-1]=.5/3.5;reference=np.zeros((30,2));reference[25]=[.005,-.01]
        np.testing.assert_allclose(centered_prediction(self.network(),x,reference),[.005,-.01])

    def test_prediction_hook_restored_on_failure(self):
        before=student.predict
        with patch.object(student,'run',side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):
                run_centered(None,None,None,None,None,None,None,None,{},None,1.)
        self.assertIs(student.predict,before)

    def test_fresh_split(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT))
        self.assertFalse(set(range(789000,789040))&set(HELDOUT))


if __name__=='__main__':unittest.main()
