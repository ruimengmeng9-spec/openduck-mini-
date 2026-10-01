import unittest
from unittest.mock import patch
import numpy as np
from diagnostics.train_getup_vertical_imu_r97 import vertical_offset,run,TRAIN,HELDOUT,CAP


class VerticalFeedbackTest(unittest.TestCase):
    def test_order_and_fade(self):
        x=vertical_offset(.1,(1,2,-1),0.)
        np.testing.assert_allclose(x[[0,4,8,9]],[.1,-.1,CAP,-.1])
        self.assertEqual(np.count_nonzero(x),4)
        np.testing.assert_allclose(vertical_offset(.1,(1,1,1),2.35),.5*vertical_offset(.1,(1,1,1),0.))
        np.testing.assert_array_equal(vertical_offset(.1,(1,1,1),3.5),np.zeros(10))

    def test_zero_error_and_bounds(self):
        np.testing.assert_array_equal(vertical_offset(0.,(2,-2,1),.5),np.zeros(10))
        self.assertLessEqual(abs(vertical_offset(99.,(2,2,2),0.)).max(),CAP)
        for e,g,t in [(np.nan,(1,1,1),0),(0,(3,0,0),0),(0,(1,1),0),(0,(1,1,1),-1)]:
            with self.assertRaises(ValueError):vertical_offset(e,g,t)

    def test_literal_baseline(self):
        with patch('diagnostics.train_getup_vertical_imu_r97.rollout',return_value=('identity',[])) as fn:
            self.assertEqual(run(None,None,None,None,None,None,None,None,None,np.zeros(3),(0,0)),('identity',[],None))
            fn.assert_called_once()

    def test_fresh_split(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT))
        self.assertEqual((HELDOUT[0],HELDOUT[-1]),(797000,797039))


if __name__=='__main__':unittest.main()
