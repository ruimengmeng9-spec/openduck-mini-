import unittest
from unittest.mock import patch
import numpy as np
from diagnostics.train_getup_head_feedback_r87 import matrix,correction,controlled_rollout,TRAIN,HELDOUT
from diagnostics.search_getup_reference_feedback_r64 import residual


class HeadFeedbackTest(unittest.TestCase):
    def test_zero_exact_original(self):
        error=np.array([.2,-.1,.3,-.2]);gain=np.arange(8)*.1
        np.testing.assert_array_equal(correction(error,gain,matrix(np.zeros(8)),2.),residual(error,gain))

    def test_only_head_receives_new_response(self):
        error=np.array([.2,-.1,.3,-.2]);gain=np.arange(8)*.1
        original=residual(error,gain);head=matrix(np.array([.1]*4+[-.1]*4))
        value=correction(error,gain,head,2.)
        np.testing.assert_array_equal(value[:8],original[:8])
        self.assertAlmostEqual(value[8],.02);self.assertAlmostEqual(value[9],-.02)

    def test_outside_window_exact_original(self):
        e=np.ones(4);g=np.ones(8)
        for t in (0.,.5,3.5,4.):
            np.testing.assert_array_equal(correction(e,g,matrix(np.ones(8)),t),residual(e,g))

    def test_shared_cap(self):
        self.assertLessEqual(abs(correction(np.ones(4),np.ones(8),matrix(np.full(8,2.)),2.)).max(),.18)

    def test_zero_rollout_delegates(self):
        with patch('diagnostics.train_getup_head_feedback_r87.rollout',return_value=('original',[])) as fn:
            self.assertEqual(controlled_rollout(None,None,{},None,None,None,None,None,np.zeros(8)),('original',[]))
            fn.assert_called_once()

    def test_invalid_parameters(self):
        for v in (np.zeros(7),np.full(8,2.1),np.full(8,np.nan)):
            with self.assertRaises(ValueError):matrix(v)

    def test_fresh_split(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT));self.assertFalse(set(range(786000,786040))&set(HELDOUT))


if __name__=='__main__':unittest.main()
