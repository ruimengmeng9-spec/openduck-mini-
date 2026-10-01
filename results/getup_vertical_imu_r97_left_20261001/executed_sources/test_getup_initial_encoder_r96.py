import unittest
from unittest.mock import patch
import numpy as np
from diagnostics.probe_getup_initial_encoder_r96 import encoder_offset,run,TRAIN,HELDOUT,CAP


class InitialEncoderTest(unittest.TestCase):
    def test_group_scaling_and_fade(self):
        delta=np.arange(10.)*.001
        np.testing.assert_array_equal(encoder_offset(delta,(1,2),.8,0.),np.r_[delta[:8],2*delta[8:]])
        np.testing.assert_allclose(encoder_offset(delta,(1,2),.8,.4),.5*np.r_[delta[:8],2*delta[8:]])
        np.testing.assert_array_equal(encoder_offset(delta,(1,2),.8,.8),np.zeros(10))

    def test_nominal_and_zero_gain_exact(self):
        np.testing.assert_array_equal(encoder_offset(np.zeros(10),(2,2),1.2,.1),np.zeros(10))
        np.testing.assert_array_equal(encoder_offset(np.ones(10),(0,0),.4,.1),np.zeros(10))

    def test_cap_and_bad_inputs(self):
        self.assertLessEqual(np.max(np.abs(encoder_offset(np.ones(10)*100,(2,2),.4,0.))),CAP)
        for delta,scales,fade,t in [(np.ones(9),(1,1),.4,0.),(np.ones(10),(3,1),.4,0.),
                                    (np.ones(10),(1,1),.6,0.),(np.ones(10),(1,1),.4,-.1)]:
            with self.assertRaises(ValueError):encoder_offset(delta,scales,fade,t)

    def test_literal_baseline(self):
        with patch('diagnostics.probe_getup_initial_encoder_r96.rollout',return_value=('base',[])) as fn:
            self.assertEqual(run(None,None,None,None,None,None,None,None,None,(0,0),.4,(0,0),True),('base',[]))
            fn.assert_called_once()

    def test_fresh_split(self):
        self.assertFalse(set(TRAIN)&set(HELDOUT))
        self.assertEqual((HELDOUT[0],HELDOUT[-1]),(796000,796039))

    def test_literal_nominal_zero_error(self):
        with patch('diagnostics.probe_getup_initial_encoder_r96.rollout',return_value=('nominal',[])) as fn:
            self.assertEqual(run(None,None,None,None,None,None,None,None,np.zeros(10),(2,-1),1.2,(0,0),True),
                             ('nominal',[]))
            fn.assert_called_once()


if __name__=='__main__':unittest.main()
