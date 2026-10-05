import unittest
import numpy as np
from diagnostics.probe_getup_common_prefix_r136 import causal_feedback,ORIGINAL_FEEDBACK


class PrefixTests(unittest.TestCase):
    def test_ten_controls_then_original_formula(self):
        reference=np.zeros(55,np.float32);current=reference.copy();current[15]=.03;gains=np.ones(6)
        for k in range(11):
            current[-1]=k/529
            actual=causal_feedback(current,reference,[9,10,11],gains,10)
            np.testing.assert_array_equal(actual,np.zeros(3) if k<10 else ORIGINAL_FEEDBACK(current,reference,[9,10,11],gains))

    def test_zero_delay_nominal_boundaries(self):
        obs=np.linspace(-1,1,55,dtype=np.float32)
        np.testing.assert_array_equal(causal_feedback(obs,obs,[9,10,11],np.ones(6),10),np.zeros(3))
        with self.assertRaises(ValueError):causal_feedback(obs,obs,[9,10,11],np.ones(6),11)
        with self.assertRaises(ValueError):causal_feedback(np.zeros(56),obs,[9,10,11],np.ones(6),0)


if __name__=='__main__':unittest.main()
