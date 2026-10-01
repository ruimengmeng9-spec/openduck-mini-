import unittest
import numpy as np
from diagnostics.search_getup_prefix_timing_r75 import warped_prefix


class PrefixTimingTests(unittest.TestCase):
    def setUp(self):
        self.ck = {'prefix_targets': np.arange(450*14,dtype=float).reshape(450,14),
                   'prefix_phases': np.r_[np.zeros(308,int),np.ones(100,int),np.full(42,2)]}

    def test_identity_is_bit_exact(self):
        out = warped_prefix(self.ck,np.ones(4))
        for key in self.ck: np.testing.assert_array_equal(out[key],self.ck[key])

    def test_changes_only_early_timing_and_preserves_endpoints(self):
        out = warped_prefix(self.ck,np.array([.8,1,1.2,.9]))
        n = np.count_nonzero(out['prefix_phases']==0)
        np.testing.assert_array_equal(out['prefix_targets'][n:],self.ck['prefix_targets'][308:])
        np.testing.assert_array_equal(out['prefix_phases'][n:],self.ck['prefix_phases'][308:])
        np.testing.assert_array_equal(out['prefix_targets'][0],self.ck['prefix_targets'][0])
        np.testing.assert_array_equal(out['prefix_targets'][n-1],self.ck['prefix_targets'][307])
        np.testing.assert_array_equal(self.ck['prefix_targets'],np.arange(450*14,dtype=float).reshape(450,14))

    def test_interpolates_within_original_spatial_envelope(self):
        out = warped_prefix(self.ck,np.full(4,1.3))
        n = np.count_nonzero(out['prefix_phases']==0)
        self.assertGreater(n,308)
        self.assertGreaterEqual(out['prefix_targets'][:n].min(),self.ck['prefix_targets'][:308].min())
        self.assertLessEqual(out['prefix_targets'][:n].max(),self.ck['prefix_targets'][:308].max())

    def test_bad_factors_and_invalid_phase_rejected(self):
        for v in [np.ones(3),np.full(4,np.nan),np.full(4,.69),np.full(4,1.31)]:
            with self.assertRaises(ValueError): warped_prefix(self.ck,v)
        self.ck['prefix_phases'][100]=1
        with self.assertRaises(ValueError): warped_prefix(self.ck,np.ones(4))


if __name__=='__main__': unittest.main()
