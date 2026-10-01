import unittest
from types import SimpleNamespace
import numpy as np
from diagnostics.getup_independent_native import canonical_quaternion, POSES
from diagnostics.probe_getup_support_transfer_r28 import templates
from diagnostics.search_getup_reverse_path_r29 import fallen_orientation, proposals


class SupportTransferTests(unittest.TestCase):
    def sim(self):
        return SimpleNamespace(home=np.zeros(14),lower=np.full(14,-1.57),upper=np.full(14,1.57))

    def test_four_forward_fall_labels_match_existing_canonical_rotations(self):
        for pose in POSES:
            self.assertEqual(fallen_orientation(canonical_quaternion(pose)),pose)
        self.assertIsNone(fallen_orientation(np.array([1.,0.,0.,0.])))

    def test_normal_knee_templates_preserve_physical_limits(self):
        sim=self.sim(); original=sim.lower.copy()
        rows=list(templates(sim))
        self.assertEqual(len(rows),96)
        for _,q,d in rows:
            self.assertTrue(np.all(q>=sim.lower) and np.all(q<=sim.upper))
            self.assertTrue(np.all(q[:,[3,12]]>=0))
            self.assertTrue(np.all(d>0))
            np.testing.assert_array_equal(q[-1],sim.home)
        np.testing.assert_array_equal(sim.lower,original)

    def test_forward_proposals_are_bounded_and_nonzero_duration(self):
        sim=self.sim()
        for q,d in proposals(sim,np.random.default_rng(1),40):
            self.assertTrue(np.isfinite(q).all())
            self.assertTrue(np.all(q>=sim.lower) and np.all(q<=sim.upper))
            self.assertTrue(np.all(d>0))


if __name__=='__main__':
    unittest.main()
