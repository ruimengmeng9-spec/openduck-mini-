import unittest
from types import SimpleNamespace
import numpy as np
from diagnostics.search_getup_contact_archive_r30 import contact_cell, mutate_target


class ContactArchiveTests(unittest.TestCase):
    def test_contact_changes_and_low_tilt_states_remain_distinct(self):
        m=dict(height_m=.07,feet=[False,False],torso_contact=True,foot_load_fraction=0.)
        first=contact_cell(m,[1,0,0],[-1,1])
        self.assertNotEqual(first,contact_cell(m,[0,1,0],[-1,1]))
        self.assertNotEqual(first,contact_cell(m|dict(feet=[True,False]),[1,0,0],[-1,1]))
        self.assertNotEqual(first,contact_cell(m,[1,0,0],[1,1]))

    def test_proposals_preserve_limits_and_include_head_exploration(self):
        sim=SimpleNamespace(home=np.zeros(14),lower=np.full(14,-1.57),upper=np.full(14,1.57))
        parent=dict(result=dict(actual_joint_angles=np.zeros(14),previous_target=np.zeros(14)))
        rng=np.random.default_rng(30)
        original=sim.lower.copy()
        for kind in range(7):
            q=mutate_target(sim,parent,rng,kind)
            self.assertTrue(np.isfinite(q).all())
            self.assertTrue(np.all(q>=sim.lower) and np.all(q<=sim.upper))
        self.assertTrue(np.any(q[7:9]!=0))
        np.testing.assert_array_equal(original,sim.lower)


if __name__=='__main__':
    unittest.main()
