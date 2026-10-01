import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from diagnostics.getup_ik_lift import plan_target


class KinematicProposalTests(unittest.TestCase):
    def test_desired_root_is_only_written_to_scratch_not_dynamic_robot(self):
        qpos=np.zeros(21)
        qpos[2],qpos[3]=.1,1.
        foot_pos=np.array([[0.,.05,0.],[0.,-.05,0.]])
        rotations=np.tile(np.eye(3).reshape(1,9),(2,1))
        sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos,xpos=foot_pos,xmat=rotations),
                            feet=[0,1],qadr=np.arange(7,21),home=np.zeros(14),prev=np.zeros(14),
                            lower=np.full(14,-1.57),upper=np.full(14,1.57),model=object())
        scratch=SimpleNamespace(qpos=qpos.copy(),xpos=foot_pos.copy(),xmat=rotations.copy())
        before=qpos.copy()
        with patch('diagnostics.getup_ik_lift.mujoco.mj_kinematics'):
            target,plan=plan_target(sim,scratch,np.zeros(2),[np.eye(3),np.eye(3)],.02,.1,.3)
        np.testing.assert_array_equal(sim.data.qpos,before)
        self.assertAlmostEqual(scratch.qpos[2],.12)
        self.assertTrue(plan['scratch_only'])
        self.assertTrue(np.isfinite(target).all())
        self.assertTrue(np.all(target>=sim.lower))
        self.assertTrue(np.all(target<=sim.upper))


if __name__=='__main__':
    unittest.main()
