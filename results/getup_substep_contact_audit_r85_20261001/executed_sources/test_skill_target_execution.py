import unittest
from types import SimpleNamespace
import numpy as np
from diagnostics.skill_target_execution import SkillTargetExecutor


class ExecutionTest(unittest.TestCase):
    def setUp(self):
        self.s = SimpleNamespace(default_actuator=np.zeros(2),action_scale=.25,
            max_motor_velocity=5.24,prev_motor_targets=np.zeros(2),motor_targets=np.zeros(2),
            last_action=np.array([.1,.2]),last_last_action=np.array([.3,.4]),
            last_last_last_action=np.array([.5,.6]),data=SimpleNamespace(ctrl=np.zeros(2)))
        self.e = SkillTargetExecutor(self.s,.02,np.full(2,-1.),np.full(2,1.),.01)

    def test_legacy_has_no_extra_filter_and_preserves_raw_history(self):
        self.e.apply([.075,-.05],[.3,-.2],False)
        np.testing.assert_allclose(self.s.motor_targets,[.075,-.05])
        np.testing.assert_allclose(self.s.last_action,[.3,-.2])

    def test_three_history_frames_are_shifted_freshly(self):
        self.e.apply([.075,-.05],[.3,-.2],False)
        np.testing.assert_allclose(self.s.last_last_action,[.1,.2])
        np.testing.assert_allclose(self.s.last_last_last_action,[.3,.4])

    def test_backward_filter_and_preslew_history(self):
        self.e.apply([.3,-.3],None,True)
        np.testing.assert_allclose(self.e.filtered,[.2,-.2])
        np.testing.assert_allclose(self.s.motor_targets,[.1048,-.1048])
        np.testing.assert_allclose(self.s.last_action,[.8,-.8])

    def test_backward_entry_uses_actual_previous_motor_target(self):
        self.e.apply([.8,-.8],[3.2,-3.2],False)
        physical=self.s.prev_motor_targets.copy()
        self.e.apply([.2,-.2],None,True)
        np.testing.assert_allclose(self.e.filtered,physical+2/3*(np.array([.2,-.2])-physical))

    def test_joint_clamp_and_slew_are_retained(self):
        self.e.apply([2.,-2.],[8.,-8.],False)
        np.testing.assert_allclose(self.e.filtered,[1.,-1.])
        np.testing.assert_allclose(self.s.motor_targets,[.1048,-.1048])
        np.testing.assert_allclose(self.s.last_action,[8.,-8.])

    def test_bad_target_rejected_before_state_change(self):
        with self.assertRaises(ValueError):
            self.e.apply([np.nan,0.],[0.,0.],False)
        np.testing.assert_allclose(self.s.prev_motor_targets,[0.,0.])


if __name__=='__main__':
    unittest.main()
