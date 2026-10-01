import unittest

import numpy as np

from diagnostics.getup_native_curriculum import at_goal,reward_terms
from diagnostics.train_getup_native_ppo import generalized_advantage


class NativePpoTests(unittest.TestCase):
    def metric(self,height=.165):
        return dict(height_m=height,up_z=.99,stable=True,foot_load_fraction=1.,
                    joint_home_error_mean_rad=.1,joint_home_error_max_rad=.2)

    def test_supported_low_stand_is_not_goal_stand(self):
        self.assertTrue(at_goal(self.metric()))
        self.assertFalse(at_goal(self.metric(.14)))
        self.assertEqual(reward_terms(self.metric(.14),.14,.1)['stable'],0.)

    def test_wrong_joint_configuration_is_not_goal_stand(self):
        m=self.metric();m['joint_home_error_max_rad']=2.
        self.assertFalse(at_goal(m))

    def test_timeout_bootstraps_but_does_not_cross_reset(self):
        rewards=np.array([[1.],[100.]],np.float32)
        values=np.array([[2.],[0.]],np.float32)
        next_values=np.array([[5.],[0.]],np.float32)
        done=np.array([[True],[False]])
        terminated=np.array([[False],[False]])
        adv,_=generalized_advantage(rewards,values,next_values,done,terminated)
        self.assertAlmostEqual(float(adv[0,0]),3.95,places=5)
        terminated[0,0]=True
        adv,_=generalized_advantage(rewards,values,next_values,done,terminated)
        self.assertAlmostEqual(float(adv[0,0]),-1.,places=5)

    def test_reward_prefers_real_goal_not_stationary_crouch(self):
        home=sum(reward_terms(self.metric(),.165,.1).values())
        low=sum(reward_terms(self.metric(.14),.14,.2).values())
        self.assertGreater(home,low)
        partial=self.metric();partial['foot_load_fraction']=.2;partial['stable']=False
        self.assertGreater(home,sum(reward_terms(partial,.165,.1).values()))


if __name__=='__main__':
    unittest.main()
