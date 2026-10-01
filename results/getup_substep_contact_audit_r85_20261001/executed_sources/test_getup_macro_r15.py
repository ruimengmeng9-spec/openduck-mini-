import unittest
import numpy as np
from diagnostics.getup_macro_curriculum_r15 import advance_macro, macro_advantage


class FakeEpisode:
    def __init__(self,terminal_at): self.steps=0; self.terminal_at=terminal_at
    def step(self,action):
        self.steps+=1
        done=self.steps==self.terminal_at
        return np.array([99 if done else self.steps]),1.,done,False,np.array([self.steps]),{'steps':self.steps} if done else None


class MacroTests(unittest.TestCase):
    def test_stops_at_timeout_and_does_not_step_reset_episode(self):
        e=FakeEpisode(2)
        obs,reward,done,term,bootstrap,info,n=advance_macro(e,np.zeros(14),5,.9)
        self.assertEqual(e.steps,2);self.assertEqual(n,2)
        self.assertAlmostEqual(reward,1.9);self.assertTrue(done);self.assertFalse(term)
        self.assertEqual(bootstrap[0],2);self.assertEqual(obs[0],99)

    def test_partial_macro_discount_and_reset_boundary(self):
        rewards=np.array([[1.],[100.]],np.float32)
        values=np.array([[2.],[0.]],np.float32)
        next_values=np.array([[5.],[0.]],np.float32)
        done=np.array([[True],[False]])
        terminated=np.array([[False],[False]])
        counts=np.array([[2.],[5.]])
        adv,_=macro_advantage(rewards,values,next_values,done,terminated,counts,5)
        self.assertAlmostEqual(float(adv[0,0]),1+5*.99**.4-2,places=5)
        terminated[0,0]=True
        adv,_=macro_advantage(rewards,values,next_values,done,terminated,counts,5)
        self.assertAlmostEqual(float(adv[0,0]),-1.,places=5)

    def test_full_macro_reward_is_discounted(self):
        e=FakeEpisode(100)
        *_,n=advance_macro(e,np.zeros(14),5,.9)
        self.assertEqual(n,5);self.assertEqual(e.steps,5)


if __name__=='__main__': unittest.main()
