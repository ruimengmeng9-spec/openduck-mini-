import unittest
import numpy as np
from diagnostics.train_getup_sequence_r24 import sequence_choice,select_radius


class TrajectoryPolicyTest(unittest.TestCase):
    def setUp(self):
        a=np.zeros(50);b=np.ones(50)
        self.library=dict(obs=np.array([a,b]),rescue=np.array([True,False]),
            knots=np.array([np.full((2,8),.2),np.zeros((2,8))]),seed=np.array([1,2]))

    def test_exact_teacher_sequence_no_averaging(self):
        q,c=sequence_choice(np.zeros(50),self.library,.3)
        np.testing.assert_array_equal(q,np.full((2,8),.2));self.assertEqual(c['prototype_seed'],1)

    def test_home_anchor_and_unknown_fallback(self):
        for obs in (np.ones(50),np.full(50,100.)):
            q,c=sequence_choice(obs,self.library,.3)
            np.testing.assert_array_equal(q,np.zeros((2,8)));self.assertEqual(c['mode'],'home')

    def test_zero_radius_always_home(self):
        q,c=sequence_choice(np.zeros(50),self.library,0.)
        np.testing.assert_array_equal(q,np.zeros((2,8)))

    def test_invalid_observation_and_radius(self):
        for obs,r in ((np.zeros(49),.3),(np.full(50,np.nan),.3),(np.zeros(50),-1.)):
            with self.assertRaises(ValueError):sequence_choice(obs,self.library,r)

    def test_selection_penalizes_lost_success_and_baseline_wins_ties(self):
        summary={'0.0':dict(new_successes=0,new_failures=0),'0.3':dict(new_successes=2,new_failures=1)}
        self.assertEqual(select_radius(summary),0.)
        summary['0.6']=dict(new_successes=1,new_failures=0)
        self.assertEqual(select_radius(summary),.6)


if __name__=='__main__':unittest.main()
