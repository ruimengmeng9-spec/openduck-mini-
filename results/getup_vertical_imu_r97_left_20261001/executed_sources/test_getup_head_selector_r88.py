import unittest
import numpy as np
from diagnostics.audit_getup_head_selector_r88 import observation, choose, screen


class HeadSelectorTest(unittest.TestCase):
    def test_encoder_and_imu_inputs_only(self):
        e=np.arange(104).reshape(26,4);q=np.arange(14);v=np.arange(14)*2
        x=observation(e,q,v,'fusion')
        self.assertEqual(x.shape,(40,))
        np.testing.assert_array_equal(x[:12],e[[5,15,25]].ravel())
        np.testing.assert_array_equal(x[12:26],q)

    def test_baseline_tie_preference(self):
        x=np.zeros((3,4));y=np.ones((3,2));p=np.array([[0.]*8,[.1]*8])
        self.assertEqual(choose(np.zeros(4),x,y,p,3,.25),0)

    def test_query_label_excluded(self):
        x=np.array([[0.,0.],[1.,1.],[2.,2.]])
        y=np.array([[0,1],[1,0],[1,0]],dtype=bool);p=np.array([[0.]*8,[.1]*8])
        first=screen(x,y,p,1,.25)[0]['selected_arm']
        y[0]=~y[0]
        self.assertEqual(first,screen(x,y,p,1,.25)[0]['selected_arm'])

    def test_invalid_inputs_rejected(self):
        for query,k in (([np.nan,0.],1),([0.,0.],4)):
            with self.assertRaises(ValueError):
                choose(query,np.zeros((3,2)),np.zeros((3,2)),np.zeros((2,8)),k,0.)


if __name__=='__main__':unittest.main()
