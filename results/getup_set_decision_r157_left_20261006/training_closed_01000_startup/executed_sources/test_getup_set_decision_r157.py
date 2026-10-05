import unittest
import numpy as np
from diagnostics import train_getup_set_decision_r157 as run

class TestSetDecision(unittest.TestCase):
    def test_gradient_finite_difference(self):
        logits=np.array([[.2,-.4,.7,.1],[-.3,.5,.4,.1]])
        success=np.array([[1,0,1,0],[0,1,0,0]],bool)
        _,gradient=run.set_loss_gradient(logits,success)
        for i in range(2):
            for j in range(4):
                plus=logits.copy();minus=logits.copy();plus[i,j]+=1e-6;minus[i,j]-=1e-6
                numerical=(run.set_loss_gradient(plus,success)[0]-run.set_loss_gradient(minus,success)[0])/2e-6
                self.assertAlmostEqual(gradient[i,j],numerical,places=8)
    def test_any_success_not_forced_single_teacher(self):
        labels=np.array([[1,0,1,0]],bool)
        a=run.set_loss_gradient(np.array([[20.,0.,-20.,0.]]),labels)[0]
        b=run.set_loss_gradient(np.array([[-20.,0.,20.,0.]]),labels)[0]
        self.assertAlmostEqual(a,b,places=12);self.assertLess(a,1e-7)
    def test_all_success_zero_gradient(self):
        loss,g=run.set_loss_gradient(np.array([[.1,.2,.3,.4]]),np.ones((1,4),bool))
        self.assertEqual(loss,0.);np.testing.assert_array_equal(g,np.zeros((1,4)))
    def test_invalid_empty_and_nan(self):
        for logits,y in [(np.zeros((1,4)),np.zeros((1,4),bool)),(np.full((1,4),np.nan),np.ones((1,4),bool))]:
            with self.assertRaises(ValueError):run.set_loss_gradient(logits,y)
    def test_export_causal_signature_and_finite_bounds(self):
        import inspect
        self.assertEqual(list(inspect.signature(run.old.predict).parameters),['model','sensors'])
        m=dict(mean=np.zeros(50),std=np.ones(50),weight=np.zeros((50,4)),bias=np.array([0.,1.,0.,0.]),global_gains=np.zeros((4,6)))
        g,c,_=run.old.predict(m,np.zeros(50));self.assertEqual(c,1);np.testing.assert_array_equal(g,np.zeros(6))
        with self.assertRaises(ValueError):run.old.predict(m,np.zeros(51))
        with self.assertRaises(ValueError):run.old.predict(m,np.full(50,np.nan))
    def test_nominal_scalar_and_total_bound(self):
        current=np.arange(55,dtype=np.float32)/100
        extra=run.old.local.local_feedback(current,current,np.array([9,10,11]),np.array([1.,-1.,1.,-1.,1.,-1.]))
        np.testing.assert_array_equal(extra,np.zeros(3))
        base=np.zeros(14);target=np.full(14,.18)
        merged=run.old.local.merge_target(target,base,np.full(3,100.),np.array([9,10,11]),np.full(14,-2.),np.full(14,2.))
        self.assertLessEqual(np.max(np.abs(merged-base)),.18)

if __name__=='__main__':unittest.main()
