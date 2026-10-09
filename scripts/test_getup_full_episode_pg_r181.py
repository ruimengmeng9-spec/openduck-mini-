"""Mathematical and causal regressions; no simulation dynamics."""
import inspect
import unittest
import numpy as np
from diagnostics import train_getup_full_episode_pg_r181 as run

class Tests(unittest.TestCase):
    def setUp(self):
        self.p=run.initial_actor();self.probe={k:v.copy() for k,v in self.p.items()};self.probe['output_raw'][:]=.005
        self.rng=np.random.default_rng(181);self.x=self.rng.normal(size=55).astype(np.float32);self.n=self.rng.normal(size=55).astype(np.float32)
    def test_initial_exact_zero(self):
        f,a,d=run.causal_features(self.x,self.n,self.x,self.n,0)
        np.testing.assert_array_equal(f,np.zeros(150));self.assertEqual(a,0)
        e,h,m,z=run.actor_step(self.probe,f,np.zeros(8),a,np.ones(14));np.testing.assert_array_equal(e,np.zeros(14))
    def test_nominal_exact_zero_sequence(self):
        h=np.zeros(8)
        for k in range(529):
            x=self.rng.normal(size=55).astype(np.float32);f,a,d=run.causal_features(x,x,self.n,self.n,k)
            e,h,m,z=run.actor_step(self.probe,f,h,a,self.rng.normal(size=14));np.testing.assert_array_equal(e,np.zeros(14));np.testing.assert_array_equal(h,np.zeros(8))
    def test_zero_head_deterministic(self):
        f=self.rng.uniform(-1,1,150);e,h,m,z=run.actor_step(self.p,f,np.zeros(8),.5,np.zeros(14));np.testing.assert_array_equal(e,np.zeros(14))
    def test_bounded_output(self):
        for _ in range(30):
            p={k:self.rng.normal(0,10,v.shape) for k,v in self.p.items()};e,h,m,z=run.actor_step(p,self.rng.uniform(-1,1,150),self.rng.uniform(-1,1,8),1.,self.rng.normal(0,100,14));self.assertLessEqual(np.abs(e).max(),.18);self.assertLessEqual(np.abs(h).max(),1.)
    def test_recurrence_row_bound(self):
        a,b,c=run.matrices(self.probe);self.assertLess(np.abs(b).sum(1).max(),.5)
    def test_context_causal_interface(self):
        self.assertEqual(list(inspect.signature(run.causal_features).parameters),['current','nominal','initial','initial_nominal','control'])
        self.assertEqual(list(inspect.signature(run.actor_step).parameters),['p','features','state','amplitude','innovation'])
        source=inspect.getsource(run.causal_features)+inspect.getsource(run.actor_step)
        for token in ['qpos','qvel','case_seed','directory','success','labels']:self.assertNotIn(token,source)
    def test_invalid_actor(self):
        p={k:v.copy() for k,v in self.p.items()};p['input_raw'][0,0]=np.nan
        with self.assertRaises(ValueError):run.check_actor(p)
    def test_invalid_sensor(self):
        with self.assertRaises(ValueError):run.causal_features(self.x[:50],self.n,self.x,self.n,0)
        with self.assertRaises(ValueError):run.causal_features(self.x,self.n,self.x,self.n,529)
    def test_no_extra_target_preserves_identity(self):
        t=np.zeros(14);self.assertIs(run.local.merge_target(t,t,np.zeros(14),np.arange(14),-np.ones(14),np.ones(14)),t)
    def test_existing_cap_and_slew(self):
        base=np.zeros(14);target=np.ones(14)*.1
        merged=run.local.merge_target(target,base,np.ones(14)*.18,np.arange(14),-np.ones(14),np.ones(14));self.assertLessEqual(np.abs(merged).max(),.18)
        planned=run.planner.apply_limits(merged,np.zeros(14),-np.ones(14),np.ones(14));self.assertLessEqual(np.abs(planned).max(),5.24*.02)
    def test_iid_companion_baseline(self):
        r=np.array([[1.,2.,3.],[4.,5.,6.]])
        a=np.concatenate([r[0]-r[1],r[1]-r[0]]);np.testing.assert_array_equal(a[:3],-a[3:])
        # For independent unit-normal score and independent companion return,
        # E[(R1-R2)*score1] recovers the single-policy gradient.
        n=100000;u=self.rng.normal(size=n);v=self.rng.normal(size=n)
        self.assertAlmostEqual(float(np.mean((u-v)*u)),1.,delta=.02)
    def test_scalar_jax_means(self):
        import jax;jax.config.update('jax_enable_x64',True)
        import jax.numpy as jp
        fs=self.rng.uniform(-1,1,(12,150));h=np.zeros(8);means=[]
        for f in fs:
            _,h,m,_=run.actor_step(self.probe,f,h,.5,np.zeros(14));means.append(m)
        np.testing.assert_allclose(run.differentiable_means({k:jp.asarray(v) for k,v in self.probe.items()},jp.asarray(fs)),means,atol=1e-12,rtol=1e-12)
    def test_all_blocks_score_gradients_finite_difference(self):
        import jax;jax.config.update('jax_enable_x64',True)
        import jax.numpy as jp
        p={k:jp.asarray(v) for k,v in self.probe.items()};f=self.rng.uniform(-.5,.5,(2,7,150));amp=np.ones((2,7))*.5
        means=np.stack([run.differentiable_means(p,jp.asarray(v)) for v in f]);z=means+run.SIGMA*amp[...,None]*self.rng.normal(size=(2,7,14));adv=np.array([.7,-.7])
        args=[jp.asarray(v) for v in (f,z,amp,adv)];grad=jax.grad(run.score_loss)(p,*args)
        for key in p:
            ix=np.unravel_index(np.argmax(np.abs(np.asarray(grad[key]))),p[key].shape);eps=1e-7
            plus={k:np.asarray(v).copy() for k,v in p.items()};minus={k:np.asarray(v).copy() for k,v in p.items()};plus[key][ix]+=eps;minus[key][ix]-=eps
            finite=(float(run.score_loss(plus,*args))-float(run.score_loss(minus,*args)))/(2*eps)
            self.assertAlmostEqual(float(grad[key][ix]),finite,delta=max(1e-5,abs(finite)*1e-5));self.assertGreater(np.linalg.norm(np.asarray(grad[key])),0)
    def test_exact_zero_likelihood_omitted(self):
        import jax;jax.config.update('jax_enable_x64',True)
        import jax.numpy as jp
        p={k:jp.asarray(v) for k,v in self.p.items()};loss=run.score_loss(p,jp.zeros((2,4,150)),jp.zeros((2,4,14)),jp.zeros((2,4)),jp.ones(2))
        self.assertEqual(float(loss),0.)

if __name__=='__main__':unittest.main()
