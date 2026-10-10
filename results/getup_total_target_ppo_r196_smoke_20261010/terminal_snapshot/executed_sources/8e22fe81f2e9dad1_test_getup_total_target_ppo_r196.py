"""Offline causal/kernel/learner regressions; no environment integration."""
import ast
import os
from pathlib import Path
import unittest
import numpy as np
from diagnostics import train_getup_total_target_ppo_r196 as run
from diagnostics import total_target_ppo_kernel_r196 as k


class Regression(unittest.TestCase):
    def setUp(self):
        self.w=k.initial_weights();self.x=k.features(np.arange(50)/50,np.zeros(50),17)

    def test_fresh_actor_zero(self):
        np.testing.assert_array_equal(k.network(self.w,self.x,'actor'),np.zeros(10))
        self.assertTrue(np.any(self.w['actor_w0']))

    def test_feature_layout(self):
        self.assertEqual(self.x.shape,(101,));self.assertEqual(self.x[-1],17/528)
        np.testing.assert_array_equal(self.x[:50],np.tanh(np.arange(50)/50))

    def test_native_history_changes_feature(self):
        y=k.features(np.arange(50)/50,np.ones(50),17)
        np.testing.assert_array_equal(y[:50],self.x[:50]);self.assertTrue(np.any(y[50:100]!=self.x[50:100]))

    def test_illegal_sensor(self):
        for a in (np.zeros(49),np.full(50,np.nan)):
            with self.assertRaises(ValueError):k.features(a,np.zeros(50),1)

    def test_illegal_clock(self):
        for n in (-1,2280,1.5):
            with self.assertRaises(ValueError):k.features(np.zeros(50),np.zeros(50),n)

    def test_phase_home_cap(self):
        self.assertEqual(k.features(np.zeros(50),np.zeros(50),2279)[-1],1.)

    def test_object_bypass(self):
        fixed=np.arange(14)/10;ids=np.arange(10)
        self.assertIs(k.replace_legs(fixed,np.zeros(14),np.zeros(10),ids,-np.ones(14),np.ones(14),False),fixed)

    def test_replaces_not_adds(self):
        fixed=np.full(14,.12);ref=np.zeros(14);ids=np.arange(10)
        adjusted=k.replace_legs(fixed,ref,np.zeros(10),ids,-np.ones(14),np.ones(14),True)
        np.testing.assert_array_equal(adjusted[:10],np.zeros(10));np.testing.assert_array_equal(adjusted[10:],fixed[10:])

    def test_bound_and_joint(self):
        a=k.replace_legs(np.zeros(14),np.zeros(14),np.full(10,1e6),np.arange(10),np.full(14,-.1),np.full(14,.1),True)
        np.testing.assert_array_equal(a[:10],np.full(10,.1));self.assertLessEqual(np.abs(a).max(),k.BOUND)

    def test_actor_sample_likelihood(self):
        noise=np.arange(10)/10;z=k.SIGMA*noise
        expected=-.5*np.sum(noise**2+2*np.log(k.SIGMA)+np.log(2*np.pi))
        self.assertEqual(k.log_probability(z,np.zeros(10)),expected)

    def test_returns_full_tail(self):
        r=np.zeros(2279);r[-1]=10.
        g=k.complete_returns(r)
        self.assertAlmostEqual(g[0],10*k.GAMMA**2278,places=11);self.assertEqual(g[-1],10.)

    def test_returns_episode_boundary(self):
        np.testing.assert_array_equal(k.complete_returns([1.,2.]),np.array([1.+2*k.GAMMA,2.]))

    def test_gaussian_ratio_same(self):
        mean=k.network(self.w,self.x,'actor');z=mean+k.SIGMA*np.ones(10)
        self.assertEqual(np.exp(k.log_probability(z,mean)-k.log_probability(z,mean)),1.)

    def test_jax_numpy_and_gradient(self):
        learner=k.Learner(self.w);jp=learner.jp
        x=np.stack([self.x,self.x+.01]);p=learner.weights
        np.testing.assert_allclose(k.network(p,jp.asarray(x),'critic',jp),k.network(self.w,x,'critic'),atol=1e-12,rtol=0)
        z=np.full((2,10),k.SIGMA*.5);old=k.log_probability(z,np.zeros((2,10)))
        args=[x,z,old,np.array([1.,-.5]),np.array([2.,-1.]),np.ones(2)]
        (_,s),g=learner.gradient(p,*[jp.asarray(a) for a in args]);self.assertTrue(np.isfinite(np.asarray(s)).all())
        self.assertGreater(np.linalg.norm(g['actor_w2']),0)
        self.assertEqual(float(np.linalg.norm(g['actor_w0'])),0.)
        eps=1e-8;plus={a:np.array(b) for a,b in self.w.items()};minus={a:np.array(b) for a,b in self.w.items()}
        plus['actor_b2'][0]+=eps;minus['actor_b2'][0]-=eps
        f=lambda q:float(learner.objective(q,*[jp.asarray(a) for a in args])[0])
        self.assertAlmostEqual((f(plus)-f(minus))/(2*eps),float(g['actor_b2'][0]),delta=.01)

    def test_home_mask_no_actor_gradient(self):
        learner=k.Learner(self.w);jp=learner.jp;x=np.stack([self.x]);z=np.zeros((1,10))
        (_,s),g=learner.gradient(learner.weights,*[jp.asarray(a) for a in (x,z,np.zeros(1),np.ones(1),np.ones(1),np.zeros(1))])
        for name,v in g.items():
            if name.startswith('actor'):np.testing.assert_array_equal(v,np.zeros_like(v))

    def test_gate_null_standard_allowed(self):
        def report(cases):
            rows=[dict(controls=2279,complete_R157_parity={'a':True}) for _ in cases]
            return dict(physical_failures=0,nominal_success=True if None in cases else None,rows=rows)
        reports={name:report(cases) for name,mode,cases,parity in run.SMOKE_SPEC}
        self.assertTrue(run.smoke_gate(reports))
        reports['initial_standard']['nominal_success']=False;self.assertFalse(run.smoke_gate(reports))

    def test_invalid_smoke_blocks(self):
        reports={name:dict(physical_failures=0,nominal_success=True if None in cases else None,rows=[dict(controls=2279,complete_R157_parity={'a':True}) for _ in cases]) for name,mode,cases,parity in run.SMOKE_SPEC}
        reports['initial_sample']['physical_failures']=1;self.assertFalse(run.smoke_gate(reports))

    def test_fixed_budget(self):
        self.assertEqual(6+6+25+2*25+2*25+25,162)
        self.assertEqual(k.EPOCHS,4);self.assertEqual(k.LEARNING_RATE,1e-6)

    def test_scalar_original_limits(self):
        lo=np.full(14,-1.);hi=np.full(14,1.);prev=np.zeros(14);target=np.full(14,.18)
        a=run.planner.apply_limits(target,prev,lo,hi)
        self.assertLessEqual(np.abs(a-prev).max(),5.24*.02+1e-12)

    def test_kernel_no_simulator_or_metadata(self):
        tree=ast.parse(Path(k.__file__).read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(s and s.startswith('diagnostics') for s in imports))
        self.assertNotIn('mujoco',Path(k.__file__).read_text())
        self.assertNotIn('case_seed',Path(k.__file__).read_text())


if __name__=='__main__':
    where=os.environ.get('OPEN_DUCK_R196_TEST_CAPTURE')
    if where:
        destination=Path(where);run.capture_sources(destination)
    unittest.main()
