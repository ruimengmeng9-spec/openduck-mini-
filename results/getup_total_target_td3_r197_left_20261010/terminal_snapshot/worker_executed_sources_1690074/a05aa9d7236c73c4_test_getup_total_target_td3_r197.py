"""Offline regressions for causal action, terminal backup, and fresh replay learner."""
import ast
import os
from pathlib import Path
import unittest
import numpy as np
from diagnostics import train_getup_total_target_td3_r197 as run
from diagnostics import total_target_td3_kernel_r197 as k

class Regression(unittest.TestCase):
    def setUp(self):
        self.w=k.initial_weights();self.x=k.features(np.arange(50)/50,np.zeros(50),17)

    def test_fresh_actor_zero_twin_q_independent(self):
        np.testing.assert_array_equal(k.network(self.w,self.x,'actor'),np.zeros(10))
        self.assertTrue(np.any(self.w['actor_w0']))
        self.assertFalse(np.array_equal(self.w['q1_w0'],self.w['q2_w0']))
        self.assertTrue(np.any(self.w['q1_w2']))

    def test_feature_layout(self):
        self.assertEqual(self.x.shape,(101,));self.assertEqual(self.x[-1],17/528)
        np.testing.assert_array_equal(self.x[:50],np.tanh(np.arange(50)/50))

    def test_initial_only_causal_history(self):
        y=k.features(np.arange(50)/50,np.ones(50),17)
        np.testing.assert_array_equal(y[:50],self.x[:50])
        self.assertTrue(np.any(y[50:100]!=self.x[50:100]))

    def test_illegal_sensor_clock(self):
        for a in (np.zeros(49),np.full(50,np.nan)):
            with self.assertRaises(ValueError):k.features(a,np.zeros(50),1)
        for c in (-1,2280,1.5):
            with self.assertRaises(ValueError):k.features(np.zeros(50),np.zeros(50),c)

    def test_phase_home_cap(self):
        self.assertEqual(k.features(np.zeros(50),np.zeros(50),2279)[-1],1.)

    def test_object_bypass(self):
        fixed=np.arange(14)/10
        self.assertIs(k.replace_legs(fixed,np.zeros(14),np.zeros(10),np.arange(10),-np.ones(14),np.ones(14),False),fixed)

    def test_replaces_not_adds_and_head_unchanged(self):
        fixed=np.full(14,.12)
        out=k.replace_legs(fixed,np.zeros(14),np.zeros(10),np.arange(10),-np.ones(14),np.ones(14),True)
        np.testing.assert_array_equal(out[:10],np.zeros(10));np.testing.assert_array_equal(out[10:],fixed[10:])

    def test_action_bound_joint_no_extra_tanh(self):
        out=k.replace_legs(np.zeros(14),np.zeros(14),np.ones(10),np.arange(10),-np.ones(14),np.ones(14),True)
        np.testing.assert_array_equal(out[:10],np.full(10,.18))
        out=k.replace_legs(np.zeros(14),np.zeros(14),np.ones(10),np.arange(10),np.full(14,-.1),np.full(14,.1),True)
        np.testing.assert_array_equal(out[:10],np.full(10,.1))
        with self.assertRaises(ValueError):k.replace_legs(np.zeros(14),np.zeros(14),np.full(10,1.01),np.arange(10),-np.ones(14),np.ones(14),True)

    def test_clipped_exploration_not_executed_gaussian(self):
        np.testing.assert_array_equal(k.sample_action(np.ones(10),np.ones(10)),np.ones(10))
        np.testing.assert_array_equal(k.sample_action(np.zeros(10),np.ones(10)),np.full(10,k.SIGMA))

    def test_returns_full_original_home_tail(self):
        r=np.zeros(2279);r[-1]=10
        g=k.complete_returns(r)
        self.assertAlmostEqual(g[0],10*k.GAMMA**2278,places=11);self.assertEqual(g[-1],10)

    def test_terminal_no_reset_bootstrap(self):
        obs=np.zeros((4,50));a=np.zeros((4,10));r=np.array([1.,2.,3.,4.])
        p=k.episode_replay(obs,a,r,[False,True,True,True])
        np.testing.assert_array_equal(p['control'],[1,2,3])
        np.testing.assert_array_equal(p['bootstrap'],[1,1,0])
        np.testing.assert_array_equal(p['home_tail'],np.zeros(3))
        np.testing.assert_array_equal(p['terminal'],[False,False,True])
        self.assertEqual(k.bellman(4.,0.,0.,999.,999.),4.)

    def test_home_boundary_full_recorded_tail(self):
        n=2279;obs=np.zeros((n,50));r=np.zeros(n);r[529]=2;r[-1]=5
        p=k.episode_replay(obs,np.zeros((n,10)),r,(np.arange(n)>0)&(np.arange(n)<529))
        self.assertEqual(len(p['reward']),528);self.assertEqual(p['control'][-1],528)
        self.assertEqual(p['bootstrap'][-1],0)
        self.assertEqual(p['home_tail'][-1],k.complete_returns(r)[529])
        self.assertEqual(k.bellman(p['reward'][-1],0.,p['home_tail'][-1],999.,999.),k.complete_returns(r)[528])

    def test_replay_alignment_and_mask_rejects_home_actor(self):
        with self.assertRaises(ValueError):k.episode_replay(np.zeros((4,50)),np.zeros((4,10)),np.ones(4),np.ones(4))

    def test_twin_min_backup(self):
        expected=np.array([2.+k.GAMMA*3,4.])
        np.testing.assert_array_equal(k.bellman(np.array([2.,4.]),np.array([1.,0.]),np.zeros(2),np.array([3.,99.]),np.array([5.,88.])),expected)

    def test_jax_numpy_twin_and_bounded_actor(self):
        learner=k.Learner(self.w);jp=learner.jp
        x=np.stack([self.x,self.x+.01]);inp=np.c_[x,np.ones((2,10))*.1]
        for name in ('q1','q2'):
            np.testing.assert_allclose(k.network(learner.critics,jp.asarray(inp),name,jp),k.network(self.w,inp,name),atol=1e-12,rtol=0)
        self.assertLessEqual(np.abs(k.network(self.w,x,'actor')).max(),1.)

    def test_target_smoothing_clipping_and_stop_gradient(self):
        l=k.Learner(self.w);jp=l.jp;x=np.stack([self.x])
        noise=np.full((1,10),100.);r=np.array([2.]);boot=np.array([1.]);tail=np.array([0.])
        y=l.target_value(l.target_actor,l.target_critics,*[jp.asarray(v) for v in (x,noise,r,boot,tail)])
        inp=np.c_[x,np.full((1,10),k.TARGET_CLIP)]
        expected=k.bellman(r,boot,tail,k.network(self.w,inp,'q1')[:,0],k.network(self.w,inp,'q2')[:,0])
        np.testing.assert_allclose(y,expected,atol=1e-12,rtol=0)
        grad=l.jax.grad(lambda p:jp.sum(l.target_value(p,l.target_critics,*[jp.asarray(v) for v in (x,noise,r,boot,tail)])))(l.target_actor)
        for v in grad.values():np.testing.assert_array_equal(v,np.zeros_like(v))

    def test_actor_gradient_finite_difference(self):
        l=k.Learner(self.w);jp=l.jp;x=jp.asarray(np.stack([self.x]))
        loss,g=l.actor_gradient(l.actor,l.critics,x)
        self.assertTrue(np.isfinite(float(loss)));self.assertGreater(np.linalg.norm(g['actor_w2']),0.)
        plus={n:np.array(v) for n,v in l.actor.items()};minus={n:v.copy() for n,v in plus.items()}
        eps=1e-6;plus['actor_b2'][0]+=eps;minus['actor_b2'][0]-=eps
        diff=(float(l.actor_objective(plus,l.critics,x))-float(l.actor_objective(minus,l.critics,x)))/(2*eps)
        self.assertAlmostEqual(diff,float(g['actor_b2'][0]),delta=1e-7)

    def test_critic_gradient_no_actor_optimization(self):
        l=k.Learner(self.w);jp=l.jp;x=jp.asarray(np.stack([self.x]));a=jp.zeros((1,10));y=jp.ones(1)
        (_,s),g=l.critic_gradient(l.critics,x,a,y)
        self.assertTrue(np.isfinite(np.asarray(s)).all())
        self.assertFalse(any(n.startswith('actor') for n in g))
        np.testing.assert_array_equal(l.actor['actor_w2'],np.zeros((16,10)))

    def test_fixed_updates_delay_targets_and_checkpoint(self):
        l=k.Learner(self.w);rng=np.random.default_rng(297)
        p=k.episode_replay(np.zeros((4,50)),np.zeros((4,10)),np.array([0.,1.,-2.,3.]),[False,True,True,True])
        before={n:np.array(v) for n,v in l.target_actor.items()}
        out=l.update(p,rng)
        self.assertEqual(out['updates'],64);self.assertEqual(out['actor_updates'],32);self.assertEqual(l.count,64)
        np.testing.assert_array_equal(out['statistics'][:,4],np.arange(1,65)%2==0)
        self.assertEqual(out['replay_indices'].shape,(64,512))
        self.assertEqual(out['target_noise'].shape,(64,512,10))
        self.assertTrue(any(np.any(np.array(l.target_actor[n])!=v) for n,v in before.items()))
        where=os.environ.get('OPEN_DUCK_R197_TEST_CAPTURE')
        if where:
            path=Path(where).parent/'regression_checkpoint'
            l.checkpoint(path)
            restored=l.serialization.msgpack_restore(path.with_suffix('.learner.msgpack').read_bytes())
            self.assertEqual(restored['count'],64)
            for n,v in l.actor.items():np.testing.assert_array_equal(restored['actor'][n],v)
            run.local.write_json(path.with_suffix('.rng.json'),rng.bit_generator.state)

    def test_gate_null_standard_and_invalid_block(self):
        reports={name:dict(physical_failures=0,nominal_success=True if None in cases else None,rows=[dict(controls=2279,complete_R157_parity={'a':True}) for c in cases]) for name,mode,cases,parity in run.SMOKE_SPEC}
        self.assertTrue(run.smoke_gate(reports))
        reports['initial_sample']['physical_failures']=1;self.assertFalse(run.smoke_gate(reports))

    def test_original_limits_budget(self):
        a=run.planner.apply_limits(np.full(14,.18),np.zeros(14),-np.ones(14),np.ones(14))
        self.assertLessEqual(np.abs(a).max(),5.24*.02+1e-12)
        self.assertEqual(6+6+25+50+50+25,162)
        self.assertEqual((k.UPDATES,k.DELAY,k.BATCH),(64,2,512))

    def test_kernel_no_simulator_or_metadata(self):
        source=Path(k.__file__).read_text();tree=ast.parse(source)
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(s and s.startswith('diagnostics') for s in imports))
        self.assertNotIn('mujoco',source);self.assertNotIn('case_seed',source)

if __name__=='__main__':
    learning_runtime=k.Learner(k.initial_weights())
    where=os.environ.get('OPEN_DUCK_R197_TEST_CAPTURE')
    if where:
        destination=Path(where)
        manifest=run.capture_sources(destination)
        run.local.write_json(destination.parent/'source_manifest.json',manifest)
    unittest.main()

