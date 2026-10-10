"""Offline causal-interface and independently specified soft-learning regressions."""
import ast
import os
from pathlib import Path
import shutil
import unittest
import numpy as np
import mujoco
from diagnostics import train_getup_total_target_sac_r199 as run
from diagnostics import total_target_sac_kernel_r199 as k

class Regression(unittest.TestCase):
    def setUp(self):
        self.w=k.initial_weights();self.x=k.features(np.arange(50)/50,np.zeros(50),17)

    def test_01_fresh_seed_and_initial_distribution(self):
        rng=np.random.default_rng(299)
        np.testing.assert_array_equal(self.w['actor_w0'],rng.normal(0,1/np.sqrt(101),(101,32)))
        mu,ls=k.distribution(self.w,self.x)
        np.testing.assert_array_equal(mu,np.zeros(10));np.testing.assert_array_equal(ls,np.full(10,np.log(1e-4)))
        self.assertTrue(np.any(self.w['actor_w1']));self.assertFalse(np.array_equal(self.w['q1_w0'],self.w['q2_w0']))

    def test_02_features_actual_layout(self):
        np.testing.assert_array_equal(self.x[:50],np.tanh(np.arange(50)/50))
        np.testing.assert_array_equal(self.x[50:100],np.zeros(50));self.assertEqual(self.x[-1],17/528)

    def test_03_causal_initial_not_metadata(self):
        other=k.features(np.arange(50)/50,np.ones(50),17)
        np.testing.assert_array_equal(other[:50],self.x[:50]);self.assertTrue(np.any(other[50:100]!=self.x[50:100]))
        for sensor in (np.zeros(49),np.zeros(55),np.full(50,np.nan)):
            with self.assertRaises(ValueError):k.features(sensor,np.zeros(50),1)
        for clock in (-1,2280,1.5):
            with self.assertRaises(ValueError):k.features(np.zeros(50),np.zeros(50),clock)

    def test_04_clock_boundaries(self):
        for clock,value in ((0,0),(528,1),(529,1),(2279,1)):self.assertEqual(k.features(np.zeros(50),np.zeros(50),clock)[-1],value)

    def test_05_squash_and_density_independent_formula(self):
        mu=np.linspace(-.4,.4,10);ls=np.linspace(-10,-8,10);noise=np.linspace(-2,2,10)
        a,z,lp=k.squashed(mu,ls,noise)
        np.testing.assert_array_equal(z,mu+np.exp(ls)*noise);np.testing.assert_array_equal(a,np.tanh(z))
        expected=np.sum(-.5*(noise**2+np.log(2*np.pi))-ls-np.log(1-a*a))
        self.assertAlmostEqual(lp,expected,places=12)

    def test_06_extreme_latent_stable_not_executed_density(self):
        a,z,lp=k.squashed(np.full(10,1000.),np.full(10,-9.),np.zeros(10))
        self.assertTrue(np.isfinite(lp));np.testing.assert_array_equal(a,np.ones(10))
        self.assertGreater(lp,0.) # Finite stable density evaluation, not physical safety.

    def test_07_state_dependent_logstd_and_bounds(self):
        self.w['actor_w2'][:,10:]=.1
        _,a=k.distribution(self.w,self.x);_,b=k.distribution(self.w,self.x+.1)
        self.assertTrue(np.any(a!=b));self.w['actor_b2'][10:]=100
        np.testing.assert_array_equal(k.distribution(self.w,self.x)[1],np.full(10,-4.))
        self.w['actor_b2'][10:]=-100
        np.testing.assert_array_equal(k.distribution(self.w,self.x)[1],np.full(10,-12.))

    def test_08_inactive_original_object(self):
        fixed=np.arange(14)/100
        self.assertIs(k.replace_legs(fixed,None,None,None,None,None,False),fixed)

    def test_09_leg_replacement_head_joint_not_second_tanh(self):
        fixed=np.full(14,.12);legs=np.arange(10)
        out=k.replace_legs(fixed,np.zeros(14),np.ones(10),legs,-np.ones(14),np.ones(14),True)
        np.testing.assert_array_equal(out[:10],np.full(10,.18));np.testing.assert_array_equal(out[10:],fixed[10:])
        np.testing.assert_array_equal(k.replace_legs(fixed,np.zeros(14),np.zeros(10),legs,-np.ones(14),np.ones(14),True)[:10],np.zeros(10))
        np.testing.assert_array_equal(k.replace_legs(fixed,np.zeros(14),np.ones(10),legs,np.full(14,-.1),np.full(14,.1),True)[:10],np.full(10,.1))
        for a in (np.full(10,1.01),np.full(10,np.nan)):
            with self.assertRaises(ValueError):k.replace_legs(fixed,np.zeros(14),a,legs,-np.ones(14),np.ones(14),True)

    def test_10_static_actual_model_mapping_and_slew(self):
        model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
        self.assertEqual(run.utility.model_digest(model),'53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3')
        m=run.utility.kernel.Mapping(model);lo,hi=model.jnt_range[model.actuator_trnid[:,0]].T
        ref=(lo+hi)/2;fixed=ref.copy();fixed[m.head]+=.01
        out=k.replace_legs(fixed,ref,np.ones(10),m.legs,lo,hi,True)
        np.testing.assert_array_equal(out[m.head],fixed[m.head])
        np.testing.assert_array_equal(out[m.legs],np.clip(ref[m.legs]+.18,lo[m.legs],hi[m.legs]))
        plan=run.planner.apply_limits(out,ref,lo,hi);self.assertLessEqual(np.max(np.abs(plan-ref)),5.24*.02+1e-12)

    def test_11_actual_native_fixture_and_original_planning_boundary(self):
        path=run.ROOT/'outputs/getup_parameter_pair_r198_smoke_20261011/zero_parity/case_769000/trajectory.npz'
        model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE));lo,hi=model.jnt_range[model.actuator_trnid[:,0]].T
        with np.load(path,allow_pickle=False) as z:
            obs=z['observations'];fixed=z['original_double_pre_slew_target_rad'];applied=z['applied']
            self.assertEqual(obs.shape,(2279,55));self.assertEqual(z['construction_preparation_sensors'].shape,(40,50));self.assertEqual(z['preparation_sensors'].shape,(40,50))
            for c in (0,1,528,529,2278):
                x=k.features(obs[c,:50],obs[0,:50],c);mu,ls=k.distribution(self.w,x)
                np.testing.assert_array_equal(np.tanh(mu),np.zeros(10))
            for c in range(1,2279):
                np.testing.assert_array_equal(run.planner.apply_limits(fixed[c],applied[c-1],lo,hi),applied[c])
        # This old record is an interface fixture only, never SAC replay or supervision.

    def test_12_worker_frozen_loader_offline_new_fixture(self):
        where=os.environ.get('OPEN_DUCK_R199_TEST_CAPTURE')
        if not where:self.skipTest('New evidence destination required for worker capture fixture')
        parent=Path(where).parent/'worker_fixture';parent.mkdir();frozen=parent/'frozen';frozen.mkdir()
        old=run.ROOT/'outputs/getup_parameter_pair_r198_smoke_20261011/frozen'
        for name in ('snapshot.npz','initial_selector.npz','nominal_sensor_trajectory.npz'):shutil.copy2(old/name,frozen/name)
        run.init_worker(frozen)
        self.assertEqual(run.local.NOMINAL.shape,(2279,55));self.assertIn('feature_mode',run.local.WEIGHTS)

    def test_13_complete_original_returns(self):
        r=np.zeros(2279);r[-1]=10
        self.assertAlmostEqual(k.complete_returns(r)[0],10*k.GAMMA**2278,places=11)
        for rewards in (np.zeros(0),np.full(4,np.nan)):
            with self.assertRaises(ValueError):k.complete_returns(rewards)

    def test_14_terminal_no_reset_bootstrap_or_entropy(self):
        p=k.episode_replay(np.zeros((4,50)),np.zeros((4,10)),np.array([0.,1.,2.,-3.]),[False,True,True,True])
        np.testing.assert_array_equal(p['control'],[1,2,3]);np.testing.assert_array_equal(p['bootstrap'],[1,1,0]);np.testing.assert_array_equal(p['terminal'],[False,False,True])
        self.assertEqual(k.soft_bellman(-3.,0.,0.,999.,999.,-123.,.5),-3.)

    def test_15_home_tail_original_without_policy_entropy(self):
        r=np.zeros(2279);r[529]=2;r[-1]=5
        p=k.episode_replay(np.zeros((2279,50)),np.zeros((2279,10)),r,(np.arange(2279)>0)&(np.arange(2279)<529))
        self.assertEqual(len(p['reward']),528);self.assertEqual(p['bootstrap'][-1],0)
        self.assertEqual(p['home_tail'][-1],k.complete_returns(r)[529])
        self.assertEqual(k.soft_bellman(p['reward'][-1],0.,p['home_tail'][-1],999.,999.,-100.,.5),k.complete_returns(r)[528])

    def test_16_mask_alignment_nonfinite_and_bounds(self):
        for active in (np.ones(4),np.zeros(3)):
            with self.assertRaises(ValueError):k.episode_replay(np.zeros((4,50)),np.zeros((4,10)),np.ones(4),active)
        with self.assertRaises(ValueError):k.episode_replay(np.zeros((4,50)),np.ones((4,10))*2,np.ones(4),[False,True,True,True])

    def test_17_soft_backup_entropy_changes_only_bootstrap(self):
        y=k.soft_bellman(np.array([1.,2.]),np.array([1.,0.]),np.array([0.,3.]),np.array([4.,8.]),np.array([6.,7.]),np.array([-2.,-99.]),.1)
        np.testing.assert_array_equal(y,np.array([1.+k.GAMMA*4.2,2.+k.GAMMA*3]))

    def test_18_jax_numpy_distribution_and_logdensity(self):
        l=k.Learner(self.w);x=np.stack([self.x,self.x+.1]);noise=np.ones((2,10))
        mu,ls=k.distribution(self.w,x);a,z,lp=k.squashed(mu,ls,noise)
        jmu,jls=k.distribution(l.actor,l.jp.asarray(x),l.jp);ja,jz,jlp=k.squashed(jmu,jls,l.jp.asarray(noise),l.jp)
        for n,j in ((mu,jmu),(ls,jls),(a,ja),(z,jz),(lp,jlp)):np.testing.assert_allclose(n,j,atol=1e-12,rtol=0)

    def test_19_target_stop_gradient_current_actor_no_target_actor(self):
        l=k.Learner(self.w);jp=l.jp;x=jp.asarray(np.stack([self.x]));noise=jp.ones((1,10));r=jp.ones(1);boot=jp.ones(1);tail=jp.zeros(1)
        y=l.target_value(l.actor,l.target_critics,l.log_alpha,x,noise,r,boot,tail)
        mu,ls=k.distribution(self.w,np.asarray(x));a,_,lp=k.squashed(mu,ls,np.ones((1,10)))
        inp=np.c_[np.asarray(x),a]
        expected=k.soft_bellman(np.ones(1),np.ones(1),np.zeros(1),k.network(self.w,inp,'q1')[:,0],k.network(self.w,inp,'q2')[:,0],lp,1e-4)
        np.testing.assert_allclose(y,expected,atol=1e-12,rtol=0)
        grad=l.jax.grad(lambda p:jp.sum(l.target_value(p,l.target_critics,l.log_alpha,x,noise,r,boot,tail)))(l.actor)
        for v in grad.values():np.testing.assert_array_equal(v,np.zeros_like(v))
        self.assertFalse(hasattr(l,'target_actor'))

    def test_20_reparameterized_mu_and_logstd_gradients(self):
        l=k.Learner(self.w);x=l.jp.asarray(np.stack([self.x]));noise=l.jp.asarray(np.linspace(-1,1,10)[None,:])
        (_,s),g=l.actor_gradient(l.actor,l.critics,l.log_alpha,x,noise)
        self.assertTrue(np.isfinite(s).all());self.assertGreater(np.linalg.norm(g['actor_b2'][:10]),0.);self.assertGreater(np.linalg.norm(g['actor_b2'][10:]),0.)
        for index in (0,10):
            plus={n:np.array(v) for n,v in l.actor.items()};minus={n:v.copy() for n,v in plus.items()};eps=1e-6
            plus['actor_b2'][index]+=eps;minus['actor_b2'][index]-=eps
            diff=(float(l.actor_objective(plus,l.critics,l.log_alpha,x,noise)[0])-float(l.actor_objective(minus,l.critics,l.log_alpha,x,noise)[0]))/(2*eps)
            self.assertAlmostEqual(diff,float(g['actor_b2'][index]),delta=1e-7)

    def test_21_alpha_dual_direction_and_stop_policy_gradient(self):
        l=k.Learner(self.w);jp=l.jp
        _,low=l.alpha_gradient(l.log_alpha,jp.full(4,80.)) # entropy -80 < -75: increase alpha under descent.
        _,high=l.alpha_gradient(l.log_alpha,jp.full(4,70.))
        self.assertLess(float(low),0.);self.assertGreater(float(high),0.)
        grad=l.jax.grad(lambda lp:l.alpha_objective(l.log_alpha,lp))(jp.full(4,80.))
        np.testing.assert_array_equal(grad,np.zeros(4))

    def test_22_critic_gradient_does_not_update_actor(self):
        l=k.Learner(self.w);jp=l.jp
        (_,s),g=l.critic_gradient(l.critics,jp.asarray(self.x[None,:]),jp.zeros((1,10)),jp.ones(1))
        self.assertFalse(any(n.startswith('actor') for n in g));self.assertTrue(np.isfinite(s).all())
        np.testing.assert_array_equal(l.actor['actor_w2'],np.zeros((16,20)))

    def test_23_fixed_all_three_updates_targets_checkpoint_rng(self):
        l=k.Learner(self.w);rng=np.random.default_rng(299)
        p=k.episode_replay(np.zeros((4,50)),np.zeros((4,10)),np.array([0.,1.,-2.,3.]),[False,True,True,True])
        before={n:np.array(v) for n,v in l.target_critics.items()};out=l.update(p,rng)
        self.assertEqual((out['updates'],out['actor_updates'],out['alpha_updates'],l.count),(64,64,64,64))
        self.assertEqual(out['replay_indices'].shape,(64,512));self.assertEqual(out['next_policy_noise'].shape,(64,512,10));self.assertEqual(out['actor_policy_noise'].shape,(64,512,10))
        self.assertFalse(np.array_equal(out['next_policy_noise'],out['actor_policy_noise']))
        self.assertTrue(any(np.any(np.array(l.target_critics[n])!=v) for n,v in before.items()))
        self.assertGreater(np.linalg.norm(np.array(l.actor['actor_b2'])[10:]),0.)
        where=os.environ.get('OPEN_DUCK_R199_TEST_CAPTURE')
        if where:
            path=Path(where).parent/'regression_checkpoint';l.checkpoint(path)
            restored=l.serialization.msgpack_restore(path.with_suffix('.learner.msgpack').read_bytes())
            self.assertEqual(restored['count'],64);self.assertIn('alpha_state',restored);self.assertNotIn('target_actor',restored)
            for n,v in l.actor.items():np.testing.assert_array_equal(restored['actor'][n],v)
            self.assertEqual(float(restored['log_alpha']),float(l.log_alpha));run.local.write_json(path.with_suffix('.rng.json'),rng.bit_generator.state)

    def test_24_smoke_null_standard_and_invalid_block(self):
        reports={name:dict(physical_failures=0,nominal_success=True if None in cases else None,rows=[dict(controls=2279,complete_R157_parity={'a':True}) for c in cases]) for name,mode,cases,parity in run.SMOKE_SPEC}
        self.assertTrue(run.smoke_gate(reports));reports['initial_sample']['physical_failures']=1;self.assertFalse(run.smoke_gate(reports))

    def test_25_selection_no_success_union_and_budget(self):
        self.assertEqual(6+6+25+50+50+25,162);self.assertEqual((k.UPDATES,k.BATCH),(64,512))
        base=dict(successes=18,rows=[dict(case_seed=i,success=i<18) for i in range(24)])
        report=dict(successes=19,nominal_success=True,physical_failures=0,rows=[dict(case_seed=i,success=i<19) for i in range(24)])
        self.assertTrue(run.eligible(report,base));report['rows'][0]['success']=False;self.assertFalse(run.eligible(report,base))
        report['rows'][0]['success']=True;report['physical_failures']=1;self.assertFalse(run.eligible(report,base))

    def test_26_no_simulator_or_old_learner_import(self):
        source=Path(k.__file__).read_text();tree=ast.parse(source)
        self.assertNotIn('mujoco',source);self.assertNotIn('case_seed',source)
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('diagnostics') for n in ast.walk(tree)))
        runner=Path(run.__file__).read_text();self.assertNotIn('utility.init_worker(',runner)
        for text in ("active=mode!='baseline' and 0<k<529","if not active:assert adjusted is fixed","np.testing.assert_array_equal(applied,wanted)","np.testing.assert_array_equal(sim.prev,wanted)","OUTPUT,SMOKE,LAUNCH,PREFLIGHT"):
            self.assertIn(text,runner)

if __name__=='__main__':
    learning_runtime=k.Learner(k.initial_weights())
    where=os.environ.get('OPEN_DUCK_R199_TEST_CAPTURE')
    if where:
        destination=Path(where);manifest=run.capture_sources(destination)
        run.local.write_json(destination.parent/'source_manifest.json',manifest)
    unittest.main()
