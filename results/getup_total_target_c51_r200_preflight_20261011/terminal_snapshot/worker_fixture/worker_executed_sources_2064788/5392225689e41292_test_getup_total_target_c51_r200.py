"""Offline categorical-learning and unchanged causal-control interface regressions."""
import ast
import os
from pathlib import Path
import shutil
import unittest
import numpy as np
import mujoco
from diagnostics import train_getup_total_target_c51_r200 as run
from diagnostics import total_target_c51_kernel_r200 as k

def independent_projection(r,boot,tail,p):
    out=np.zeros_like(p)
    for row in range(len(r)):
        for j,z in enumerate(k.SUPPORT):
            value=float(np.clip(r[row]+k.GAMMA*(tail[row]+boot[row]*z),k.VMIN,k.VMAX))
            x=(value-k.VMIN)/((k.VMAX-k.VMIN)/(k.ATOMS-1))
            lower=int(np.floor(x));upper=int(np.ceil(x))
            if lower==upper:out[row,lower]+=p[row,j]
            else:
                out[row,lower]+=p[row,j]*(upper-x)
                out[row,upper]+=p[row,j]*(x-lower)
    return out

class Regression(unittest.TestCase):
    def setUp(self):
        self.w=k.initial_weights();self.x=k.features(np.arange(50)/50,np.zeros(50),17)

    def test_01_fresh_initial_exact_tie_zero_code(self):
        rng=np.random.default_rng(300)
        np.testing.assert_array_equal(self.w['value_w0'],rng.normal(0,1/np.sqrt(101),(101,32)))
        self.assertTrue(np.any(self.w['value_w1']));self.assertEqual(len(self.w),6)
        q,p=k.values(self.w,self.x)
        np.testing.assert_array_equal(p,np.full((21,51),1/51))
        np.testing.assert_array_equal(q,np.full(21,q[0]))
        d=k.select(self.w,self.x,rng,False);self.assertEqual(d['chosen'],0)
        np.testing.assert_array_equal(d['action'],np.zeros(10))

    def test_02_features_and_causal_initial(self):
        np.testing.assert_array_equal(self.x[:50],np.tanh(np.arange(50)/50))
        np.testing.assert_array_equal(self.x[50:100],np.zeros(50));self.assertEqual(self.x[-1],17/528)
        other=k.features(np.arange(50)/50,np.ones(50),17)
        np.testing.assert_array_equal(other[:50],self.x[:50]);self.assertTrue(np.any(other[50:100]))
        for clock,value in ((0,0),(528,1),(529,1),(2279,1)):self.assertEqual(k.features(np.zeros(50),np.zeros(50),clock)[-1],value)

    def test_03_invalid_causal_inputs(self):
        for sensor in (np.zeros(49),np.zeros(55),np.full(50,np.nan)):
            with self.assertRaises(ValueError):k.features(sensor,np.zeros(50),1)
        for clock in (-1,2280,1.5):
            with self.assertRaises(ValueError):k.features(np.zeros(50),np.zeros(50),clock)

    def test_04_fixed_action_book_not_parameter_proposals(self):
        book=k.codebook();self.assertEqual(book.shape,(21,10))
        np.testing.assert_array_equal(book[0],np.zeros(10))
        for j in range(10):
            np.testing.assert_array_equal(book[1+2*j],np.eye(10)[j]*1e-4)
            np.testing.assert_array_equal(book[2+2*j],-np.eye(10)[j]*1e-4)

    def test_05_sample_two_draws_exact_rng_order(self):
        a=np.random.default_rng(30001);b=np.random.default_rng(30001)
        for _ in range(200):
            d=k.select(self.w,self.x,a,True);gate=b.random();draw=int(b.integers(21))
            self.assertEqual(d['exploration_gate'],gate);self.assertEqual(d['exploration_code'],draw)
            self.assertEqual(d['chosen'],draw if gate<.1 else 0)
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)

    def test_06_deterministic_rng_not_consumed_and_state_dependent_choice(self):
        rng=np.random.default_rng(300);before=rng.bit_generator.state
        k.select(self.w,self.x,rng,False);self.assertEqual(before,rng.bit_generator.state)
        self.w['value_w2'][:,51:102]=np.linspace(-.1,.1,51)[None,:]
        q1,_=k.values(self.w,self.x);q2,_=k.values(self.w,self.x+.2)
        self.assertTrue(np.any(q1!=q2))
        self.w['value_b2'][3*51+50]=20
        self.assertEqual(k.select(self.w,self.x,rng,False)['chosen'],3)

    def test_07_stable_extreme_softmax(self):
        logits=np.linspace(-10000,10000,1071).reshape(21,51)
        p=k.probabilities(logits);self.assertTrue(np.isfinite(p).all())
        np.testing.assert_allclose(p.sum(-1),1.,atol=1e-15,rtol=0)

    def test_08_inactive_original_object_and_leg_replacement(self):
        fixed=np.full(14,.12);legs=np.arange(10);ref=np.zeros(14);lo=-np.ones(14);hi=np.ones(14)
        self.assertIs(k.replace_legs(fixed,None,None,None,None,None,False),fixed)
        out=k.replace_legs(fixed,ref,k.codebook()[1],legs,lo,hi,True)
        np.testing.assert_array_equal(out[:10],.18*k.codebook()[1]);np.testing.assert_array_equal(out[10:],fixed[10:])
        np.testing.assert_array_equal(k.replace_legs(fixed,ref,k.codebook()[0],legs,lo,hi,True)[:10],np.zeros(10))
        for bad in (np.ones(10),np.full(10,np.nan),np.zeros(9)):
            with self.assertRaises(ValueError):k.replace_legs(fixed,ref,bad,legs,lo,hi,True)

    def test_09_static_actual_model_mapping_and_joint_slew(self):
        model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
        self.assertEqual(run.utility.model_digest(model),'53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3')
        m=run.utility.kernel.Mapping(model);lo,hi=model.jnt_range[model.actuator_trnid[:,0]].T
        ref=(lo+hi)/2;fixed=ref.copy();fixed[m.head]+=.01
        for code in k.codebook():
            out=k.replace_legs(fixed,ref,code,m.legs,lo,hi,True)
            np.testing.assert_array_equal(out[m.head],fixed[m.head])
            np.testing.assert_array_equal(out[m.legs],np.clip(ref[m.legs]+.18*code,lo[m.legs],hi[m.legs]))
            plan=run.planner.apply_limits(out,ref,lo,hi)
            self.assertLessEqual(np.max(np.abs(plan-ref)),5.24*.02+1e-12)

    def test_10_actual_native_full_original_planning_fixture(self):
        path=run.ROOT/'outputs/getup_parameter_pair_r198_smoke_20261011/zero_parity/case_769000/trajectory.npz'
        model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE));lo,hi=model.jnt_range[model.actuator_trnid[:,0]].T
        with np.load(path,allow_pickle=False) as z:
            obs=z['observations'];fixed=z['original_double_pre_slew_target_rad'];applied=z['applied']
            self.assertEqual(obs.shape,(2279,55))
            for key in ('construction_preparation_sensors','preparation_sensors'):self.assertEqual(z[key].shape,(40,50))
            for c in (0,1,528,529,2278):
                self.assertEqual(k.select(self.w,k.features(obs[c,:50],obs[0,:50],c),np.random.default_rng(1),False)['chosen'],0)
            for c in range(1,2279):np.testing.assert_array_equal(run.planner.apply_limits(fixed[c],applied[c-1],lo,hi),applied[c])
        # Old native data is an interface fixture only, never new replay/supervision.

    def test_11_worker_frozen_loader_new_offline_fixture(self):
        where=os.environ.get('OPEN_DUCK_R200_TEST_CAPTURE')
        if not where:self.skipTest('New evidence destination required')
        parent=Path(where).parent/'worker_fixture';parent.mkdir();frozen=parent/'frozen';frozen.mkdir()
        old=run.ROOT/'outputs/getup_parameter_pair_r198_smoke_20261011/frozen'
        for name in ('snapshot.npz','initial_selector.npz','nominal_sensor_trajectory.npz'):shutil.copy2(old/name,frozen/name)
        run.init_worker(frozen)
        self.assertEqual(run.local.NOMINAL.shape,(2279,55));self.assertIn('feature_mode',run.local.WEIGHTS)

    def test_12_complete_original_returns(self):
        r=np.zeros(2279);r[-1]=10
        self.assertAlmostEqual(k.complete_returns(r)[0],10*k.GAMMA**2278,places=11)
        for r in (np.zeros(0),np.full(4,np.nan)):
            with self.assertRaises(ValueError):k.complete_returns(r)

    def test_13_invalid_terminal_no_cross_reset_bootstrap(self):
        p=k.episode_replay(np.zeros((4,50)),np.zeros(4,dtype=int),np.array([0.,1.,2.,-3.]),[False,True,True,True])
        np.testing.assert_array_equal(p['control'],[1,2,3]);np.testing.assert_array_equal(p['bootstrap'],[1,1,0])
        np.testing.assert_array_equal(p['terminal'],[False,False,True]);self.assertEqual(p['home_tail'][-1],0.)

    def test_14_home_backup_preserves_complete_original_tail(self):
        r=np.zeros(2279);r[529]=2;r[-1]=5
        p=k.episode_replay(np.zeros((2279,50)),np.zeros(2279,dtype=int),r,(np.arange(2279)>0)&(np.arange(2279)<529))
        self.assertEqual(len(p['reward']),528);self.assertEqual(p['bootstrap'][-1],0)
        self.assertEqual(p['home_tail'][-1],k.complete_returns(r)[529])
        self.assertEqual(p['reward'][-1]+k.GAMMA*p['home_tail'][-1],k.complete_returns(r)[528])

    def test_15_replay_invalid_mask_codes_and_reward(self):
        for active in (np.ones(4),np.zeros(3)):
            with self.assertRaises(ValueError):k.episode_replay(np.zeros((4,50)),np.zeros(4,dtype=int),np.ones(4),active)
        for codes in (np.zeros(4),np.full(4,21),np.full(4,-1)):
            with self.assertRaises(ValueError):k.episode_replay(np.zeros((4,50)),codes,np.ones(4),[False,True,True,True])

    def test_16_independent_general_projection_mass_and_clip_record(self):
        rng=np.random.default_rng(30016);p=rng.random((12,51));p/=p.sum(1)[:,None]
        r=np.linspace(-300,300,12);boot=np.tile([0.,1.],6);tail=np.linspace(-10,10,12)
        mass,raw,low,high=k.project(r,boot,tail,p)
        np.testing.assert_allclose(mass,independent_projection(r,boot,tail,p),atol=1e-14,rtol=0)
        np.testing.assert_allclose(mass.sum(1),1.,atol=2e-14,rtol=0)
        self.assertAlmostEqual(low[0],1.,places=14);self.assertAlmostEqual(high[-1],1.,places=14)
        np.testing.assert_array_equal(raw,r[:,None]+k.GAMMA*(tail[:,None]+boot[:,None]*k.SUPPORT))
        np.testing.assert_array_equal(r,np.linspace(-300,300,12))

    def test_17_exact_atom_half_atom_and_support_endpoints(self):
        p=np.full((5,51),1/51);r=np.array([k.VMIN,k.VMAX,0.,2.56,999.])
        mass,raw,low,high=k.project(r,np.zeros(5),np.zeros(5),p)
        np.testing.assert_allclose(mass,independent_projection(r,np.zeros(5),np.zeros(5),p),atol=1e-14,rtol=0)
        self.assertAlmostEqual(mass[0,0],1.,places=14);self.assertAlmostEqual(mass[1,-1],1.,places=14)
        self.assertAlmostEqual(mass[2,25],1.,places=14);self.assertAlmostEqual(mass[3,25],.5,places=14)
        self.assertAlmostEqual(mass[3,26],.5,places=14);self.assertEqual(low[0],0.);self.assertEqual(high[1],0.)
        self.assertAlmostEqual(high[-1],1.,places=14)

    def test_18_jax_numpy_network_and_projection_boundary(self):
        l=k.Learner(self.w);x=np.stack([self.x,self.x+.1])
        for n,j in zip(k.values(self.w,x),k.values(l.online,l.jp.asarray(x),l.jp)):
            np.testing.assert_allclose(n,j,atol=1e-12,rtol=0)
        r=np.array([-300.,1.]);boot=np.array([0.,1.]);tail=np.zeros(2);p=np.full((2,51),1/51)
        for n,j in zip(k.project(r,boot,tail,p),k.project(l.jp.asarray(r),l.jp.asarray(boot),l.jp.asarray(tail),l.jp.asarray(p),l.jp)):
            np.testing.assert_allclose(n,j,atol=1e-12,rtol=0)

    def test_19_frozen_target_greedy_and_stop_gradient(self):
        l=k.Learner(self.w);jp=l.jp;x=jp.asarray(self.x[None,:]);r=jp.ones(1);boot=jp.ones(1);tail=jp.zeros(1)
        mass,raw,low,high,greedy,prob=l.target_value(l.target,x,r,boot,tail)
        self.assertEqual(int(greedy[0]),0)
        np.testing.assert_allclose(mass,independent_projection(np.ones(1),np.ones(1),np.zeros(1),np.full((1,51),1/51)),atol=1e-14,rtol=0)
        g=l.jax.grad(lambda t:jp.sum(l.target_value(t,x,r,boot,tail)[0]))(l.target)
        for v in g.values():np.testing.assert_array_equal(v,np.zeros_like(v))
        self.assertFalse(hasattr(l,'actor'));self.assertFalse(hasattr(l,'target_actor'))

    def test_20_cross_entropy_gradient_and_unselected_codes(self):
        l=k.Learner(self.w);jp=l.jp;x=jp.asarray(self.x[None,:]);c=jp.array([3]);mass=np.zeros((1,51));mass[0,25]=1
        loss,g=l.gradient(l.online,x,c,jp.asarray(mass))
        self.assertAlmostEqual(float(loss),np.log(51),places=14)
        expected=np.zeros(1071);expected[3*51:4*51]=1/51;expected[3*51+25]-=1
        np.testing.assert_allclose(g['value_b2'],expected,atol=1e-15,rtol=0)
        plus={n:np.array(v) for n,v in l.online.items()};minus={n:v.copy() for n,v in plus.items()};eps=1e-6
        plus['value_b2'][3*51+25]+=eps;minus['value_b2'][3*51+25]-=eps
        fd=(float(l.objective(plus,x,c,jp.asarray(mass)))-float(l.objective(minus,x,c,jp.asarray(mass))))/(2*eps)
        self.assertAlmostEqual(fd,float(g['value_b2'][3*51+25]),delta=1e-8)

    def test_21_projected_target_is_stopped_in_loss(self):
        l=k.Learner(self.w);jp=l.jp
        g=l.jax.grad(lambda m:l.objective(l.online,jp.asarray(self.x[None,:]),jp.array([0]),m))(jp.full((1,51),1/51))
        np.testing.assert_array_equal(g,np.zeros((1,51)))

    def test_22_fixed_update_rng_hard_copy_full_checkpoint(self):
        l=k.Learner(self.w);rng=np.random.default_rng(300);independent=np.random.default_rng(300)
        p=k.episode_replay(np.zeros((4,50)),np.zeros(4,dtype=int),np.array([0.,1.,-2.,3.]),[False,True,True,True])
        out=l.update(p,rng);self.assertEqual((out['updates'],out['hard_target_copies'],l.count),(64,1,64))
        self.assertEqual(out['projected_masses'].shape,(64,512,51))
        for ix in out['replay_indices']:np.testing.assert_array_equal(ix,independent.integers(0,3,size=512))
        self.assertEqual(rng.bit_generator.state,independent.bit_generator.state)
        for n,v in l.online.items():np.testing.assert_array_equal(v,l.target[n])
        self.assertGreater(out['changes']['value_b2'],0.)
        where=os.environ.get('OPEN_DUCK_R200_TEST_CAPTURE')
        if where:
            path=Path(where).parent/'regression_checkpoint';l.checkpoint(path)
            data=l.serialization.msgpack_restore(path.with_suffix('.learner.msgpack').read_bytes())
            self.assertEqual(data['count'],64);self.assertEqual(set(data),{'online','target','state','count'})
            for n,v in l.online.items():
                np.testing.assert_array_equal(data['online'][n],v);np.testing.assert_array_equal(data['target'][n],v)
            run.local.write_json(path.with_suffix('.rng.json'),rng.bit_generator.state)

    def test_23_smoke_null_standard_and_invalid_block(self):
        reports={name:dict(physical_failures=0,nominal_success=True if None in cases else None,rows=[dict(controls=2279,complete_R157_parity={'a':True}) for c in cases]) for name,mode,cases,parity in run.SMOKE_SPEC}
        self.assertTrue(run.smoke_gate(reports));reports['initial_sample']['physical_failures']=1;self.assertFalse(run.smoke_gate(reports))

    def test_24_selection_no_success_union_fixed_budget(self):
        self.assertEqual(6+6+25+50+50+25,162);self.assertEqual((k.UPDATES,k.BATCH),(64,512))
        base=dict(successes=18,rows=[dict(case_seed=i,success=i<18) for i in range(24)])
        report=dict(successes=19,nominal_success=True,physical_failures=0,rows=[dict(case_seed=i,success=i<19) for i in range(24)])
        self.assertTrue(run.eligible(report,base));report['rows'][0]['success']=False;self.assertFalse(run.eligible(report,base))
        report['rows'][0]['success']=True;report['physical_failures']=1;self.assertFalse(run.eligible(report,base))

    def test_25_no_simulator_or_old_learner_import(self):
        source=Path(k.__file__).read_text();tree=ast.parse(source)
        self.assertNotIn('mujoco',source);self.assertNotIn('case_seed',source)
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('diagnostics') for n in ast.walk(tree)))
        runner=Path(run.__file__).read_text();self.assertNotIn('utility.init_worker(',runner)
        for text in ("active=mode!='baseline' and 0<k<529","if not active:assert adjusted is fixed","np.testing.assert_array_equal(applied,wanted)","np.testing.assert_array_equal(sim.prev,wanted)","OUTPUT,SMOKE,LAUNCH,PREFLIGHT"):
            self.assertIn(text,runner)

if __name__=='__main__':
    learning_runtime=k.Learner(k.initial_weights())
    where=os.environ.get('OPEN_DUCK_R200_TEST_CAPTURE')
    if where:
        destination=Path(where);manifest=run.capture_sources(destination)
        run.local.write_json(destination.parent/'source_manifest.json',manifest)
    unittest.main()

