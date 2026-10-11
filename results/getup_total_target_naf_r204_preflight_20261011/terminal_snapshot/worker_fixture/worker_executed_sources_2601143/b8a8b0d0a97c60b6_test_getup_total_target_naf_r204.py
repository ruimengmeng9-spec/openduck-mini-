"""New offline causal-control/episode/selection interface regressions; no new integration."""
import ast
import os
from pathlib import Path
import shutil
import unittest
import numpy as np
import mujoco
from diagnostics import train_getup_total_target_naf_r204 as run
from diagnostics import total_target_naf_kernel_r204 as k

class Regression(unittest.TestCase):
    def setUp(self):
        self.w=k.initial_weights()
    def test_01_immutable_numeric_kernel(self):
        self.assertEqual(run.prior.digest(k.__file__),run.KERNEL_SHA)
        self.assertNotIn('mujoco',Path(k.__file__).read_text())
        self.assertEqual((k.SEED,k.INPUTS,k.ACTIONS,k.OUTPUTS),(304,101,10,66))
        self.assertEqual((k.RATE,k.UPDATES,k.BATCH,k.TAU,k.SIGMA),(1e-4,64,512,.005,1e-4))
        self.assertEqual(sum(v.size for v in self.w.values()),4914)
    def test_02_actual_native_causal_layout(self):
        p=run.ROOT/'outputs/getup_parameter_pair_r198_smoke_20261011/zero_parity/case_769000/trajectory.npz'
        with np.load(p,allow_pickle=False) as z:
            obs=z['observations'];self.assertEqual(obs.shape,(2279,55))
            for c in (0,1,528,529,2278):
                x=k.features(obs[c,:50],obs[0,:50],c)
                np.testing.assert_array_equal(x[:50],np.tanh(obs[c,:50].astype(float)))
                np.testing.assert_array_equal(x[50:100],np.tanh(obs[0,:50].astype(float)))
                self.assertEqual(x[-1],min(c/528,1));self.assertEqual(x.shape,(101,))
                np.testing.assert_array_equal(k.request(self.w,x)['mean'],np.zeros(10))
                other=obs[c].copy();other[50:]=999
                np.testing.assert_array_equal(x,k.features(other[:50],obs[0,:50],c))
    def test_03_static_full_model_and_assets_unchanged(self):
        assets=run.utility.model_files(run.local.prior.program.SCENE)
        model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
        self.assertEqual(run.utility.model_digest(model),'53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3')
        self.assertEqual(assets,run.utility.model_files(run.local.prior.program.SCENE))
        self.assertTrue(assets)
    def test_04_actual_native_original_full_recursive_planning(self):
        p=run.ROOT/'outputs/getup_parameter_pair_r198_smoke_20261011/zero_parity/case_769000/trajectory.npz'
        model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
        lo,hi=model.jnt_range[model.actuator_trnid[:,0]].T
        with np.load(p,allow_pickle=False) as z:
            applied=z['applied'];fixed=z['original_double_pre_slew_target_rad'];obs=z['observations']
            for c in range(1,2279):
                np.testing.assert_array_equal(run.planner.apply_limits(fixed[c],applied[c-1],lo,hi),applied[c])
    def test_05_actual_head_leg_joint_slew_boundary(self):
        model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
        m=run.utility.kernel.Mapping(model);lo,hi=model.jnt_range[model.actuator_trnid[:,0]].T
        ref=(lo+hi)/2;fixed=ref.copy();fixed[m.head]+=.01
        for request in (np.zeros(10),np.linspace(-1e-4,1e-4,10),np.linspace(-1,1,10)):
            adjusted=k.replace_legs(fixed,ref,request,m.legs,lo,hi,True)
            np.testing.assert_array_equal(adjusted[m.head],fixed[m.head])
            np.testing.assert_array_equal(adjusted[m.legs],np.clip(ref[m.legs]+.18*request,lo[m.legs],hi[m.legs]))
            self.assertLessEqual(np.abs(adjusted-ref).max(),.18+1e-12)
            plan=run.planner.apply_limits(adjusted,ref,lo,hi)
            self.assertLessEqual(np.abs(plan-ref).max(),5.24*.02+1e-12)
        for c in (0,529,2278):
            self.assertIs(k.replace_legs(fixed,None,None,None,None,None,False),fixed)
    def test_06_constructor_and_reset_original_40x50(self):
        p=run.ROOT/'outputs/getup_parameter_pair_r198_smoke_20261011/zero_parity/case_769000/trajectory.npz'
        with np.load(p,allow_pickle=False) as z:
            self.assertEqual(z['construction_preparation_sensors'].shape,(40,50))
            self.assertEqual(z['preparation_sensors'].shape,(40,50))
            self.assertEqual(z['construction_preparation_times'].shape,(40,))
    def test_07_worker_direct_frozen_loader_new_fixture(self):
        where=os.environ.get('OPEN_DUCK_R204_TEST_CAPTURE')
        self.assertTrue(where,'Evidence capture destination is mandatory; missing capture must fail, never skip')
        parent=Path(where).parent/'worker_fixture';parent.mkdir();frozen=parent/'frozen';frozen.mkdir()
        old=run.ROOT/'outputs/getup_parameter_pair_r198_smoke_20261011/frozen'
        for name in ('snapshot.npz','initial_selector.npz','nominal_sensor_trajectory.npz'):shutil.copy2(old/name,frozen/name)
        run.init_worker(frozen)
        self.assertIn('feature_mode',run.local.WEIGHTS);self.assertEqual(run.local.NOMINAL.shape,(2279,55))
        self.assertIsNotNone(run.MODEL)
    def test_08_episode_replay_full_home_tail_and_invalid(self):
        for n in (46,304,2279):
            obs=np.zeros((n,50));r=np.linspace(-1,1,n);active=(np.arange(n)>0)&(np.arange(n)<529)
            e=run.episode_batch(obs,r,active,np.zeros((n,10)))
            independent=np.empty(n);tail=0.
            for c in range(n-1,-1,-1):tail=r[c]+k.GAMMA*tail;independent[c]=tail
            np.testing.assert_array_equal(e['original_full_returns'],independent)
            controls=np.flatnonzero(active);np.testing.assert_array_equal(e['control'],controls)
            np.testing.assert_array_equal(e['original_reward'],r)
            for j,c in enumerate(controls):
                np.testing.assert_array_equal(e['features'][j],k.features(obs[c],obs[0],int(c)))
                self.assertEqual(e['reward'][j],r[c])
                self.assertEqual(e['bootstrap'][j],float(c+1<n and c+1<529))
                self.assertEqual(e['home_tail'][j],independent[c+1] if c+1==529 and c+1<n else 0.)
                self.assertEqual(e['terminal'][j],c==n-1)
        r=np.zeros(2279);r[529]=2.;r[-1]=5.
        e=run.episode_batch(np.zeros((2279,50)),r,(np.arange(2279)>0)&(np.arange(2279)<529),np.zeros((2279,10)))
        self.assertGreater(e['home_tail'][-1],0);self.assertEqual(e['control'][-1],528)
    def test_09_episode_invalid_interfaces_block(self):
        obs=np.zeros((4,50));r=np.zeros(4);z=np.zeros((4,10))
        for mask in (np.ones(4,dtype=bool),np.ones(4),np.zeros(3,dtype=bool)):
            with self.assertRaises(ValueError):run.episode_batch(obs,r,mask,z)
        with self.assertRaises(ValueError):run.episode_batch(np.zeros((4,55)),r,[False,True,True,True],z)
    def test_10_fresh_joint_Q_target_and_full_checkpoint_state(self):
        learner=_INITIAL_LEARNER
        self.assertEqual(learner.count,0)
        for name,v in self.w.items():
            np.testing.assert_array_equal(learner.numpy_weights()[name],v)
            np.testing.assert_array_equal(learner.numpy_target()[name],v)
        self.assertTrue(learner.state_bytes())
        x=np.arange(101)/101.;v,m,L,raw=k.heads(self.w,x)
        np.testing.assert_array_equal(m,np.zeros(10));np.testing.assert_array_equal(raw,np.zeros(10))
        self.assertEqual(L.shape,(10,10))
        self.assertTrue(np.all(np.diag(L)>0))
        self.assertEqual(k.q_value(self.w,x,m),v)
        y=k.bellman(np.array([1.,2.,3.]),np.array([1.,0.,0.]),np.array([0.,0.,5.]),np.array([7.,8.,9.]))
        np.testing.assert_array_equal(y,[1+k.GAMMA*7,2,3+k.GAMMA*5])
        self.assertEqual(learner.count,0)
    def test_11_sample_noise_order_and_deterministic_no_consumption(self):
        a=np.random.default_rng(3041000);b=np.random.default_rng(3041000)
        for c in range(1,529):
            x=k.features(np.zeros(50),np.zeros(50),c)
            d=k.request(self.w,x,a);independent=b.normal(size=10)
            np.testing.assert_array_equal(d['noise'],independent)
            np.testing.assert_array_equal(d['proposal'],d['mean']+1e-4*independent)
            np.testing.assert_array_equal(d['action'],np.clip(d['proposal'],-1,1))
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        import copy
        before=copy.deepcopy(a.bit_generator.state)
        np.testing.assert_array_equal(k.request(self.w,x)['mean'],np.zeros(10))
        self.assertEqual(before,a.bit_generator.state)
        saturated={n:v.copy() for n,v in self.w.items()};saturated['q_b2'][1:11]=1000.
        d=k.request(saturated,x,a)
        self.assertTrue(np.all(d['mean']==1.));self.assertTrue(np.any(d['proposal']>1.))
        self.assertTrue(np.all(d['action']<=1.));self.assertTrue(np.any(d['action']==1.))
    def test_12_smoke_standard_null_and_invalid_block(self):
        reports={name:dict(physical_failures=0,nominal_success=True if None in cases else None,rows=[dict(controls=2279,complete_R157_parity={'old':True}) for c in cases]) for name,mode,cases,parity in run.SMOKE_SPEC}
        self.assertTrue(run.smoke_gate(reports));reports['initial_sample']['physical_failures']=1;self.assertFalse(run.smoke_gate(reports))
        reports['initial_sample']['physical_failures']=0;reports['initial_sample']['rows'][0]['controls']=304;self.assertFalse(run.smoke_gate(reports))
    def test_13_common_retention_no_union(self):
        base=dict(successes=18,rows=[dict(case_seed=i,success=i<18) for i in range(24)])
        report=dict(successes=19,nominal_success=True,physical_failures=0,rows=[dict(case_seed=i,success=i<19) for i in range(24)])
        self.assertTrue(run.eligible(report,base))
        report['rows'][0]['success']=False;self.assertFalse(run.eligible(report,base))
        report['rows'][0]['success']=True;report['physical_failures']=1;self.assertFalse(run.eligible(report,base))
    def test_14_checkpoint_earlier_tie_no_oracle(self):
        self.assertIsNone(run.select_checkpoint([]))
        self.assertEqual(run.select_checkpoint([(19,1.,2.,2),(19,1.,2.,1)]),1)
        self.assertEqual(run.select_checkpoint([(19,1.,2.,1),(20,-1.,-2.,2)]),2)
    def test_15_fixed_budget_source_strict_gate_no_old_mutation(self):
        self.assertEqual(6+6+25+50+50+25,162)
        source=Path(run.__file__).read_text();self.assertNotIn('utility.init_worker(',source)
        for token in ("active=mode!='baseline' and 0<k<529","if not active:assert adjusted is fixed","np.testing.assert_array_equal(applied,wanted)","np.testing.assert_array_equal(sim.prev,wanted)","learner.update(","smoke_natural_exit.json","formal/regression_natural_exit.json","complete_new_original_replay.npz","pre_update_learner_state.msgpack","full_naf_decisions.npz","policy_lower","policy_value","policy_raw_mean"):
            self.assertIn(token,source)
        self.assertEqual(source.count('learner.update('),1)
        for token in ('actor_update(', 'value_update(', 'normalize_advantage(', 'log_probability(', 'lstd_system(', 'save_pre_solve('):
            self.assertNotIn(token,source)
        tree=ast.parse(Path(k.__file__).read_text())
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('diagnostics') for n in ast.walk(tree)))
        launch=Path(run.__file__).with_name('launch_getup_total_target_naf_r204.py').read_text()
        self.assertIn('OPEN_DUCK_R204_TEST_CAPTURE',launch)
        self.assertNotIn('OPEN_DUCK_R201_TEST_CAPTURE',launch)
        self.assertIn("result['skipped']==result['failures']==result['errors']==0",launch)
    def test_16_manifest_original_copies_and_explicit_sources(self):
        where=os.environ.get('OPEN_DUCK_R204_TEST_CAPTURE')
        self.assertTrue(where,'Evidence capture destination is mandatory; missing capture must fail, never skip')
        import json
        manifest=json.loads((Path(where).parent/'source_manifest.json').read_text())
        for p,entry in manifest.items():
            self.assertEqual(run.prior.digest(p),entry['sha256'])
            self.assertEqual(run.prior.digest(Path(where)/entry['copy']),entry['sha256'])
        for name in ('train_getup_total_target_naf_r204.py','total_target_naf_kernel_r204.py','test_getup_total_target_naf_r204.py','launch_getup_total_target_naf_r204.py'):
            self.assertTrue(any(Path(p).name==name for p in manifest))

if __name__=='__main__':
    import json
    import sys
    import time
    where=os.environ.get('OPEN_DUCK_R204_TEST_CAPTURE')
    if not where:raise RuntimeError('Mandatory source capture missing; zero-skip gate refuses execution')
    destination=Path(where)
    assert not destination.exists() and not (destination.parent/'interface_result.json').exists()
    _INITIAL_LEARNER=k.Learner(k.initial_weights())
    _INITIAL_LEARNER.state_bytes()
    manifest=run.capture_sources(destination)
    run.local.write_json(destination.parent/'source_manifest.json',manifest)
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Regression)
    started=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    runtime_manifest=run.capture_sources(destination,manifest)
    run.local.write_json(destination.parent/'runtime_source_manifest.json',runtime_manifest)
    unchanged=all(run.prior.digest(p)==entry['sha256'] and run.prior.digest(destination/entry['copy'])==entry['sha256'] for p,entry in runtime_manifest.items())
    receipt=dict(tests_run=result.testsRun,skipped=len(result.skipped),failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.monotonic()-started,source_hashes_unchanged=unchanged,new_dynamic_attempts=0,startup_source_count=len(manifest),complete_source_count=len(runtime_manifest),native_binary_snapshot=False)
    run.local.write_json(destination.parent/'interface_result.json',receipt)
    sys.exit(0 if result.wasSuccessful() and result.testsRun==16 and not result.skipped and unchanged else 1)
