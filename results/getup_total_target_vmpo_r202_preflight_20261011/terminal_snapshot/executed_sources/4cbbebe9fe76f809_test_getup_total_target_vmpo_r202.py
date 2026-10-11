"""New offline causal-control/episode/selection interface regressions; no new integration."""
import ast
import os
from pathlib import Path
import shutil
import unittest
import numpy as np
import mujoco
from diagnostics import train_getup_total_target_vmpo_r202 as run
from diagnostics import total_target_vmpo_kernel_r202 as k

class Regression(unittest.TestCase):
    def setUp(self):
        self.w=k.initial_weights()
    def test_01_immutable_numeric_kernel(self):
        self.assertEqual(run.prior.digest(k.__file__),run.KERNEL_SHA)
        self.assertNotIn('mujoco',Path(k.__file__).read_text())
        self.assertEqual((k.SEED,k.VALUE_UPDATES,k.ACTOR_STEPS,k.ETA_STEPS),(302,64,64,80))
    def test_02_actual_native_causal_layout(self):
        p=run.ROOT/'outputs/getup_parameter_pair_r198_smoke_20261011/zero_parity/case_769000/trajectory.npz'
        with np.load(p,allow_pickle=False) as z:
            obs=z['observations'];self.assertEqual(obs.shape,(2279,55))
            for c in (0,1,528,529,2278):
                x=k.features(obs[c,:50],obs[0,:50],c)
                np.testing.assert_array_equal(x[:50],np.tanh(obs[c,:50].astype(float)))
                np.testing.assert_array_equal(x[50:100],np.tanh(obs[0,:50].astype(float)))
                self.assertEqual(x[-1],min(c/528,1))
                np.testing.assert_array_equal(k.network(self.w,x,'actor'),np.zeros(10))
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
        for latent in (np.zeros(10),np.linspace(-1e-4,1e-4,10),np.linspace(-100,100,10)):
            adjusted=k.replace_legs(fixed,ref,latent,m.legs,lo,hi,True)
            np.testing.assert_array_equal(adjusted[m.head],fixed[m.head])
            np.testing.assert_array_equal(adjusted[m.legs],np.clip(ref[m.legs]+.18*np.tanh(latent),lo[m.legs],hi[m.legs]))
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
        where=os.environ.get('OPEN_DUCK_R202_TEST_CAPTURE')
        if not where:self.skipTest('New evidence destination required')
        parent=Path(where).parent/'worker_fixture';parent.mkdir();frozen=parent/'frozen';frozen.mkdir()
        old=run.ROOT/'outputs/getup_parameter_pair_r198_smoke_20261011/frozen'
        for name in ('snapshot.npz','initial_selector.npz','nominal_sensor_trajectory.npz'):shutil.copy2(old/name,frozen/name)
        run.init_worker(frozen)
        self.assertIn('feature_mode',run.local.WEIGHTS);self.assertEqual(run.local.NOMINAL.shape,(2279,55))
        self.assertIsNotNone(run.MODEL)
    def test_08_episode_MC_full_home_tail_and_invalid(self):
        for n in (46,304,2279):
            obs=np.zeros((n,50));r=np.linspace(-1,1,n);active=(np.arange(n)>0)&(np.arange(n)<529)
            e=run.episode_batch(obs,r,active,np.zeros((n,10)),self.w)
            independent=np.empty(n);tail=0.
            for c in range(n-1,-1,-1):tail=r[c]+k.GAMMA*tail;independent[c]=tail
            np.testing.assert_array_equal(e['returns'],independent)
            np.testing.assert_array_equal(e['active'],active)
            np.testing.assert_array_equal(e['raw_advantage'],independent-k.network(self.w,e['features'],'value')[:,0])
            self.assertEqual(e['returns'][-1],r[-1])
        r=np.zeros(2279);r[529]=2;r[-1]=5
        e=run.episode_batch(np.zeros((2279,50)),r,(np.arange(2279)>0)&(np.arange(2279)<529),np.zeros((2279,10)),self.w)
        self.assertGreater(e['returns'][528],0);self.assertFalse(e['active'][529])
    def test_09_episode_invalid_interfaces_block(self):
        obs=np.zeros((4,50));r=np.zeros(4);z=np.zeros((4,10))
        for mask in (np.ones(4,dtype=bool),np.ones(4),np.zeros(3,dtype=bool)):
            with self.assertRaises(ValueError):run.episode_batch(obs,r,mask,z,self.w)
        with self.assertRaises(ValueError):run.episode_batch(np.zeros((4,55)),r,[False,True,True,True],z,self.w)
    def test_10_advantage_active_only_preupdate_value(self):
        raw=np.array([999.,-2.,1.,7.,-999.]);active=np.array([False,True,True,True,False])
        norm,summary=run.normalize_advantage(raw,active)
        np.testing.assert_array_equal(norm[~active],np.zeros(2))
        np.testing.assert_array_equal(norm[active],(raw[active]-raw[active].mean())/(raw[active].std()+1e-8))
        self.assertEqual(summary['active_count'],3)
        raw[~active]*=1e6;np.testing.assert_array_equal(run.normalize_advantage(raw,active)[0],norm)
    def test_11_sample_noise_order_and_deterministic_no_consumption(self):
        a=np.random.default_rng(3021000);b=np.random.default_rng(3021000)
        for c in range(1,529):
            mean=k.network(self.w,k.features(np.zeros(50),np.zeros(50),c),'actor')
            noise=a.normal(size=10);ind=b.normal(size=10)
            np.testing.assert_array_equal(noise,ind)
            np.testing.assert_array_equal(mean+k.SIGMA*noise,mean+1e-4*ind)
            self.assertEqual(float(k.log_probability(mean+k.SIGMA*noise,mean)),float(k.log_probability(mean+1e-4*ind,mean)))
        self.assertEqual(a.bit_generator.state,b.bit_generator.state)
        before=a.bit_generator.state;mean=k.network(self.w,k.features(np.zeros(50),np.zeros(50),1),'actor')
        latent=mean+k.SIGMA*np.zeros(10);np.testing.assert_array_equal(np.tanh(latent),np.zeros(10))
        self.assertEqual(before,a.bit_generator.state)
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
    def test_15_fixed_budget_source_and_no_old_capture_mutation(self):
        self.assertEqual(6+6+25+50+50+25,162)
        source=Path(run.__file__).read_text();self.assertNotIn('utility.init_worker(',source)
        for token in ("active=mode!='baseline' and 0<k<529","if not active:assert adjusted is fixed","np.testing.assert_array_equal(applied,wanted)","np.testing.assert_array_equal(sim.prev,wanted)","OUTPUT,SMOKE,LAUNCH,PREFLIGHT,NUMERIC,NUMERIC_B,NUMERIC_SOURCE","learner.actor_update(","learner.value_update("):
            self.assertIn(token,source)
        tree=ast.parse(Path(k.__file__).read_text())
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('diagnostics') for n in ast.walk(tree)))
        self.assertNotIn('numeric_r202',source[source.index('def main():'):])
    def test_16_manifest_original_copies_and_explicit_sources(self):
        where=os.environ.get('OPEN_DUCK_R202_TEST_CAPTURE')
        if not where:self.skipTest('New evidence destination required')
        import json
        manifest=json.loads((Path(where).parent/'source_manifest.json').read_text())
        for p,entry in manifest.items():
            self.assertEqual(run.prior.digest(p),entry['sha256'])
            self.assertEqual(run.prior.digest(Path(where)/entry['copy']),entry['sha256'])
        for name in ('train_getup_total_target_vmpo_r202.py','total_target_vmpo_kernel_r202.py','test_getup_total_target_vmpo_r202.py','launch_getup_total_target_vmpo_r202.py'):
            self.assertTrue(any(Path(p).name==name for p in manifest))

if __name__=='__main__':
    learning_runtime=k.Learner(k.initial_weights())
    where=os.environ.get('OPEN_DUCK_R202_TEST_CAPTURE')
    if where:
        destination=Path(where);manifest=run.capture_sources(destination)
        run.local.write_json(destination.parent/'source_manifest.json',manifest)
    unittest.main()
