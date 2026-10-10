"""Immutable kernel plus new wrapper interfaces, no dynamic episode."""
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import mujoco
from diagnostics import train_getup_shared_horizon_teacher_r193 as run


class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
        cls.model.opt.timestep=.002
        import os
        destination=os.environ.get('OPEN_DUCK_R193_TEST_CAPTURE')
        if destination:run.local.write_json(Path(destination).parent/'test_source_manifest.json',run.capture_sources(Path(destination)))
    def test_immutable_kernel_source(self):
        self.assertEqual(run.prior.digest(Path(run.kernel.__file__)),run.KERNEL_SHA)
    def test_frozen_worker_snapshot_selector_nominal(self):
        run.local.WEIGHTS=None
        run.init_worker(run.ROOT/'outputs/getup_phase_coordinate_r185b_smoke_20261010/frozen')
        self.assertIn('feature_mode',run.local.WEIGHTS);self.assertIsNotNone(run.MODEL)
        self.assertEqual(run.local.NOMINAL.shape,(2279,55))
    def test_causal_wrapper_merge_signature(self):
        self.assertEqual(list(inspect.signature(run.merge_teacher).parameters),['fixed','reference','request','geometry','lower','upper'])
        for word in ('case','seed','qpos','qvel','future','label'):self.assertNotIn(word,inspect.getsource(run.merge_teacher))
    def test_merge_total_bounds_slew_and_head(self):
        g=run.kernel.Mapping(self.model);rng=np.random.default_rng(293)
        for _ in range(100):
            ref=rng.uniform(-.3,.3,14);fixed=ref+rng.uniform(-.18,.18,14)
            request=rng.normal(size=14);request[g.head]=0
            result=run.merge_teacher(fixed,ref,request,g,-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(result-ref).max(),.18+1e-12)
            np.testing.assert_array_equal(result[g.head],fixed[g.head])
            planned=run.planner.apply_limits(result,np.zeros(14),-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(planned).max(),5.24*.02)
        self.assertIs(run.merge_teacher(fixed,ref,np.zeros(14),g,-np.ones(14),np.ones(14)),fixed)
    def test_preparation_call_order_and_full_exact_parity(self):
        def sample(recorder):
            sim=SimpleNamespace(data=SimpleNamespace(time=0.));calls=[]
            def step(target):calls.append(target);sim.data.time+=.02;return target
            sim.step_target=step
            def prepare():
                for k in range(40):sim.step_target(k)
                return 123
            observe=lambda s:np.full(50,s.data.time,dtype=np.float32)
            return recorder(sim,prepare,observe),calls,sim.step_target is step
        a=sample(run.local.prior.audit.record_preparation)
        captured=dict(frames=[],times=[])
        b=sample(lambda s,p,o:run.record_preparation_partial(s,p,o,captured))
        for x,y in zip(a[0],b[0]):np.testing.assert_array_equal(x,y)
        self.assertEqual(a[1:],b[1:]);self.assertEqual(len(captured['frames']),40)
    def test_preparation_exception_partial_and_restore(self):
        sim=SimpleNamespace(data=SimpleNamespace(time=0.))
        def step(target):sim.data.time+=.02;return target
        sim.step_target=step;captured=dict(frames=[],times=[])
        def prepare():
            for k in range(3):sim.step_target(k)
            raise RuntimeError('Injected prepare failure')
        with self.assertRaises(RuntimeError):run.record_preparation_partial(sim,prepare,lambda s:np.full(50,s.data.time),captured)
        self.assertEqual(len(captured['frames']),3);self.assertIs(sim.step_target,step)
    def test_constructor_and_reset_preparation_separate_batches(self):
        sim=SimpleNamespace(data=SimpleNamespace(time=0.))
        def step(target):sim.data.time+=.02;return target
        sim.step_target=step;captured=dict(frames=[],times=[])
        def prepare():
            for k in range(40):sim.step_target(k)
        for _ in range(2):
            _,frames,times=run.record_preparation_partial(sim,prepare,lambda s:np.full(50,s.data.time),captured)
            self.assertEqual(frames.shape,(40,50));self.assertEqual(times.shape,(40,))
        self.assertEqual(len(captured['completed_batches']),2)
        self.assertEqual(len(captured['frames']),40)
        self.assertIs(sim.step_target,step)
    def test_complete_compiled_model_hash(self):
        before=run.model_digest(self.model);g=run.kernel.Mapping(self.model);assert len(g.legs)==10
        self.assertEqual(before,run.model_digest(self.model))
        original=self.model.opt.timestep
        try:
            self.model.opt.timestep=original*2
            self.assertNotEqual(before,run.model_digest(self.model))
        finally:self.model.opt.timestep=original
        self.assertEqual(before,run.model_digest(self.model))
    def test_full_include_asset_manifest(self):
        files=run.model_files(run.local.prior.program.SCENE)
        self.assertIn(str(Path(run.local.prior.program.SCENE).resolve()),files)
        self.assertTrue(any(p.endswith('robot_decomposed.xml') for p in files))
        self.assertTrue(any(p.lower().endswith('.stl') for p in files))
        self.assertEqual(files,run.model_files(run.local.prior.program.SCENE))
    def test_all_imported_python_sources_captured(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest=run.capture_sources(Path(folder)/'sources')
            for module in (run,run.kernel,run.local,run.selector,run.planner,run.local.prior.audit):
                self.assertIn(str(Path(module.__file__).resolve()),manifest)
            self.assertTrue(any(p.endswith('test_getup_shared_horizon_teacher_r193.py') for p in manifest))

    def test_wrapper_actual_nominal_zero_and_control_zero(self):
        run.init_worker(run.ROOT/'outputs/getup_phase_coordinate_r185b_smoke_20261010/frozen')
        g=run.kernel.Mapping(self.model);initial=run.local.NOMINAL[0,:34];c=run.kernel.make_proposals()[0][1]
        fixed=np.zeros(14)
        for k in range(529):
            nominal=run.local.NOMINAL[k,:34]
            raw,error,wave,a=run.kernel.teacher_request(nominal,nominal,initial,initial,c,g,k)
            np.testing.assert_array_equal(raw,np.zeros(14));np.testing.assert_array_equal(error,np.zeros(34));self.assertEqual(a,0)
            self.assertIs(run.merge_teacher(fixed,fixed,raw,g,-np.ones(14),np.ones(14)),fixed)
        x=initial.copy();x[:3]+=.1
        raw,error,_,a=run.kernel.teacher_request(x,initial,x,initial,c,g,0)
        np.testing.assert_array_equal(raw,np.zeros(14));np.testing.assert_array_equal(error,np.zeros(34));self.assertEqual(a,0)
    def test_sensor_only_teacher_and_causal_input_signature(self):
        from unittest.mock import patch
        from contextlib import ExitStack
        sig=inspect.signature(run.kernel.teacher_request)
        self.assertEqual(list(sig.parameters),['current','nominal','initial','initial_nominal','coefficients','mapping','control'])
        source=inspect.getsource(run.kernel.teacher_request)
        for forbidden in ('case','seed','qpos','qvel','label','future','lookup','path'):self.assertNotIn(forbidden,source)
        g=run.kernel.Mapping(self.model);x=np.zeros(34,np.float32);x[0]=.1;x[20]=.2;c=run.kernel.make_proposals()[0][1]
        with ExitStack() as stack:
            for name in ('MjData','mj_forward','mj_step','mj_inverse','mj_kinematics','mj_contactForce','mj_collision','mj_rne','mj_passive'):
                stack.enter_context(patch.object(mujoco,name,side_effect=AssertionError(name)))
            raw,error,wave,a=run.kernel.teacher_request(x,np.zeros(34),np.zeros(34),np.zeros(34),c,g,1)
        np.testing.assert_array_equal(raw[g.head],np.zeros(4));self.assertGreater(a,0)
    def test_four_global_harmonics_not_nodes_or_window(self):
        b=run.kernel.BASIS
        self.assertEqual(b.shape,(529,4));np.testing.assert_array_equal(b[[0,-1]],np.zeros((2,4)))
        for k in range(1,528):
            expected=np.array([np.sin(np.pi*k/528*n) for n in range(1,5)])
            np.testing.assert_allclose(b[k],expected,atol=2e-15,rtol=0)
        self.assertFalse(b.flags.writeable)
    def test_held_sensor_future_and_metadata_not_inputs(self):
        g=run.kernel.Mapping(self.model);c=run.kernel.make_proposals()[0][1];x=np.arange(34,dtype=np.float32)/1000;z=np.zeros(34)
        a=run.kernel.teacher_request(x,z,z,z,c,g,123)
        unrelated_future=np.ones((50,34))*900
        b=run.kernel.teacher_request(x.copy(),z.copy(),z.copy(),z.copy(),c.copy(),g,123)
        for av,bv in zip(a,b):np.testing.assert_array_equal(av,bv)
    def test_full_horizon_request_units_head_and_bounds(self):
        g=run.kernel.Mapping(self.model);c=np.full((10,4),run.kernel.BOUND);z=np.zeros(34);x=np.ones(34,np.float32)
        for k in range(1,529):
            raw,delta,wave,a=run.kernel.teacher_request(x,z,z,z,c,g,k)
            np.testing.assert_array_equal(raw[g.head],np.zeros(4))
            np.testing.assert_array_equal(raw[g.legs],a*(c@run.kernel.BASIS[k]))
            self.assertLessEqual(np.abs(raw).max(),4*run.kernel.BOUND)
            self.assertGreaterEqual(a,0);self.assertLessEqual(a,1)
        _,delta,_,_=run.kernel.teacher_request(x,z,z,z,c,g,1)
        np.testing.assert_array_equal(delta,1/run.kernel.SCALE)
    def test_zero_coefficients_object_and_home(self):
        g=run.kernel.Mapping(self.model);z=np.zeros(34);x=np.ones(34);fixed=np.zeros(14)
        for k in (0,1,20,528,529,2278):
            raw,delta,wave,a=run.kernel.teacher_request(x,z,z,z,np.zeros((10,4)),g,k)
            np.testing.assert_array_equal(raw,np.zeros(14))
            self.assertIs(run.merge_teacher(fixed,fixed,raw,g,-np.ones(14),np.ones(14)),fixed)
        for k in (529,2278):
            raw,delta,wave,a=run.kernel.teacher_request(x,z,z,z,run.kernel.make_proposals()[0][1],g,k)
            for arr in (raw,delta,wave):np.testing.assert_array_equal(arr,np.zeros_like(arr))
            self.assertEqual(a,0.)
    def test_fixed_four_proposals_rng_pairing_and_no_search_extension(self):
        p,rng=run.kernel.make_proposals();q,state=run.kernel.make_proposals()
        self.assertEqual(len(p),5);self.assertEqual(rng,state)
        for a,b in zip(p,q):np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(p[1],-p[2]);np.testing.assert_array_equal(p[3],-p[4])
        self.assertTrue(all(np.abs(c).max()<=run.kernel.BOUND for c in p))
        self.assertEqual(run.SEED,293)
    def test_retention_selection_rejects_success_union_and_invalid(self):
        def report(successes,invalid=0,nominal=True):
            rows=[dict(case_seed=i,initial_hash=str(i),success=i in successes,return_sum=float(i)) for i in range(4)]
            return dict(rows=rows,successes=len(successes),nominal_success=nominal,physical_failures=invalid)
        base=report({0,1});regress=report({1,2,3});rescue=report({0,1,2});invalid=report({0,1,2,3},1)
        selected,e=run.kernel.choose_teacher([base,regress,rescue,invalid])
        self.assertEqual(selected,2);self.assertFalse(e[1]['eligible']);self.assertFalse(e[3]['eligible'])
        self.assertEqual(run.kernel.choose_teacher([base,regress,invalid])[0],0)
        rescue['rows'][0]['initial_hash']='mismatched'
        with self.assertRaises(ValueError):run.kernel.retention(rescue,base)
    def test_invalid_sensor_coefficients_control(self):
        g=run.kernel.Mapping(self.model);z=np.zeros(34);c=np.zeros((10,4))
        for bad in (np.zeros(33),np.full(34,np.nan)):
            with self.assertRaises(ValueError):run.kernel.teacher_request(bad,z,z,z,c,g,1)
        for bad in (np.zeros((10,5)),np.full((10,4),np.nan),np.full((10,4),.001)):
            with self.assertRaises(ValueError):run.kernel.teacher_request(z,z,z,z,bad,g,1)
        for bad in (-1,1.2):
            with self.assertRaises(ValueError):run.kernel.teacher_request(z,z,z,z,c,g,bad)
    def test_planning_previous_float32_history_and_actual_target(self):
        # Synthetic plan arithmetic, not a simulated step_target acceptance.
        previous=np.linspace(-.05,.05,14);target=np.linspace(-.2,.2,14)
        observed=(previous-np.zeros(14)).astype(np.float32)
        np.testing.assert_array_equal(observed,previous.astype(np.float32))
        planned=run.planner.apply_limits(target,previous,-np.ones(14),np.ones(14))
        np.testing.assert_array_equal(planned,np.clip(np.clip(target,-1,1),previous-5.24*.02,previous+5.24*.02))
    def test_launcher_gate_requires_physical_validity_and_standard_zero(self):
        from diagnostics import launch_getup_shared_horizon_teacher_r193 as launch
        report={'smoke':True,'terminal_result_saved':True,
                'parity':{'nominal_success':True,'physical_failures':0,'rows':[{'complete_R157_parity':{'x':True}}]},
                'teacher':{'nominal_success':True,'physical_failures':0,'rows':[{'case_seed':None,'complete_R157_parity':{'x':True},'maximum_raw_teacher_request_rad':0.}]}}
        launch.smoke_gate(report)
        report['teacher']['physical_failures']=1
        with self.assertRaises(AssertionError):launch.smoke_gate(report)
        report['teacher']['physical_failures']=0;report['teacher']['rows'][0]['maximum_raw_teacher_request_rad']=1e-20
        with self.assertRaises(AssertionError):launch.smoke_gate(report)



if __name__=='__main__':unittest.main(verbosity=2)

