"""Immutable kernel plus new wrapper interfaces, no dynamic episode."""
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import mujoco
from diagnostics import train_getup_adaptive_sensor_mpc_r195 as run


class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
        cls.model.opt.timestep=.002
        import os
        destination=os.environ.get('OPEN_DUCK_R195_TEST_CAPTURE')
        if destination:run.local.write_json(Path(destination).parent/'test_source_manifest.json',run.capture_sources(Path(destination)))
    def test_immutable_kernel_source(self):
        self.assertEqual(run.prior.digest(Path(run.kernel.__file__)),run.KERNEL_SHA)
    def test_frozen_worker_snapshot_selector_nominal(self):
        run.local.WEIGHTS=None
        run.init_worker(run.ROOT/'outputs/getup_phase_coordinate_r185b_smoke_20261010/frozen')
        self.assertIn('feature_mode',run.local.WEIGHTS);self.assertIsNotNone(run.MODEL)
        self.assertEqual(run.local.NOMINAL.shape,(2279,55))
    def test_causal_wrapper_merge_signature(self):
        self.assertEqual(list(inspect.signature(run.merge_adaptive).parameters),['fixed','reference','request','geometry','lower','upper'])
        for word in ('case','seed','qpos','qvel','future','label'):self.assertNotIn(word,inspect.getsource(run.merge_adaptive))
    def test_merge_total_bounds_slew_and_head(self):
        g=run.kernel.Mapping(self.model);rng=np.random.default_rng(293)
        for _ in range(100):
            ref=rng.uniform(-.3,.3,14);fixed=ref+rng.uniform(-.18,.18,14)
            request=rng.normal(size=14);request[g.head]=0
            result=run.merge_adaptive(fixed,ref,request,g,-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(result-ref).max(),.18+1e-12)
            np.testing.assert_array_equal(result[g.head],fixed[g.head])
            planned=run.planner.apply_limits(result,np.zeros(14),-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(planned).max(),5.24*.02)
        self.assertIs(run.merge_adaptive(fixed,ref,np.zeros(14),g,-np.ones(14),np.ones(14)),fixed)
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
            self.assertTrue(any(p.endswith('test_getup_adaptive_sensor_mpc_r195.py') for p in manifest))

    def test_actual_nominal_zero_and_home(self):
        run.init_worker(run.ROOT/'outputs/getup_phase_coordinate_r185b_smoke_20261010/frozen')
        initial=run.local.NOMINAL[0,:34]
        for mode in ('zero','positive','negative','candidate'):
            controller=run.kernel.Controller(mode)
            for k in range(529):
                nominal=run.local.NOMINAL[k,:34]
                raw=controller.request(nominal,nominal,initial,initial,k)
                np.testing.assert_array_equal(raw,np.zeros(10))
                controller.commit(np.zeros(10))
            for k in (529,2278):
                np.testing.assert_array_equal(controller.request(np.ones(34),np.zeros(34),initial,initial,k),np.zeros(10))
    def test_complete_online_recurrence(self):
        model=run.kernel.OnlineModel();rng=np.random.default_rng(195)
        theta=np.zeros((34,78));p=100*np.eye(78);previous=None;pending=None
        for _ in range(40):
            x=rng.normal(0,.02,34);u=rng.normal(0,1e-5,10)
            change=np.zeros(34) if previous is None else x-previous
            if pending is not None:
                pf=p@pending;d=run.kernel.FORGET+pending@pf
                theta+=np.outer(change-theta@pending,pf/d)
                p=(p-np.outer(pf,pf)/d)/run.kernel.FORGET;p=(p+p.T)*.5
            model.observe(x)
            np.testing.assert_array_equal(model.theta,theta);np.testing.assert_array_equal(model.p,p)
            model.commit(u);pending=np.r_[x,change,u/run.kernel.BOUND];previous=x.copy()
            np.testing.assert_array_equal(model.pending,pending)
        self.assertEqual(model.count,39)
    def test_observe_only_uses_realized_transition(self):
        model=run.kernel.OnlineModel();model.observe(np.ones(34));model.commit(np.ones(10)*1e-5)
        np.testing.assert_array_equal(model.theta,np.zeros((34,78)))
        model.observe(np.ones(34)*1.1)
        self.assertGreater(np.abs(model.theta).max(),0)
        with self.assertRaises(RuntimeError):model.commit(np.zeros(10));model.commit(np.zeros(10))
    def test_independent_probe_reconstruction(self):
        rng=np.random.default_rng(195);o=rng.normal(size=(66,55)).astype(np.float32)
        n=rng.normal(size=(66,55)).astype(np.float32);a=rng.normal(0,1e-5,(66,14));b=np.zeros_like(a);legs=np.arange(10)
        model=run.kernel.OnlineModel()
        for k in range(65):
            x=run.kernel.delta(o[k,:34],n[k,:34],o[0,:34],n[0,:34])
            if k==0:x[:]=0
            model.observe(x)
            if k<64:model.commit(a[k,legs])
        theta,p=run.independent_probe_model(o,a,b,n,legs)
        np.testing.assert_array_equal(theta,model.theta);np.testing.assert_array_equal(p,model.p)
    def test_mpc_matches_independent_constant_input_quadratic(self):
        model=run.kernel.OnlineModel();model.theta[:,68:]=np.eye(34,10)*.02
        x=np.linspace(-.003,.003,34);change=np.zeros(34);b=model.theta[:,68:]
        lhs=run.kernel.PENALTY*np.eye(10);rhs=np.zeros(10)
        for horizon in range(1,4):
            response=horizon*b
            lhs+=response.T@response/34;rhs+=response.T@x/34
        expected=run.kernel.BOUND*np.clip(-np.linalg.solve(lhs,rhs),-1,1)
        np.testing.assert_allclose(model.mpc(x,change),expected,atol=1e-20,rtol=1e-13)
        self.assertLess(np.linalg.norm(model.predict(x,change,expected/run.kernel.BOUND)),np.linalg.norm(x))
    def test_mpc_zero_model_and_action_bound(self):
        model=run.kernel.OnlineModel();x=np.ones(34)
        np.testing.assert_array_equal(model.mpc(x,x),np.zeros(10))
        model.theta[:,68:]=np.ones((34,10))
        self.assertLessEqual(np.abs(model.mpc(x*1e4,x)).max(),run.kernel.BOUND)
    def test_program_common_prefix_and_probe(self):
        a=run.kernel.Controller('positive');b=run.kernel.Controller('negative')
        x=np.zeros(34);initial=x.copy()
        for k in range(65):
            x[0]=k*.002
            ar=a.request(x,np.zeros(34),initial,initial,k);br=b.request(x,np.zeros(34),initial,initial,k)
            if k<64:np.testing.assert_array_equal(ar,br)
            else:self.assertGreater(np.abs(ar-br).max(),0)
            a.commit(ar);b.commit(br)
        self.assertEqual(a.rng,b.rng)
    def test_causal_sensor_only_and_no_private_physics(self):
        from contextlib import ExitStack
        from unittest.mock import patch
        source=inspect.getsource(run.kernel)
        for name in ('qpos','qvel','MjData','mj_forward','mj_step','mj_kinematics','mj_rne','lookup'):
            self.assertNotIn(name,source)
        controller=run.kernel.Controller('candidate')
        with ExitStack() as stack:
            for name in ('MjData','mj_forward','mj_step','mj_inverse','mj_kinematics','mj_contactForce','mj_collision','mj_rne','mj_passive'):
                stack.enter_context(patch.object(mujoco,name,side_effect=AssertionError(name)))
            for k in range(100):
                x=np.full(34,k*.001,dtype=np.float32)
                raw=controller.request(x,np.zeros(34),np.zeros(34),np.zeros(34),k);controller.commit(raw)
                self.assertLessEqual(np.abs(raw).max(),run.kernel.BOUND)
    def test_control_zero_object_and_head(self):
        mapping=run.kernel.Mapping(self.model);controller=run.kernel.Controller('positive')
        raw=controller.request(np.ones(34),np.zeros(34),np.ones(34),np.zeros(34),0)
        np.testing.assert_array_equal(raw,np.zeros(10))
        fixed=np.zeros(14);request=np.zeros(14);request[mapping.legs]=raw
        self.assertIs(run.merge_adaptive(fixed,fixed,request,mapping,-np.ones(14),np.ones(14)),fixed)
    def test_numeric_failure_preserved_fallback(self):
        from unittest.mock import patch
        controller=run.kernel.Controller('candidate')
        controller.request(np.zeros(34),np.zeros(34),np.zeros(34),np.zeros(34),0);controller.commit(np.zeros(10))
        with patch.object(controller.model,'mpc',side_effect=np.linalg.LinAlgError('Injected')):
            raw=controller.request(np.ones(34),np.zeros(34),np.zeros(34),np.zeros(34),64)
        np.testing.assert_array_equal(raw,np.zeros(10));self.assertEqual(controller.fallbacks,[64])
    def test_finite_parameters_and_invalid_shapes(self):
        self.assertEqual((run.kernel.CALIBRATION,run.kernel.PREDICTION,run.SEED),(64,3,295))
        with self.assertRaises(ValueError):run.kernel.Controller('extra')
        with self.assertRaises(ValueError):run.kernel.delta(np.zeros(35),np.zeros(34),np.zeros(34),np.zeros(34))
        with self.assertRaises(ValueError):run.kernel.OnlineModel().mpc(np.full(34,np.nan),np.zeros(34))
    def test_launcher_gate_rejects_model_and_physics_failure(self):
        from diagnostics import launch_getup_adaptive_sensor_mpc_r195 as launcher
        row=dict(case_seed=None,complete_R157_parity={'all':True},maximum_raw_adaptive_request_rad=0,adaptive_fallback_controls=[])
        report=dict(smoke=True,terminal_result_saved=True,causal_effect={'passed':True},reports={'zero_parity':{'nominal_success':True,'physical_failures':0,'rows':[row]},'calibration_standard':{'nominal_success':True,'physical_failures':0,'rows':[row]}})
        self.assertTrue(launcher.smoke_gate(report))
        report['causal_effect']['passed']=False;self.assertFalse(launcher.smoke_gate(report))
        report['causal_effect']['passed']=True;report['reports']['zero_parity']['physical_failures']=1
        self.assertFalse(launcher.smoke_gate(report))

if __name__=='__main__':unittest.main(verbosity=2)
