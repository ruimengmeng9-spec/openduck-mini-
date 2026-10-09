"""Immutable kernel plus new wrapper interfaces, no dynamic episode."""
import inspect
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import mujoco
from diagnostics import probe_getup_centroidal_momentum_r188b as run
from diagnostics.test_centroidal_momentum_kernel_r187b import MomentumTests


class WrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    def test_immutable_kernel_source(self):
        self.assertEqual(run.prior.digest(Path(run.kernel.__file__)),run.KERNEL_SHA)
    def test_frozen_worker_snapshot_selector_nominal(self):
        run.local.WEIGHTS=None
        run.init_worker(run.ROOT/'outputs/getup_phase_coordinate_r185b_smoke_20261010/frozen')
        self.assertIn('feature_mode',run.local.WEIGHTS);self.assertIsNotNone(run.MODEL)
        self.assertEqual(run.local.NOMINAL.shape,(2279,55))
    def test_causal_wrapper_merge_signature(self):
        self.assertEqual(list(inspect.signature(run.merge_momentum).parameters),['fixed','reference','request','geometry','lower','upper'])
        for word in ('case','seed','qpos','qvel','future','label'):self.assertNotIn(word,inspect.getsource(run.merge_momentum))
    def test_merge_total_bounds_slew_and_head(self):
        g=run.kernel.MomentumGeometry(self.model);rng=np.random.default_rng(288)
        for _ in range(100):
            ref=rng.uniform(-.3,.3,14);fixed=ref+rng.uniform(-.18,.18,14)
            request=rng.normal(size=14);request[g.head]=0
            result=run.merge_momentum(fixed,ref,request,g,-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(result-ref).max(),.18+1e-12)
            np.testing.assert_array_equal(result[g.head],fixed[g.head])
            planned=run.planner.apply_limits(result,np.zeros(14),-np.ones(14),np.ones(14))
            self.assertLessEqual(np.abs(planned).max(),5.24*.02)
        self.assertIs(run.merge_momentum(fixed,ref,np.zeros(14),g,-np.ones(14),np.ones(14)),fixed)
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
        before=run.model_digest(self.model);g=run.kernel.MomentumGeometry(self.model);g.measure(np.zeros(34,np.float32))
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
            self.assertTrue(any(p.endswith('test_getup_centroidal_momentum_r188b.py') for p in manifest))


if __name__=='__main__':unittest.main()


