"""Reference identity and original acceptance regression checks."""
import ast
import json
from pathlib import Path
import unittest
import numpy as np

from diagnostics.getup_reference_env_r100 import ReferenceEpisode,checked_residual,full_trial
from diagnostics.search_getup_reference_feedback_r64 import features,residual

ROOT=Path('/data/shijinsheng/open_duck')
AUDIT=ROOT/'outputs/getup_path_audit_r99_20261005'
SCENE=ROOT/'training/getup_decomposed_r4/model/scene.xml'
STAND=ROOT/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'


class ReferenceTests(unittest.TestCase):
    def test_zero_policy_retains_nominal_complete_path_acceptance(self):
        expected=next(r for r in json.loads((AUDIT/'results.json').read_text())['results']
                      if r['seed'] is None and r['decision_stride']==1)
        actual=full_trial(str(SCENE),str(STAND),AUDIT/'frozen_path.npz',None,None,None)
        self.assertTrue(actual['success'])
        # Old audit restored a snapshot into an already-used simulator; this
        # environment freshly prepares every fall. Do not claim bitwise identity
        # across those initialization procedures.
        self.assertAlmostEqual(actual['final']['up_z'],expected['final']['up_z'],places=6)
        self.assertLessEqual(abs(actual['strict_tail_s']-expected['strict_tail_s']),.02000001)
        self.assertEqual(actual['initial_hash'],expected['initial_hash'])

    def test_zero_residual_matches_original_formula_control_by_control(self):
        env=ReferenceEpisode(str(SCENE),str(STAND),AUDIT/'frozen_path.npz',200,True)
        other=ReferenceEpisode(str(SCENE),str(STAND),AUDIT/'frozen_path.npz',200,True)
        env.reset(None);other.reset(None)
        self.assertEqual(env.initial_hash,other.initial_hash)
        for k,target in enumerate(env.targets):
            original=target.copy()
            original[env.ids]=np.clip(original[env.ids]+residual(
                features(other.sim)-env.reference[k],env.gains[env.phases[k]]),
                other.sim.lower[env.ids],other.sim.upper[env.ids])
            other.sim.step_target(original)
            env.step(np.zeros(10),auto_reset=False)
            np.testing.assert_array_equal(env.sim.prev,other.sim.prev)
            np.testing.assert_array_equal(env.sim.data.qpos,other.sim.data.qpos)

    def test_full_training_horizon_covers_actual_reference(self):
        env=ReferenceEpisode(str(SCENE),str(STAND),AUDIT/'frozen_path.npz',200)
        self.assertEqual(env.observe().shape,(55,))
        self.assertGreater(env.maximum_controls*.02,12.)
        self.assertLess(env.recovery_controls*.02,12.)
        self.assertTrue(env.initial['torso_contact'])
        self.assertLess(env.initial['up_z'],.5)

    def test_residual_validation(self):
        np.testing.assert_array_equal(checked_residual(np.zeros(10)),np.zeros(10))
        self.assertAlmostEqual(checked_residual(np.ones(10))[0],.18)
        for action in (np.zeros(14),np.full(10,np.nan),np.full(10,1.1)):
            with self.assertRaises(ValueError):checked_residual(action)

    def test_no_intermediate_root_write(self):
        tree=ast.parse(Path(__file__).with_name('getup_reference_env_r100.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        step=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
        for node in ast.walk(step):
            if isinstance(node,ast.Attribute):
                self.assertNotIn(node.attr,('restore','prepare','qpos','qvel','mj_resetData'))


if __name__=='__main__':unittest.main()
