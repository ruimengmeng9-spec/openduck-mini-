"""Regression checks for the two audited Oct 2 mixed-training defects."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from diagnostics.getup_fullfallen_env_r32 import FullFallEpisode
from diagnostics.getup_mixed_curriculum_env_r98 import MixedEpisode, label_completed_episode
from diagnostics.train_getup_mixed_ppo_r98 import mixed_reset_probabilities


class MixedRegression(unittest.TestCase):
    def test_reset_probability_contract_sums_to_one(self):
        for probability in (0.,.5,1.):
            rates=mixed_reset_probabilities(probability)
            self.assertAlmostEqual(sum(rates.values()),1.)
            self.assertEqual(rates['prefix_library'],probability)

    def test_prefix_not_counted_as_complete_fall(self):
        row = label_completed_episode(dict(pose='prefix_library', actual_fallen_start=True))
        self.assertFalse(row['actual_fallen_start'])
        self.assertTrue(row['mixed_prefix_start'])

    def test_all_real_fall_labels_retained(self):
        for pose in ('prone', 'supine', 'left_side', 'right_side'):
            row = label_completed_episode(dict(pose=pose, actual_fallen_start=True))
            self.assertTrue(row['actual_fallen_start'])
            self.assertFalse(row['mixed_prefix_start'])

    def test_terminal_label_does_not_read_next_reset_pose(self):
        episode = object.__new__(MixedEpisode)
        episode.pose = 'left_side'  # Next episode, after base step auto-reset.
        terminal = dict(pose='prefix_library', actual_fallen_start=True)
        with patch.object(FullFallEpisode, 'step', return_value=(None,0.,True,False,None,terminal,5)):
            row = episode.step(None)
        self.assertFalse(row[5]['actual_fallen_start'])
        self.assertTrue(row[5]['mixed_prefix_start'])

    def test_standing_never_becomes_fall(self):
        self.assertFalse(label_completed_episode(dict(pose='standing',actual_fallen_start=False))['actual_fallen_start'])

    def test_optimizer_uses_recorded_contract_rate(self):
        tree = ast.parse(Path(__file__).with_name('train_getup_mixed_ppo_r98.py').read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node,ast.Call)
                 and isinstance(node.func,ast.Attribute) and node.func.attr=='adam']
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].args[0]), "contract['optimizer_learning_rate']")

    def test_no_state_reset_in_corrected_step(self):
        tree = ast.parse(Path(__file__).with_name('getup_mixed_curriculum_env_r98.py').read_text())
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef))
        step = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='step')
        for node in ast.walk(step):
            if isinstance(node,ast.Attribute):
                self.assertNotIn(node.attr,('mj_resetData','qpos','qvel','prepare','reset'))


if __name__ == '__main__':
    unittest.main()
