import unittest
import numpy as np
from diagnostics.distill_getup_rescue_r23 import split_episodes,physical_contract
from diagnostics.eval_getup_distill_r23 import home_phase


class DatasetSplitTest(unittest.TestCase):
    def test_whole_episode_split(self):
        seeds=np.repeat(np.arange(30),200)
        train,valid=split_episodes(seeds)
        self.assertFalse(bool(np.any(train & valid)))
        self.assertTrue(bool(np.all(train | valid)))
        self.assertEqual(len(set(seeds[valid])),6)
        self.assertFalse(bool(set(seeds[valid]) & set(seeds[train])))

    def test_small_dataset_rejected(self):
        with self.assertRaises(RuntimeError):split_episodes(np.arange(4))

    def test_source_ppo_recipe_not_inherited(self):
        c=physical_contract(dict(scene_path='scene.xml',home_rad=[0.],iterations=128,
            num_envs=8,initial_std=.03,learning_rate=1e-4))
        self.assertEqual(c,dict(scene_path='scene.xml',home_rad=[0.]))

    def test_home_tail_only_after_exact_1p2s(self):
        self.assertFalse(home_phase('student_home_tail',59))
        self.assertTrue(home_phase('student_home_tail',60))
        self.assertFalse(home_phase('student',100))
        self.assertTrue(home_phase('home',0))


if __name__=='__main__':unittest.main()
