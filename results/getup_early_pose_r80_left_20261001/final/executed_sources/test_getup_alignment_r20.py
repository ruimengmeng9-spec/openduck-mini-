import json
from pathlib import Path
import unittest
import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode,reward_terms
from diagnostics.getup_alignment_curriculum_r20 import AlignmentEpisode,training_reward_terms


class AlignmentTests(unittest.TestCase):
    def metric(self,height):
        return dict(height_m=height,up_z=1.,foot_load_fraction=1.,stable=True,
                    joint_home_error_mean_rad=0.,joint_home_error_max_rad=0.)

    def test_base_reward_exactly_preserved(self):
        m=self.metric(.154)
        self.assertEqual(training_reward_terms(m,.153,0.,'base'),reward_terms(m,.153,0.))

    def test_height_reward_separates_low_pose_without_relaxing_goal(self):
        for mode in ('base','height'):
            low=training_reward_terms(self.metric(.154),.154,0.,mode)
            high=training_reward_terms(self.metric(.165),.165,0.,mode)
            self.assertEqual(low['stable'],0.);self.assertEqual(high['stable'],1.)
        old=sum(training_reward_terms(self.metric(.165),.165,0.,'base').values())-sum(training_reward_terms(self.metric(.154),.154,0.,'base').values())
        new=sum(training_reward_terms(self.metric(.165),.165,0.,'height').values())-sum(training_reward_terms(self.metric(.154),.154,0.,'height').values())
        self.assertGreater(new,old)

    def test_same_native_physics_and_reset_boundaries(self):
        root=Path('/data/shijinsheng/open_duck')
        c=json.loads((root/'training/getup_residual_ppo_r18/controller_contract.json').read_text())
        stand=str(root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
        original=NativeEpisode(c['scene_path'],stand,c['starts_rad'],420,.55)
        base=AlignmentEpisode(c['scene_path'],stand,c['starts_rad'],420,.55,reward_mode='base')
        height=AlignmentEpisode(c['scene_path'],stand,c['starts_rad'],420,.55,reward_mode='height')
        action=(original.sim.home-(original.sim.lower+original.sim.upper)/2)/((original.sim.upper-original.sim.lower)/2)
        for _ in range(205):
            a=original.step(action);b=base.step(action);h=height.step(action)
            for e in (base,height):
                np.testing.assert_array_equal(original.sim.data.qpos,e.sim.data.qpos)
                np.testing.assert_array_equal(original.sim.data.qvel,e.sim.data.qvel)
            self.assertEqual(a[1],b[1]);self.assertEqual(a[2:4],b[2:4]);self.assertEqual(a[2:4],h[2:4])
            np.testing.assert_array_equal(a[4],h[4])
            if a[5] is not None:
                self.assertEqual(a[5]['success'],h[5]['success'])


if __name__=='__main__':unittest.main()
