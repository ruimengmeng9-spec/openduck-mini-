import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from diagnostics.getup_dynamic_beam import SelfCollisionSim
from diagnostics.getup_crouch_extension import evaluate
from diagnostics.build_getup_decomposed import text_array
from diagnostics.getup_aligned_extension import strict_entry


class CollisionSearchTests(unittest.TestCase):
    def test_self_penetration_cannot_be_hidden_by_clean_floor_contact(self):
        sim = SelfCollisionSim.__new__(SelfCollisionSim)
        sim.floor = 0
        sim.data = SimpleNamespace(contact=[SimpleNamespace(geom=np.array([2, 3]), dist=-.003)])
        base = dict(penetration_m=.0001, stable=True)
        with patch('diagnostics.getup_independent_native.RecoverySim.measure', return_value=base):
            result = sim.measure()
        self.assertAlmostEqual(result['floor_penetration_m'], .0001)
        self.assertAlmostEqual(result['self_penetration_m'], .003)
        self.assertAlmostEqual(result['penetration_m'], .003)
        self.assertFalse(result['stable'])

    def test_generated_collision_coordinates_preserve_precision(self):
        points = np.array([[.00123456789, -.103847162, .29], [-.004, .015, .1]])
        decoded = np.fromstring(text_array(points), sep=' ').reshape(points.shape)
        np.testing.assert_allclose(decoded, points, rtol=1e-9, atol=1e-11)

    def test_lift_reward_prefers_upright_height_not_lying_height(self):
        def score(up, height):
            m = dict(up_z=up, height_m=height, feet=[True, True], torso_contact=False,
                     angular_speed_rad_s=0., linear_speed_mps=0.)
            result = dict(measurements=[m]*8, metric=m, maximum_penetration_m=0.)
            with patch('diagnostics.getup_crouch_extension.primitive', return_value=result):
                return evaluate(({}, np.zeros(14), .54))['score']
        self.assertGreater(score(.99, .14), score(.99, .06))
        self.assertGreater(score(.99, .06), score(0., .2))

    def test_standing_entry_requires_pose_and_motor_alignment(self):
        m=dict(up_z=.99,height_m=.16,feet=[True,True],torso_contact=False,
               linear_speed_mps=.01,angular_speed_rad_s=.1,self_penetration_m=0.,
               joint_home_error_max_rad=.1,motor_target_home_error_max_rad=.1)
        self.assertTrue(strict_entry(m))
        self.assertFalse(strict_entry(dict(m,joint_home_error_max_rad=2.)))
        self.assertFalse(strict_entry(dict(m,motor_target_home_error_max_rad=2.)))
        self.assertFalse(strict_entry(dict(m,feet=[True,False])))
        self.assertFalse(strict_entry(dict(m,height_m=.06)))


if __name__ == '__main__':
    unittest.main()
