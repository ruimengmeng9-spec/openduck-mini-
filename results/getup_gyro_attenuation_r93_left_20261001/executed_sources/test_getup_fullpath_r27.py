import unittest
import numpy as np
from diagnostics.train_getup_fullpath_r27 import decode, encode, progress
from diagnostics.validate_getup_fullpath_r27 import strict_standing


class DummySim:
    lower = np.full(14, -1.)
    upper = np.full(14, 1.)
    home = np.zeros(14)


class FullPathTests(unittest.TestCase):
    def test_final_gate_retains_standing_height_window(self):
        m = dict(stable=True, height_m=.1539, joint_home_error_mean_rad=.1)
        self.assertFalse(strict_standing(m))
        m['height_m'] = .165
        self.assertTrue(strict_standing(m))
        m['joint_home_error_mean_rad'] = .25
        self.assertFalse(strict_standing(m))
        m['joint_home_error_mean_rad'] = .1
        m['stable'] = False
        self.assertFalse(strict_standing(m))

    def test_decoder_preserves_limits_and_duration(self):
        x = np.full((14, 15), 8.)
        q, d = decode(x, DummySim())
        self.assertTrue(np.all(q <= 1))
        self.assertTrue(np.all(q >= -1))
        self.assertTrue(np.all(q[:, [7, 8]] == 0))
        self.assertTrue(np.all(d == .9))

    def test_round_trip(self):
        sim = DummySim()
        q = np.zeros((14, 14)); q[:, 3] = .3
        d = np.full(14, .24)
        q2, d2 = decode(encode(q, d, sim), sim)
        np.testing.assert_allclose(q, q2)
        np.testing.assert_allclose(d, d2)

    def test_knee_supported_crouch_not_rewarded_as_stand(self):
        crouch = dict(up_z=1., height_m=.06, foot_load_fraction=.2,
                      foot_up_alignment_to_home=[.4, .4], joint_home_error_mean_rad=1.2,
                      torso_contact=False)
        stand = dict(crouch, height_m=.165, foot_load_fraction=1.,
                     foot_up_alignment_to_home=[1., 1.], joint_home_error_mean_rad=0.)
        self.assertGreater(progress(stand), progress(crouch)+4)


if __name__ == '__main__':
    unittest.main()
