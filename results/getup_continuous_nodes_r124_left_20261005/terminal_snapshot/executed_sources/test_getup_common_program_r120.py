import unittest
import numpy as np
from diagnostics.probe_getup_common_program_r120 import frozen_program


class CommonProgramTest(unittest.TestCase):
    def test_only_verified_fixed_recipe_is_exported(self):
        row = dict(group=2, full_group_pass=True, profile=0, knots=np.zeros((6, 10)).tolist())
        p, k = frozen_program(dict(groups=[row]))
        self.assertEqual(p, 0)
        self.assertEqual(k.shape, (6, 10))
        row['full_group_pass'] = False
        with self.assertRaises(ValueError):
            frozen_program(dict(groups=[row]))

    def test_node_bounds_not_relaxed(self):
        row = dict(group=2, full_group_pass=True, profile=1, knots=np.zeros((6, 10)).tolist())
        row['knots'][0][0] = 1.001
        with self.assertRaises(ValueError):
            frozen_program(dict(groups=[row]))


if __name__ == '__main__':
    unittest.main()
