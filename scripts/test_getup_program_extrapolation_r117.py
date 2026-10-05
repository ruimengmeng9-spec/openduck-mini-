import unittest
import numpy as np
from diagnostics.probe_getup_program_extrapolation_r117 import transform


class ExtrapolationTest(unittest.TestCase):
    def test_one_component_only_and_no_input_mutation(self):
        knots=np.linspace(-.2,.2,60).reshape(6,10);saved=knots.copy()
        p,k=transform(1,knots,'no_nodes');self.assertEqual(p,1);np.testing.assert_array_equal(k,np.zeros((6,10)))
        p,k=transform(1,knots,'no_feedback');self.assertEqual(p,0);np.testing.assert_array_equal(k,knots)
        p,k=transform(1,knots,'original');self.assertEqual(p,1);np.testing.assert_array_equal(k,knots)
        np.testing.assert_array_equal(knots,saved)
        with self.assertRaises(ValueError):transform(1,knots,'unknown')


if __name__=='__main__':unittest.main()
