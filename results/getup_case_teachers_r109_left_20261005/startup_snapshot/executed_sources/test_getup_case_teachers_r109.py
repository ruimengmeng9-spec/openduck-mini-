import unittest
import numpy as np
from diagnostics.search_getup_case_teachers_r109 import knot_action,NODES


class TeacherKnotsTest(unittest.TestCase):
    def test_zero_exact(self):
        for k in range(2279):np.testing.assert_array_equal(knot_action(np.zeros((6,10)),k),np.zeros(10))

    def test_nodes_and_home(self):
        x=np.linspace(-1,1,60).reshape(6,10)
        for i,k in enumerate(NODES[1:-1]):np.testing.assert_array_equal(knot_action(x,k),x[i])
        for k in (0,529,530,2278):np.testing.assert_array_equal(knot_action(x,k),np.zeros(10))

    def test_convex_bounds(self):
        x=np.random.default_rng(209).uniform(-1,1,(6,10))
        for k in range(529):self.assertLessEqual(abs(knot_action(x,k)).max(),1.)

    def test_invalid(self):
        for x in (np.zeros((5,10)),np.full((6,10),np.nan),np.full((6,10),1.01)):
            with self.assertRaises(ValueError):knot_action(x,10)


if __name__=='__main__':unittest.main()
