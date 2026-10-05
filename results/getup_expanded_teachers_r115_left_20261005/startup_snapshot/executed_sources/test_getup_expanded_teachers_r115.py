import unittest
import numpy as np
from diagnostics.search_getup_expanded_teachers_r115 import make_proposals,EXPANDED


class ExpandedTeacherTest(unittest.TestCase):
    def test_deterministic_bounded_proposals(self):
        def state():return dict(rng=np.random.default_rng(215),profile=1,knots=np.zeros((6,10)),mean=np.zeros((6,10)),std=np.full((6,10),.1))
        a=make_proposals(state(),14);b=make_proposals(state(),14)
        self.assertEqual(len(a),14)
        for (p,k),(other,l) in zip(a,b):
            self.assertEqual(p,other);np.testing.assert_array_equal(k,l)
            self.assertIn(p,(0,1));self.assertEqual(k.shape,(6,10));self.assertTrue(np.isfinite(k).all());self.assertLessEqual(np.abs(k).max(),1.)
        self.assertEqual(len(set(EXPANDED)),40);self.assertEqual(EXPANDED[0],3160000);self.assertEqual(EXPANDED[-1],3160039)


if __name__=='__main__':unittest.main()
