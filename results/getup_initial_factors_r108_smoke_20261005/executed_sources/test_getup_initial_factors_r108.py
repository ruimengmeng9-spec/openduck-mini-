"""Non-physics regression checks; full dynamics parity is mandatory in runner."""
import unittest
import numpy as np
from diagnostics.probe_getup_initial_factors_r108 import draw_components


class InitialFactorsTest(unittest.TestCase):
    def test_original_draw_order(self):
        rng=np.random.default_rng(769000)
        expected=(rng.uniform(-.05,.05),rng.uniform(-.02,.02,14),rng.uniform(-.01,.01,20))
        actual=draw_components(769000,14,20)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)

    def test_reproducible_and_finite(self):
        for seed in (0,769000,773015):
            a,b=draw_components(seed,14,20),draw_components(seed,14,20)
            for x,y in zip(a,b):
                np.testing.assert_array_equal(x,y)
                self.assertTrue(np.isfinite(x).all())

    def test_bounds(self):
        for seed in range(100):
            a,b,c=draw_components(seed,14,20)
            self.assertLessEqual(abs(a),.05)
            self.assertLessEqual(np.max(abs(b)),.02)
            self.assertLessEqual(np.max(abs(c)),.01)


if __name__=='__main__':unittest.main()
