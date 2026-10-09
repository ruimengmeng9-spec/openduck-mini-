"""Correct the branch-coupling expectation, never the fixed R190 formula."""
from pathlib import Path
import unittest
import numpy as np
from diagnostics import test_velocity_bias_kernel_r190 as prior_tests

old=prior_tests.old
run=prior_tests.run
capture_all_sources=prior_tests.capture_all_sources


class BiasCorrectionTests(prior_tests.BiasTests):
    def test_head_inertia_without_new_head_commands(self):
        # With prescribed base velocity and qacc=0, sibling-branch velocity
        # does not enter a leg's generalized RNE torque. This is a structural
        # limitation, not whole-body mass/inertia coordination.
        x=self.sample(); a=self.g.measure(x)[0]
        full_a=prior_tests.independent_velocity_bias(self.model,self.g.data)
        x[20+self.g.head]+=.1
        b=self.g.measure(x)[0]
        full_b=prior_tests.independent_velocity_bias(self.model,self.g.data)
        np.testing.assert_array_equal(a[self.g.legs],b[self.g.legs])
        self.assertGreater(np.max(np.abs(a[self.g.head]-b[self.g.head])),1e-8)
        self.assertGreater(np.max(np.abs(full_a[self.g.rootangular]-full_b[self.g.rootangular])),1e-8)
        np.testing.assert_array_equal(self.call(x)[0][self.g.head],np.zeros(4))
        self.proof['head_velocity_leg_bias_difference_nm']=float(np.max(np.abs(a[self.g.legs]-b[self.g.legs])))
        self.proof['head_velocity_head_bias_difference_nm']=float(np.max(np.abs(a[self.g.head]-b[self.g.head])))
        self.proof['head_velocity_root_angular_reaction_difference_nm']=float(np.max(np.abs(full_a[self.g.rootangular]-full_b[self.g.rootangular])))

    def test_immutable_original_kernel_source(self):
        expected='44d1f4881ac612e66c2a99fa7bceab16ce0fd4657f3d6bb2089077db3f2d07fa'
        self.assertEqual(old.prior.digest(Path(run.__file__)),expected)


if __name__=='__main__': unittest.main(verbosity=2)
