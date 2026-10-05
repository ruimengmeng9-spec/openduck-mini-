"""Sensor-only boundaries and scalar zero preservation before dynamics."""
import unittest
import numpy as np
from diagnostics.train_getup_local_hip_r130 import local_feedback,merge_target


class LocalHipTests(unittest.TestCase):
    def test_scalar_nominal_exact_zero(self):
        obs=np.linspace(-1,1,55,dtype=np.float32)
        for gains in (np.zeros(6),np.ones(6),np.full(6,-2.)):
            np.testing.assert_array_equal(local_feedback(obs,obs,[3,4,5],gains),np.zeros(3))

    def test_only_selected_joint_position_velocity(self):
        nominal=np.zeros(55,np.float32);current=nominal.copy();gains=np.ones(6)
        irrelevant=[i for i in range(55) if i not in (9,10,11,23,24,25)]
        current[irrelevant]=12.
        np.testing.assert_array_equal(local_feedback(current,nominal,[3,4,5],gains),np.zeros(3))
        current[9]=.05
        self.assertAlmostEqual(local_feedback(current,nominal,[3,4,5],gains)[0],-.18*np.tanh(1.),places=7)
        current[23]=-.05
        np.testing.assert_array_equal(local_feedback(current,nominal,[3,4,5],gains),np.zeros(3))

    def test_finite_bounds_and_indices(self):
        obs=np.zeros(55,np.float32)
        for indices,gains in [([1,1,2],np.ones(6)),([1,2,14],np.ones(6)),([1,2,3],np.full(6,2.01)),([1,2,3],np.full(6,np.nan))]:
            with self.assertRaises(ValueError):local_feedback(obs,obs,indices,gains)
        broken=obs.copy();broken[0]=np.inf
        with self.assertRaises(ValueError):local_feedback(broken,obs,[1,2,3],np.ones(6))
        self.assertLessEqual(np.abs(local_feedback(np.full(55,1e4,np.float32),obs,[1,2,3],np.full(6,2.))).max(),.18)

    def test_merge_zero_preserves_original_object(self):
        target=np.arange(14,dtype=float);base=target.copy()
        self.assertIs(merge_target(target,base,np.zeros(3),[1,2,3],target-1,target+1),target)

    def test_combined_cap_and_untouched_other_joints(self):
        base=np.zeros(14);target=np.full(14,.15);ids=np.array([1,2,3])
        result=merge_target(target,base,np.array([.18,-.18,.1]),ids,np.full(14,-2.),np.full(14,2.))
        self.assertLessEqual(np.abs(result[ids]-base[ids]).max(),.18)
        np.testing.assert_array_equal(result[np.setdiff1d(np.arange(14),ids)],target[np.setdiff1d(np.arange(14),ids)])
        self.assertEqual(result[1],.18)
        np.testing.assert_array_equal(target,np.full(14,.15))


if __name__=='__main__':unittest.main()
