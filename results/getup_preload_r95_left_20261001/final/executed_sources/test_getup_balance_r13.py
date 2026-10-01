import unittest
from types import SimpleNamespace
import numpy as np

from diagnostics.getup_balance_search_r13 import feedback_target, home_axis_map
from diagnostics.getup_native_curriculum import CurriculumSim
from diagnostics.getup_smooth_transition_r14 import smooth_target


class BalanceTests(unittest.TestCase):
    def test_smoothing_is_a_target_transition_not_physics_change(self):
        sim=SimpleNamespace(home=np.zeros(14),prev=np.ones(14)*.4)
        saved=sim.prev.copy()
        actual=smooth_target(sim,.3)
        np.testing.assert_allclose(actual,saved*np.exp(-.02/.3))
        np.testing.assert_array_equal(sim.prev,saved)
        np.testing.assert_array_equal(smooth_target(sim,0.),sim.home)
        with self.assertRaises(ValueError): smooth_target(sim,-1.)

    def test_zero_gains_return_home(self):
        sim=SimpleNamespace(home=np.arange(14)/100,lower=np.full(14,-1.),upper=np.full(14,1.),
                            sensor=lambda name:np.array([.2,.3,.9]))
        axes={k:np.array([1.,-1.]) for k in ('roll','hip_pitch','ankle_pitch')}
        np.testing.assert_array_equal(feedback_target(sim,np.zeros(6),axes),sim.home)

    def test_feedback_is_bounded_by_original_limits(self):
        sim=SimpleNamespace(home=np.zeros(14),lower=np.full(14,-.2),upper=np.full(14,.2),
                            sensor=lambda name:np.array([100.,100.,.9]))
        axes={k:np.array([1.,-1.]) for k in ('roll','hip_pitch','ankle_pitch')}
        actual=feedback_target(sim,np.ones(6),axes)
        self.assertTrue(np.all(actual<=sim.upper)); self.assertTrue(np.all(actual>=sim.lower))
        np.testing.assert_array_equal(actual[[0,3,5,6,7,8,9,12]],0.)

    def test_actual_home_joint_axis_projection(self):
        sim=CurriculumSim('/data/shijinsheng/open_duck/training/getup_decomposed_r4/model/scene.xml',
                          '/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
        axes=home_axis_map(sim)
        print('HOME FEEDBACK AXES:',{k:v.tolist() for k,v in axes.items()},flush=True)
        for vector in axes.values(): self.assertTrue(np.all(np.abs(vector)>.9))


if __name__=='__main__': unittest.main()
