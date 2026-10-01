import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from diagnostics.audit_getup_load_support import LoadSupportSim
from diagnostics.getup_anatomical_search import anatomical_library
from diagnostics.getup_load_search import loaded_entry, quality


class LoadSupportTests(unittest.TestCase):
    def sample(self):
        return dict(up_z=.99,height_m=.16,feet=[True,True],torso_contact=False,
                    linear_speed_mps=.01,angular_speed_rad_s=.1,self_penetration_m=0.,
                    joint_home_error_max_rad=.2,motor_target_home_error_max_rad=.2,
                    joint_home_error_mean_rad=.1,foot_up_alignment_to_home=[.99,.99],
                    foot_normal_forces_n=[10.,10.],foot_load_fraction=1.)

    def test_entry_rejects_knee_supported_contact_flags(self):
        m=self.sample()
        self.assertTrue(loaded_entry(m))
        self.assertFalse(loaded_entry(dict(m,foot_load_fraction=.21)))
        self.assertFalse(loaded_entry(dict(m,foot_normal_forces_n=[0.,20.])))
        self.assertFalse(loaded_entry(dict(m,foot_up_alignment_to_home=[.9,-.4])))

    def test_quality_prefers_loaded_feet_over_loaded_knees(self):
        m=self.sample()
        self.assertGreater(quality(m),quality(dict(m,foot_load_fraction=.2)))

    def test_contact_force_accounting_changes_stability(self):
        sim=LoadSupportSim.__new__(LoadSupportSim)
        sim.floor,sim.feet=0,[1,2]
        sim.model=SimpleNamespace(geom_bodyid=np.array([0,1,2,3]),
                                  body=lambda b:SimpleNamespace(name=f'body_{b}'))
        sim.data=SimpleNamespace(contact=[SimpleNamespace(geom=np.array([0,i]),dist=0.) for i in (1,2,3)])
        base=dict(self.sample(),stable=True)
        def run(forces):
            def force(model,data,index,out):
                out[0]=forces[index]
            with patch('diagnostics.audit_getup_crouch.SupportSim.measure',return_value=base.copy()), \
                    patch('diagnostics.audit_getup_load_support.mujoco.mj_contactForce',side_effect=force):
                return sim.measure()
        self.assertTrue(run([10.,10.,0.])['stable'])
        r=run([1.,1.,18.])
        self.assertFalse(r['stable'])
        self.assertAlmostEqual(r['foot_load_fraction'],.1)
        self.assertEqual(r['nonfoot_loaded_bodies'],{'body_3':18.})

    def test_normal_knee_curriculum_does_not_change_physical_limits(self):
        lower,upper=np.full(14,-1.57),np.full(14,1.57)
        sim=SimpleNamespace(home=np.zeros(14),lower=lower,upper=upper,model=SimpleNamespace(nu=14))
        original=lower.copy()
        targets=anatomical_library(sim,np.zeros(14),np.random.default_rng(1))
        self.assertGreaterEqual(targets[:,[3,12]].min(),0.)
        np.testing.assert_array_equal(lower,original)
        self.assertTrue(np.all(targets>=lower))
        self.assertTrue(np.all(targets<=upper))


if __name__=='__main__':
    unittest.main()
