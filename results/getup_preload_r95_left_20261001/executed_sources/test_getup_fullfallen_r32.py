import unittest
from pathlib import Path
import numpy as np
from diagnostics.getup_fullfallen_contract_r32 import potential, POSES, decode_action, accepted, completion_summary, bootstrap_observation, generalized_advantage


class FullFallenUnitTests(unittest.TestCase):
    def test_loaded_home_standing_has_higher_potential_than_knee_crouch(self):
        stand=dict(up_z=1.,height_m=.165,joint_home_error_mean_rad=0.,
                   foot_load_fraction=1.,foot_up_alignment_to_home=[1.,1.])
        crouch=stand|dict(height_m=.06,joint_home_error_mean_rad=.8,foot_load_fraction=.2)
        self.assertGreater(potential(stand,0),potential(crouch,0)+2.)

    def test_decoder_contract(self):
        lower=np.full(14,-2.);upper=np.full(14,3.)
        np.testing.assert_allclose(decode_action(np.zeros(14),lower,upper),.5)
        np.testing.assert_allclose(decode_action(-np.ones(14),lower,upper),lower)
        np.testing.assert_allclose(decode_action(np.ones(14),lower,upper),upper)
        for bad in (np.ones(13),np.full(14,np.nan),np.full(14,1.01)):
            with self.assertRaises(ValueError):decode_action(bad,lower,upper)

    def test_acceptance_is_not_training_discovery(self):
        self.assertFalse(accepted(True,True,True,50,.02))
        self.assertTrue(accepted(True,True,True,1500,.02))
        self.assertFalse(accepted(False,True,True,1500,.02))
        self.assertFalse(accepted(True,True,False,1500,.02))
        self.assertFalse(accepted(True,False,True,1500,.02))

    def test_complete_requires_twenty_trials_in_all_four_orientations(self):
        rows=[dict(pose=p,seed=i,success=i<18) for p in POSES for i in range(20)]
        self.assertTrue(completion_summary(rows)[1])
        self.assertFalse(completion_summary(rows[:-1])[1])
        rows[-3]['success']=False
        self.assertFalse(completion_summary(rows)[1])

    def test_repeated_trial_cannot_replace_independent_starts(self):
        rows=[dict(pose=p,seed=0,success=True) for p in POSES for i in range(20)]
        self.assertFalse(completion_summary(rows)[1])

    def test_true_terminal_does_not_feed_invalid_state_to_critic(self):
        bad=np.full(50,np.nan,dtype=np.float32)
        np.testing.assert_array_equal(bootstrap_observation(bad,True),np.zeros(50))
        with self.assertRaises(ValueError):bootstrap_observation(bad,False)
        final=np.arange(50,dtype=np.float32)
        np.testing.assert_array_equal(bootstrap_observation(final,False),final)

    def test_timeout_bootstraps_final_state_not_reset_or_next_episode(self):
        rewards=np.array([[1.],[100.]],np.float32)
        values=np.array([[2.],[0.]],np.float32)
        next_values=np.array([[5.],[0.]],np.float32)
        done=np.array([[True],[False]])
        terminated=np.array([[False],[False]])
        adv,_=generalized_advantage(rewards,values,next_values,done,terminated)
        self.assertAlmostEqual(float(adv[0,0]),3.95,places=5)
        terminated[0,0]=True
        adv,_=generalized_advantage(rewards,values,next_values,done,terminated)
        self.assertAlmostEqual(float(adv[0,0]),-1.,places=5)

    def test_gae_rejects_nonfinite_and_inconsistent_terminal_masks(self):
        zero=np.zeros((2,1),np.float32);mask=np.zeros((2,1),bool)
        bad=zero.copy();bad[0,0]=np.nan
        with self.assertRaises(ValueError):generalized_advantage(bad,zero,zero,mask,mask)
        with self.assertRaises(ValueError):generalized_advantage(zero,zero,zero,mask,~mask)


@unittest.skipUnless(Path('/data/shijinsheng/open_duck/training/getup_decomposed_r4/model/scene.xml').exists(),'server physics assets required')
class FullFallenPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from diagnostics.getup_fullfallen_env_r32 import FullFallEpisode
        from diagnostics.validate_getup_fullpath_r27 import StrictSim
        root=Path('/data/shijinsheng/open_duck')
        cls.scene=root/'training/getup_decomposed_r4/model/scene.xml'
        cls.stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
        cls.env=FullFallEpisode(cls.scene,cls.stand,1232)
        cls.reference=StrictSim(cls.scene,cls.stand)

    def test_fallen_is_not_an_early_termination(self):
        for pose in POSES:
            e=self.env; e.reset(pose)
            a=(e.sim.home-(e.sim.lower+e.sim.upper)/2)/((e.sim.upper-e.sim.lower)/2)
            obs,r,done,terminated,boot,info,controls=e.step(a)
            self.assertFalse(terminated,pose)
            self.assertFalse(done,pose)
            self.assertEqual(controls,5)
            self.assertEqual(obs.shape,(50,))
            self.assertTrue(np.isfinite(obs).all() and np.isfinite(r))

    def test_five_motor_updates_exactly_match_existing_substep_dynamics(self):
        e=self.env; e.reset('prone')
        ref=self.reference; ref.prepare('prone',e.reset_seed,True); ref.clear_audit()
        target=e.sim.home.copy(); target[5]+=.05
        a=(target-(e.sim.lower+e.sim.upper)/2)/((e.sim.upper-e.sim.lower)/2)
        e.step(a)
        for _ in range(5):
            ref.step_target(target)
        np.testing.assert_allclose(e.sim.data.qpos,ref.data.qpos,atol=1e-12,rtol=0)
        self.assertTrue(e.sim.physical_valid())


if __name__=='__main__':
    unittest.main()
