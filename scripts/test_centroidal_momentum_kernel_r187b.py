"""Offline geometry contract only; no dynamic smoke or qualification."""
import inspect
import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from diagnostics import centroidal_momentum_kernel_r187b as run
from diagnostics import probe_getup_phase_coordinate_r185b as old


class MomentumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(old.local.prior.program.SCENE))
        with np.load(old.local.prior.OUTPUT / 'snapshot/case_None/trajectory.npz', allow_pickle=False) as z:
            cls.nominal = z['observations'][:529, :34].copy()

    def setUp(self):
        self.g = run.MomentumGeometry(self.model)
        self.x = np.zeros(34, np.float32)

    def call(self, x=None, n=None, i=None, z=None, k=1, enabled=True):
        return run.momentum_request(self.x if x is None else x, self.x if n is None else n,
                                    self.x if i is None else i, self.x if z is None else z,
                                    self.g, k, enabled)

    def test_causal_signature(self):
        self.assertEqual(list(inspect.signature(run.momentum_request).parameters),
                         ['current','nominal','initial','initial_nominal','geometry','control','enabled'])
        self.assertEqual(run.DT, .02)
        self.assertEqual(run.PINV_RCOND, 1e-10)

    def test_manual_body_sum_matrix(self):
        rng = np.random.default_rng(287)
        for _ in range(12):
            x = rng.uniform(-.3, .3, 34).astype(np.float32)
            h, _, matrix, velocity = self.g.measure(x)
            d, m = self.g.data, self.model
            summed = np.zeros_like(matrix)
            jp = np.zeros_like(matrix); jr = np.zeros_like(matrix)
            for body in range(self.g.rootbody, m.nbody):
                if body > self.g.rootbody and m.body_parentid[body] < self.g.rootbody:
                    break
                mujoco.mj_jacBodyCom(m, d, jp, jr, body)
                rotation = d.ximat[body].reshape(3, 3)
                inertia = rotation @ np.diag(m.body_inertia[body]) @ rotation.T
                arm = d.xipos[body] - d.subtree_com[self.g.rootbody]
                summed += inertia @ jr + m.body_mass[body] * np.cross(arm, jp.T).T
            summed = d.xmat[self.g.trunk].reshape(3, 3).T @ summed
            np.testing.assert_allclose(matrix, summed, atol=1e-15, rtol=3e-13)
            np.testing.assert_allclose(h, summed @ velocity, atol=1e-15, rtol=3e-13)

    def test_translation_invariance_matrix(self):
        _, _, matrix, _ = self.g.measure(self.x)
        np.testing.assert_allclose(matrix[:, self.g.rootlinear], 0, atol=1e-15, rtol=0)

    def test_gyro_frame_reconstruction(self):
        x = self.x.copy(); x[:3] = [.2, -.3, .4]; x[20:34] = .007
        _, _, _, velocity = self.g.measure(x)
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.g.data, None, jr, self.g.site)
        reconstructed = self.g.data.site_xmat[self.g.site].reshape(3, 3).T @ (jr @ velocity)
        np.testing.assert_allclose(reconstructed, x[:3].astype(float), atol=1e-15, rtol=0)

    def test_native_velocity_scale(self):
        x = self.x.copy(); x[20:34] = np.arange(14, dtype=np.float32) * .001
        velocity = self.g.measure(x)[3]
        np.testing.assert_array_equal(velocity[self.g.vadr], x[20:34].astype(float)/.05)

    def test_actual_nominal_scalar_529_zero(self):
        for k, n in enumerate(self.nominal):
            request, error, _, residual = self.call(n, n, self.nominal[0], self.nominal[0], k)
            for value in (request, error, residual):
                np.testing.assert_array_equal(value, np.zeros_like(value))

    def test_control_zero_nonzero_initial_error(self):
        x = np.random.default_rng(287).uniform(-.2, .2, 34).astype(np.float32)
        for index in (0, 1, 3):
            value = self.call(x, self.x, x, self.x, 0)[index]
            np.testing.assert_array_equal(value, np.zeros_like(value))

    def test_no_upvector_or_extra_channels(self):
        x = self.x.copy(); x[3:6] = [7, 8, 9]
        for a, b in zip(self.call(x), self.call()):
            np.testing.assert_array_equal(a, b)

    def test_head_measured_but_not_commanded(self):
        x = self.x.copy(); x[:3] = [1, 2, 3]
        a = self.g.measure(x)[2]
        x[6 + self.g.head] = [.1, -.2, .3, -.4]
        b = self.g.measure(x)[2]
        self.assertGreater(np.max(np.abs(a-b)), 1e-8)
        np.testing.assert_array_equal(self.call(x)[0][self.g.head], np.zeros(4))

    def test_fixed_least_squares_allocation(self):
        rng = np.random.default_rng(287)
        for _ in range(20):
            x = rng.uniform(-.1, .1, 34).astype(np.float32)
            request, e, a, residual = self.call(x)
            np.testing.assert_allclose(request[self.g.legs], -.02*np.linalg.pinv(a,rcond=1e-10)@e, atol=0, rtol=0)
            self.assertLessEqual(np.linalg.norm(residual), np.linalg.norm(e)+1e-14)

    def test_original_gyro_weld_connection(self):
        self.assertNotEqual(self.g.gyrobody, self.g.trunk)
        self.assertEqual(self.model.body_weldid[self.g.gyrobody], self.model.body_weldid[self.g.trunk])
        self.g.measure(self.x)
        jr = np.zeros((3,self.model.nv))
        mujoco.mj_jacSite(self.model,self.g.data,None,jr,self.g.site)
        np.testing.assert_array_equal(jr[:,self.g.vadr], np.zeros((3,14)))

    def test_no_private_dynamics(self):
        with patch.object(mujoco, 'mj_forward', side_effect=AssertionError), \
             patch.object(mujoco, 'mj_step', side_effect=AssertionError), \
             patch.object(mujoco, 'mj_comVel', side_effect=AssertionError), \
             patch.object(mujoco, 'mj_subtreeVel', side_effect=AssertionError), \
             patch.object(mujoco, 'mj_contactForce', side_effect=AssertionError):
            self.call(np.random.default_rng(287).normal(size=34).astype(np.float32))
        self.assertEqual(self.g.data.time, 0.)
        np.testing.assert_array_equal(self.g.data.qpos[self.g.rootq], [0,0,0,1,0,0,0])
        np.testing.assert_array_equal(self.g.data.qvel, np.zeros(self.model.nv))

    def test_model_immutable(self):
        names = ('body_mass','body_inertia','body_pos','body_quat','jnt_range','geom_contype','geom_conaffinity','actuator_forcerange')
        before = {n: getattr(self.model,n).copy() for n in names}
        self.call(np.random.default_rng(287).normal(size=34).astype(np.float32))
        for n, value in before.items(): np.testing.assert_array_equal(getattr(self.model,n), value)

    def test_home_disabled_and_original_merge_bounds(self):
        x = np.random.default_rng(287).normal(size=34).astype(np.float32)
        for kwargs in ({'k':529}, {'enabled':False}):
            for value in self.call(x, **kwargs):np.testing.assert_array_equal(value,np.zeros_like(value))
        request = self.call(x)[0]; target = np.zeros(14)
        merged = old.local.merge_target(target,target,request,np.arange(14),-np.ones(14),np.ones(14))
        self.assertLessEqual(np.abs(merged).max(), .18+1e-12)
        self.assertIs(old.local.merge_target(target,target,np.zeros(14),np.arange(14),-np.ones(14),np.ones(14)),target)

    def test_invalid_inputs(self):
        for value in (np.zeros(33), np.full(34,np.nan)):
            with self.assertRaises(ValueError): self.call(value)
        for k in (-1, 1.5):
            with self.assertRaises(ValueError): self.call(k=k)


if __name__ == '__main__': unittest.main()
