import unittest
from importlib.util import find_spec
import numpy as np

from diagnostics.getup_temporal_noise_r33 import (
    RHO, INNOVATION_SCALE, sample_latent, conditional_log_prob, reset_noise)


class TemporalNoiseTests(unittest.TestCase):
    def test_sample_and_density_are_conditionally_consistent(self):
        previous = np.array([[1., -.5]], dtype=np.float32)
        gaussian = np.array([[.2, -1.]], dtype=np.float32)
        mean = np.array([[.4, -.1]], dtype=np.float32)
        log_std = np.log(np.array([[.5, .7]], dtype=np.float32))
        latent, noise = sample_latent(mean, log_std, previous, gaussian)
        np.testing.assert_allclose(noise, RHO * previous + INNOVATION_SCALE * gaussian)
        manual = np.sum(-.5 * gaussian**2 - log_std - np.log(INNOVATION_SCALE)
                        - .5 * np.log(2 * np.pi), axis=-1)
        np.testing.assert_allclose(conditional_log_prob(latent, mean, log_std, previous), manual,
                                   rtol=1e-6, atol=1e-6)
        altered = conditional_log_prob(latent, mean, log_std, -previous)
        self.assertGreater(float(abs(altered[0] - manual[0])), .1)

    def test_done_episodes_forget_noise_without_affecting_other_envs(self):
        noise = np.array([[.8, -.3], [.2, .4]], dtype=np.float32)
        result = reset_noise(noise, np.array([True, False]))
        np.testing.assert_array_equal(result[0], np.zeros(2))
        np.testing.assert_array_equal(result[1], noise[1])

    def test_actor_log_std_broadcasts_over_environment_batch(self):
        mean = np.zeros((4, 14), dtype=np.float32)
        std = np.full(14, np.log(.5), dtype=np.float32)
        previous = np.full_like(mean, .25)
        z, _ = sample_latent(mean, std, previous, np.zeros_like(mean))
        self.assertEqual(conditional_log_prob(z, mean, std, previous).shape, (4,))

    @unittest.skipUnless(find_spec('jax') is not None, 'JAX runtime not installed locally')
    def test_jit_and_policy_gradient_keep_conditional_density_finite(self):
        import jax
        import jax.numpy as jp
        observations = jp.zeros((4, 14), dtype=jp.float32)
        previous = jp.full((4, 14), .3, dtype=jp.float32)
        gaussian = jp.full((4, 14), -.1, dtype=jp.float32)
        std = jp.full((14,), np.log(.5), dtype=jp.float32)

        @jax.jit
        def evaluate(mean):
            latent, noise = sample_latent(mean, std, previous, gaussian, xp=jp)
            return conditional_log_prob(latent, mean, std, previous, xp=jp), noise

        values, noise = evaluate(observations)
        self.assertTrue(np.isfinite(np.asarray(values)).all())
        self.assertEqual(np.asarray(noise).shape, (4, 14))
        latent, _ = sample_latent(observations, std, previous, gaussian, xp=jp)
        gradient = jax.grad(lambda mean: jp.sum(conditional_log_prob(latent, mean, std, previous, xp=jp)))(observations)
        self.assertTrue(np.isfinite(np.asarray(gradient)).all())
        self.assertGreater(float(jp.max(jp.abs(gradient))), 0.)

    def test_noise_marginal_scale_and_temporal_correlation(self):
        rng = np.random.default_rng(33)
        previous = np.zeros((30000, 1), dtype=np.float32)
        means = np.zeros_like(previous)
        log_std = np.zeros_like(previous)
        for _ in range(30):
            _, current = sample_latent(means, log_std, previous,
                                       rng.normal(size=previous.shape).astype(np.float32))
            previous = current
        _, following = sample_latent(means, log_std, previous,
                                     rng.normal(size=previous.shape).astype(np.float32))
        self.assertAlmostEqual(float(previous.std()), 1., delta=.03)
        self.assertAlmostEqual(float(np.corrcoef(previous[:, 0], following[:, 0])[0, 1]), RHO, delta=.02)

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            conditional_log_prob(np.zeros((2, 14)), np.zeros((1, 14)),
                                 np.zeros((2, 14)), np.zeros((2, 14)))
        with self.assertRaises(ValueError):
            reset_noise(np.zeros((2, 14)), np.array([False]))


if __name__ == '__main__':
    unittest.main()
