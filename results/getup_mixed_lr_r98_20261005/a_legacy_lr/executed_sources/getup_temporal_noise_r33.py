"""Conditionally correct AR(1) latent exploration for independent get-up PPO."""
import numpy as np

RHO = .9
INNOVATION_SCALE = float(np.sqrt(1. - RHO * RHO))


def sample_latent(mean, log_std, previous_noise, gaussian, xp=np):
    if not (mean.shape == previous_noise.shape == gaussian.shape
            and log_std.shape in (mean.shape, mean.shape[-1:])):
        raise ValueError('Temporal noise arrays need matching shapes')
    noise = RHO * previous_noise + INNOVATION_SCALE * gaussian
    return mean + xp.exp(log_std) * noise, noise


def conditional_log_prob(latent, mean, log_std, previous_noise, xp=np):
    if not (latent.shape == mean.shape == previous_noise.shape
            and log_std.shape in (mean.shape, mean.shape[-1:])):
        raise ValueError('Conditional density arrays need matching shapes')
    conditional_mean = mean + xp.exp(log_std) * RHO * previous_noise
    conditional_log_std = log_std + np.log(INNOVATION_SCALE)
    return xp.sum(-.5 * ((latent - conditional_mean) / xp.exp(conditional_log_std))**2
                  - conditional_log_std - .5 * np.log(2 * np.pi), axis=-1)


def reset_noise(noise, done, xp=np):
    if done.shape != noise.shape[:-1]:
        raise ValueError('A new episode must reset exactly its own latent noise')
    return xp.where(done[..., None], xp.zeros_like(noise), noise)
