"""State-dependent exploration for the exact R35 PPO action likelihood."""
import numpy as np


def effective_log_std(observation, xp=np):
    if observation.shape[-1] != 50:
        raise ValueError('R35 exploration requires the 50D observation contract')
    # Observation index 5 is trunk up-vector Z. Deliberately keep the measured
    # safe sigma=.02 through mildly tilted states; grow noise only when down.
    up = observation[..., 5]
    sigma = .02 + .48 * xp.clip(.5 - up, 0., 1.)
    return xp.broadcast_to(xp.log(sigma)[..., None], observation.shape[:-1] + (14,))
