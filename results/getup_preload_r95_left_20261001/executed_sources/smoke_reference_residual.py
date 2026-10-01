"""Compile one residual-controller reset and step before launching training."""

import jax
import jax.numpy as jp
import numpy as np

from playground.open_duck_mini_v2.focused_skill import focused_config
from playground.open_duck_mini_v2.reference_residual import ReferenceResidualBackward


def main():
    config = focused_config("backward")
    config.lin_vel_x = [-0.074, -0.074]
    env = ReferenceResidualBackward(
        residual_gain=0.08, ramp_s=1.0, task="flat_terrain", config=config
    )
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(84))
    state = step(state, jp.zeros(14))
    state.data.qpos.block_until_ready()
    assert np.isfinite(np.asarray(state.obs["state"])).all()
    assert np.isfinite(float(state.reward))
    assert env.action_size == 14
    assert np.asarray(state.obs["state"]).shape == (101,)
    print("REFERENCE RESIDUAL SMOKE: PASSED", flush=True)
    print("reward:", float(state.reward), "done:", float(state.done), flush=True)


if __name__ == "__main__":
    main()
