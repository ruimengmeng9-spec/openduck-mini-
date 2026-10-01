"""Jitted sanity check for the getup curriculum: fallen start, no blow-ups."""

import jax
import jax.numpy as jp
import numpy as np

from playground.open_duck_mini_v2 import focused_skill

STEPS = 60

env = focused_skill.FocusedSkill(
    skill="getup",
    task="flat_terrain",
    config=focused_skill.focused_config("getup"),
)
zero_action = jp.zeros(env.action_size)
print("home height:", float(env._init_q[2]))


def rollout(state):
    def body(carry, _):
        nxt = env.step(carry, zero_action)
        return nxt, (
            env.get_floating_base_qpos(nxt.data.qpos)[2],
            env.get_gravity(nxt.data)[-1],
        )

    return jax.lax.scan(body, state, None, length=STEPS)


rollout_jit = jax.jit(rollout)

for seed in range(4):
    state = env.reset(jax.random.PRNGKey(seed))
    h0 = float(env.get_floating_base_qpos(state.data.qpos)[2])
    g0 = float(env.get_gravity(state.data)[-1])
    final, (hs, gzs) = rollout_jit(state)
    hs = np.asarray(hs)
    gzs = np.asarray(gzs)
    nan = bool(
        np.isnan(np.asarray(final.data.qpos)).any()
        or np.isnan(np.asarray(final.data.qvel)).any()
    )
    print(
        "seed=%d start h=%.4f gz=%.4f | h[min=%.4f max=%.4f end=%.4f] "
        "gz[min=%.3f max=%.3f end=%.3f] nan=%s"
        % (seed, h0, g0, hs.min(), hs.max(), hs[-1], gzs.min(), gzs.max(), gzs[-1], nan)
    )
    if seed == 0:
        print("  final reward:", float(final.reward))
        for key, value in sorted(final.metrics.items()):
            print("    %-34s % .6f" % (key, float(value)))
