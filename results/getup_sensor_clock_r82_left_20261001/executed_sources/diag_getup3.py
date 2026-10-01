"""Validate the getup curriculum on the collision-enabled model."""
import jax
import jax.numpy as jp
import numpy as np

from playground.open_duck_mini_v2 import focused_skill

STEPS = 60
env = focused_skill.FocusedSkill(
    skill="getup", task="flat_terrain_getup",
    config=focused_skill.focused_config("getup"),
)
zero = jp.zeros(env.action_size)


def rollout(state):
    def body(carry, _):
        nxt = env.step(carry, zero)
        return nxt, (
            env.get_floating_base_qpos(nxt.data.qpos)[2],
            env.get_gravity(nxt.data)[-1],
        )
    return jax.lax.scan(body, state, None, length=STEPS)


rollout_jit = jax.jit(rollout)
for seed in range(5):
    s = env.reset(jax.random.PRNGKey(seed))
    h0 = float(env.get_floating_base_qpos(s.data.qpos)[2])
    g0 = float(env.get_gravity(s.data)[-1])
    fin, (hs, gz) = rollout_jit(s)
    hs, gz = np.asarray(hs), np.asarray(gz)
    nan = bool(np.isnan(np.asarray(fin.data.qpos)).any())
    # A settling robot should stay above the floor and remain roughly at its
    # initial attitude (no explosions), which is what the curriculum needs.
    print("seed=%d start h=%.4f gz=%+.3f | h[min=%+.4f max=%.4f end=%.4f] "
          "gz[min=%+.3f end=%+.3f] nan=%s"
          % (seed, h0, g0, hs.min(), hs.max(), hs[-1], gz.min(), gz[-1], nan))
    if seed == 0:
        print("  reward=%.4f" % float(fin.reward))
        for k, v in sorted(fin.metrics.items()):
            print("    %-34s % .6f" % (k, float(v)))
