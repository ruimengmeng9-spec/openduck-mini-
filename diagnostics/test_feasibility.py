"""Is standing up statistically reachable at all? Random-action ceiling test.

If random action sequences can reach an upright attitude from a fallen start,
the task is a learning problem.  If no random sequence ever gets close, the
model or the start distribution is blocking recovery physically.
"""
import jax
import jax.numpy as jp
import numpy as np

from playground.open_duck_mini_v2 import focused_skill

env = focused_skill.FocusedSkill("getup", task="flat_terrain_getup",
                                config=focused_skill.focused_config("getup"))
STEPS = 500


def rollout(state, rng):
    def body(carry, _):
        s, r = carry
        r, a = jax.random.split(r)
        act = jax.random.uniform(a, (env.action_size,), minval=-1.0, maxval=1.0)
        nxt = env.step(s, act)
        return (nxt, r), env.get_gravity(nxt.data)[-1]
    return jax.lax.scan(body, (state, rng), None, length=STEPS)


roll = jax.jit(rollout)
peaks = []
for seed in range(12):
    s = env.reset(jax.random.PRNGKey(seed))
    (fin, _), gz = roll(s, jax.random.PRNGKey(1000 + seed))
    gz = np.asarray(gz)
    peaks.append(float(gz.max()))
    print("seed=%2d start_up=%+.3f  random-action peak up_z=%+.3f  (up>0.85 reached: %s)"
          % (seed, float(env.get_gravity(s.data)[-1]), gz.max(), gz.max() > 0.85))
print()
print("best random peak up_z: %.3f" % max(peaks))
print("episodes reaching up_z>0.85 with random actions: %d/%d" % (sum(p > 0.85 for p in peaks), len(peaks)))
print("episodes reaching up_z>0.50 with random actions: %d/%d" % (sum(p > 0.50 for p in peaks), len(peaks)))
