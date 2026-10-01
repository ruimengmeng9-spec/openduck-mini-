"""Sanity-check the getup curriculum: fallen start, no blow-ups, reward gradient."""

import jax
import jax.numpy as jp
import numpy as np

from playground.open_duck_mini_v2 import focused_skill

print("STANDING_BASE_HEIGHT constant:", focused_skill.STANDING_BASE_HEIGHT)

env = focused_skill.FocusedSkill(
    skill="getup",
    task="flat_terrain",
    config=focused_skill.focused_config("getup"),
)
print("home keyframe base height:", float(env._init_q[2]))
print("obs:", env.observation_size["state"][0], "act:", env.action_size)

zero_action = jp.zeros(env.action_size)
heights = []
gravity_z = []

for seed in range(6):
    state = env.reset(jax.random.PRNGKey(seed))
    h0 = float(env.get_floating_base_qpos(state.data.qpos)[2])
    g0 = float(env.get_gravity(state.data)[-1])
    heights.append(h0)
    gravity_z.append(g0)
    print(f"seed={seed} start_height={h0:.4f} start_gravity_z={g0:.4f}", end="")

    state = env.step(state, zero_action)
    for _ in range(80):
        state = env.step(state, zero_action)

    h1 = float(env.get_floating_base_qpos(state.data.qpos)[2])
    g1 = float(env.get_gravity(state.data)[-1])
    nan = bool(np.isnan(np.asarray(state.data.qpos)).any())
    print(
        f" | after 80 steps height={h1:.4f} gz={g1:.4f} nan={nan} "
        f"reward={float(state.reward):.5f} done={float(state.done):.0f}"
    )

print()
print("start height  : min=%.4f max=%.4f mean=%.4f" % (
    min(heights), max(heights), sum(heights) / len(heights)))
print("start gravity : min=%.4f max=%.4f mean=%.4f" % (
    min(gravity_z), max(gravity_z), sum(gravity_z) / len(gravity_z)))

# Reward composition on the very first step of a fallen episode.
state = env.reset(jax.random.PRNGKey(0))
state = env.step(state, zero_action)
print()
print("--- reward composition while lying (first step) ---")
for key, value in sorted(state.metrics.items()):
    print(f"  {key:34s} {float(value): .6f}")
