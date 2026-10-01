"""Measure per-term getup reward at a standing pose vs a fallen pose."""
import os
os.environ.setdefault("GETUP_FALLEN_PROB", "0.0")
os.environ.setdefault("GETUP_LEAN_TILT", "0.35,0.70")
os.environ.setdefault("GETUP_RANDOM_JOINTS", "0")

import jax
import jax.numpy as jp
import numpy as np
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env

from playground.open_duck_mini_v2 import focused_skill

cfg = focused_skill.focused_config("getup")
env = focused_skill.FocusedSkill("getup", task="flat_terrain_getup", config=cfg)
print("scales of interest:")
for k in ("stand_up","height_progress","upright","tilt","termination","alive","imitation","joint_velocity","action_rate"):
    print("   %-18s %s" % (k, cfg.reward_config.scales.get(k)))

zero = jp.zeros(env.action_size)

def report(tag, state):
    st = env.step(state, zero)
    gz = float(env.get_gravity(st.data)[-1])
    h = float(env.get_floating_base_qpos(st.data.qpos)[2])
    print("  %s: up_z=%.3f height=%.4f reward=%.3f" % (tag, gz, h, float(st.reward)))
    terms = {}
    for k, v in sorted(st.metrics.items()):
        terms[k] = float(v)
    return st, terms

# Case A: exact home standing pose under the normal (upstream) init path.
base_env = focused_skill.walk_turn_stop.WalkTurnStop(task="flat_terrain", config=cfg)
st_home = base_env.reset(jax.random.PRNGKey(0))
print("\n--- HOME standing pose (upstream reset, not the getup one) ---")
print("   up_z=%.3f height=%.4f" % (
    float(base_env.get_gravity(st_home.data)[-1]),
    float(base_env.get_floating_base_qpos(st_home.data.qpos)[2])))
_, terms_home = report("home", st_home)
for k, v in sorted(terms_home.items()):
    print("       %-32s % .6f" % (k, v))

# Case B: the getup environment's own (fallen/leaning) start.
st_fall = env.reset(jax.random.PRNGKey(0))
print("\n--- GETUP reset pose ---")
_, terms_fall = report("getup-reset", st_fall)
for k, v in sorted(terms_fall.items()):
    print("       %-32s % .6f" % (k, v))

# Case C: drive the getup env to home pose and measure again.
st = env.reset(jax.random.PRNGKey(0))
info = dict(st.info)
qpos = base_env.set_actuator_joints_qpos(base_env._default_actuator, st.data.qpos)
qpos = base_env.set_floating_base_qpos(jp.array(base_env._init_q[:7]), qpos)
data = mjx_env.init(env.mjx_model, qpos=qpos, qvel=jp.zeros(env.mjx_model.nv), ctrl=base_env._default_actuator)
st = st.replace(data=data)
print("\n--- getup env, forced to HOME pose ---")
_, terms_forced = report("forced-home", st)
for k, v in sorted(terms_forced.items()):
    print("       %-32s % .6f" % (k, v))
