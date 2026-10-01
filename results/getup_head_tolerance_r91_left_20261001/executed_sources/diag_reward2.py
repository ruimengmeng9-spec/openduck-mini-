"""Per-term getup reward: fallen start vs forced-home pose (same env/model)."""
import os
os.environ.setdefault("GETUP_FALLEN_PROB", "0.0")
os.environ.setdefault("GETUP_LEAN_TILT", "0.35,0.70")
os.environ.setdefault("GETUP_RANDOM_JOINTS", "0")

import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env
from playground.open_duck_mini_v2 import focused_skill

S = focused_skill.focused_config("getup").reward_config.scales
env = focused_skill.FocusedSkill("getup", task="flat_terrain_getup")
zero = jp.zeros(env.action_size)


def show(tag, state):
    st = env.step(state, zero)
    gz = float(env.get_gravity(st.data)[-1])
    h = float(env.get_floating_base_qpos(st.data.qpos)[2])
    print("\n%s: up_z=%.3f height=%.4f  total_reward=%.3f" % (tag, gz, h, float(st.reward)))
    for k, v in sorted(st.metrics.items()):
        scale = S.get(k, None)
        contrib = (float(v) * scale) if scale is not None else None
        print("   %-26s raw=% .6f scale=%-9s -> % .4f" % (
            k, float(v), scale, contrib if contrib is not None else 0.0))


# Fallen / leaning start produced by the getup reset.
st = env.reset(jax.random.PRNGKey(0))
print("start up_z=%.3f height=%.4f" % (
    float(env.get_gravity(st.data)[-1]),
    float(env.get_floating_base_qpos(st.data.qpos)[2])))
show("GETUP reset pose", st)

# Same env, but forced into the home standing keyframe.
st2 = env.reset(jax.random.PRNGKey(0))
qpos = env.set_actuator_joints_qpos(env._default_actuator, st2.data.qpos)
qpos = env.set_floating_base_qpos(jp.array(env._init_q[:7]), qpos)
data = mjx_env.init(env.mjx_model, qpos=qpos, qvel=jp.zeros(env.mjx_model.nv), ctrl=env._default_actuator)
show("FORCED home pose", st2.replace(data=data))
