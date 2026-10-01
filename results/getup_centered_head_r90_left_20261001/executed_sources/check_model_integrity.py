"""Does the verified walk policy still balance in the collision-enabled model?

If the official walk ONNX cannot hold the home stance in scene_flat_terrain_getup
but does in scene_flat_terrain, the added collision boxes broke the physics and
every getup training run was doomed from the start.  If it balances in both, the
model is fine and the getup failures are a learning problem.
"""
import numpy as np
import mujoco

from playground.open_duck_mini_v2.mujoco_infer import MjInfer
from playground.common.onnx_infer import OnnxInfer

REPO = "/data/shijinsheng/open_duck/projects/Open_Duck_Playground"
WALK = "/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
REF = f"{REPO}/playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"

SCENES = {
    "original  ": f"{REPO}/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    "getup-coll": f"{REPO}/playground/open_duck_mini_v2/xmls/scene_flat_terrain_getup.xml",
}

for label, scene in SCENES.items():
    sim = MjInfer(scene, REF, WALK, standing=False)
    sim.policy = OnnxInfer(WALK, awd=True)
    m, d = sim.model, sim.data
    d.qpos[:] = m.keyframe("home").qpos
    d.qvel[:] = 0.0
    d.ctrl[:] = sim.default_actuator
    sim.motor_targets = np.array(sim.default_actuator, dtype=np.float64).copy()
    sim.prev_motor_targets = sim.motor_targets.copy()
    sim.commands = [0.0] * 7
    mujoco.mj_forward(m, d)
    imu = m.site("imu").id
    up = []
    for _ in range(500):  # 10 s at 50 Hz
        for _ in range(sim.decimation):
            mujoco.mj_step(m, d)
        sim.imitation_i = (sim.imitation_i + sim.phase_frequency_factor) % sim.PRM.nb_steps_in_period
        ang = sim.imitation_i / sim.PRM.nb_steps_in_period * 2 * np.pi
        sim.imitation_phase = np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
        obs = sim.get_obs(d, sim.commands)
        a = np.asarray(sim.policy.infer(obs), dtype=np.float32)
        sim.last_last_last_action = sim.last_action.copy()
        sim.last_last_action = sim.last_action.copy()
        sim.last_action = a.copy()
        tgt = sim.default_actuator + a * sim.action_scale
        mc = sim.max_motor_velocity * (sim.sim_dt * sim.decimation)
        sim.motor_targets = np.clip(tgt, sim.prev_motor_targets - mc, sim.prev_motor_targets + mc)
        sim.prev_motor_targets = sim.motor_targets.copy()
        d.ctrl[:] = sim.motor_targets
        up.append(float(d.site_xmat[imu].reshape(3, 3)[2, 2]))
    up = np.array(up)
    print("%s  final_up_z=%.4f  min_up_z=%.4f  final_height=%.4f  survived_10s=%s"
          % (label, up[-1], up.min(), float(d.qpos[2]), bool(up[-1] > 0.9)))
