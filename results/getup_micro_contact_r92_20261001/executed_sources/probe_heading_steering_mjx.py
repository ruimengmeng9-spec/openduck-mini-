"""Match native home/velocity initialization and inspect the new MJX clock."""
import json
import math
from pathlib import Path
import jax
import jax.numpy as jp
import mujoco
import mujoco.mjx as mjx
import numpy as np
from mujoco_playground._src.collision import geoms_colliding
from open_duck_agent.duck_sim import DuckSimulation
from playground.open_duck_mini_v2.heading_steering import HeadingSteering
from playground.open_duck_mini_v2.focused_skill import focused_config

root = Path('/data/shijinsheng/open_duck')
cfg = focused_config('backward')
cfg.lin_vel_x = [-.074,-.074]
env = HeadingSteering(baseline_path=str(root/'training/backward_reference_residual_r2/final.onnx'),residual_gain=.12,ramp_s=1.,initial_error=0.,task='flat_terrain',config=cfg)
robot = DuckSimulation(root/'projects/Open_Duck_Playground',root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',output_root=root/'outputs/steering_mjx_probe/traces',warmup_s=0.)
sim = robot.sim
sim.imitation_i = 0.
sim.imitation_phase = np.array([1.,0.],dtype=np.float32)
sim.data.qvel[:] += np.random.default_rng(0).uniform(-.02,.02,sim.model.nv)
mujoco.mj_forward(sim.model,sim.data)
native_obs = np.asarray(sim.get_obs(sim.data,[-.074,0,0,0,0,0,0]),dtype=np.float32)
state = jax.jit(env.reset)(jax.random.PRNGKey(0))
info = dict(state.info)
info['initial_yaw'] = jp.asarray(0.)
info['command'] = jp.array([-.074,0.,0.,0.,0.,0.,0.])
data = mjx.forward(env.mjx_model,state.data.replace(qpos=jp.asarray(sim.data.qpos),qvel=jp.asarray(sim.data.qvel),time=jp.asarray(0.),ctrl=env._default_actuator))
contact = jp.array([geoms_colliding(data,g,env._floor_geom_id) for g in env._feet_geom_id])
state = env.augment(state.replace(data=data,info=info,obs=env._get_obs(data,info,contact)))
diff = np.asarray(state.obs['state'][:101])-native_obs
print('INITIAL OBS MAX DIFFERENCE:',float(abs(diff).max()),'INDICES:',np.where(abs(diff)>.001)[0].tolist(),flush=True)
step = jax.jit(env.step)
positions=[]
headings=[]
for k in range(1500):
    state = step(state,jp.zeros(2))
    q = np.asarray(env.get_floating_base_qpos(state.data.qpos))
    positions.append(q[:3])
    w,x,y,z=q[3:7]
    headings.append(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    if (k+1)%500 == 0:
        print('TIME:',round((k+1)*env.dt,1),'YAW:',round(math.degrees(np.unwrap(headings)[-1]),3),'DONE:',float(state.done),flush=True)
    if bool(state.done):
        break
out = root/'outputs/steering_mjx_probe/results.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(dict(seed=0,zero_correction=True,matched_native_initialization=True,steps=len(headings),yaw_change_deg=math.degrees(np.unwrap(headings)[-1]),displacement_xyz_m=(np.asarray(positions[-1])-sim.data.qpos[:3]).tolist(),initial_obs_max_error=float(abs(diff).max()),initial_obs_error=diff.tolist()),indent=2))
print('SAVED',out,flush=True)
