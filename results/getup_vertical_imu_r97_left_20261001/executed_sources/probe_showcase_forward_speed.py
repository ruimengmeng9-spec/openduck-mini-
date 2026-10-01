"""Read-only controller A/B: same warm state, policy and physical limits."""
import json
import math
from pathlib import Path
import mujoco
import numpy as np
from diagnostics.backward_phase_search import NativeRollout, CpuActor, heading
from diagnostics.reference_residual_policy import ReferenceResidualPolicy

root=Path('/data/shijinsheng/open_duck')
engine=NativeRollout(root)
s=engine.sim
actor=CpuActor(root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
decoder=ReferenceResidualPolicy(engine.actor,s,.12,1.)
mujoco.mj_resetData(s.model,s.data)
engine.robot.reset()
s.data.qvel[:]=np.random.default_rng(1600).uniform(-.02,.02,s.model.nv)
s.imitation_phase=np.array([1.,0.],dtype=np.float32)
mujoco.mj_forward(s.model,s.data)
filtered=s.prev_motor_targets.copy()

def step(vx,tau,history):
    global filtered
    obs=np.asarray(s.get_obs(s.data,[vx,0.,0.,0.,0.,0.,0.]),dtype=np.float32)
    action=actor.infer(obs)
    target=np.clip(s.default_actuator+s.action_scale*action,decoder.lower,decoder.upper)
    filtered+=engine.dt/(tau+engine.dt)*(target-filtered)
    slew=np.clip(filtered,s.prev_motor_targets-s.max_motor_velocity*engine.dt,s.prev_motor_targets+s.max_motor_velocity*engine.dt)
    s.data.ctrl[:]=slew
    s.motor_targets=slew.copy()
    s.prev_motor_targets=slew.copy()
    s.last_last_last_action=s.last_last_action.copy()
    s.last_last_action=s.last_action.copy()
    s.last_action=((filtered-s.default_actuator)/s.action_scale).astype(np.float32) if history=='filtered' else action.copy()
    for _ in range(s.decimation):
        mujoco.mj_step(s.model,s.data)
    s.imitation_i=(s.imitation_i+1)%s.PRM.nb_steps_in_period
    phi=s.imitation_i/s.PRM.nb_steps_in_period*2*np.pi
    s.imitation_phase=np.array([np.cos(phi),np.sin(phi)],dtype=np.float32)

for _ in range(150):
    step(0.,.01,'filtered')
state_spec=mujoco.mjtState.mjSTATE_INTEGRATION
state=np.empty(mujoco.mj_stateSize(s.model,state_spec))
mujoco.mj_getState(s.model,s.data,state,state_spec)
names=['motor_targets','prev_motor_targets','last_action','last_last_action','last_last_last_action','imitation_phase']
history_state={name:getattr(s,name).copy() for name in names}
phase_i=s.imitation_i
filtered0=filtered.copy()
rows=[]
for vx in [.1,.15]:
    for tau,history in [(.01,'filtered'),(.01,'raw'),(0.,'raw')]:
        mujoco.mj_setState(s.model,s.data,state,state_spec)
        for name,value in history_state.items():
            setattr(s,name,value.copy())
        s.imitation_i=phase_i
        filtered=filtered0.copy()
        mujoco.mj_forward(s.model,s.data)
        q0=s.get_floating_base_qpos(s.data.qpos).copy()
        yaw0=heading(q0)
        velocities=[]
        positions=[q0[:2].copy()]
        min_up=1.
        for i in range(500):
            step(vx,tau,history)
            q=s.get_floating_base_qpos(s.data.qpos)
            min_up=min(min_up,1.-2.*(q[4]**2+q[5]**2))
            velocities.append(engine.robot._sensor('local_linvel').copy())
            positions.append(q[:2].copy())
        positions=np.array(positions)
        forward=np.array([math.cos(yaw0),math.sin(yaw0)])
        row=dict(command_mps=vx,target_tau_s=tau,history=history,
                 average_axial_first5s_mps=float((positions[250]-positions[0])@forward/5.),
                 mean_local_forward_last2s_mps=float(np.mean(np.array(velocities)[-100:,0])),
                 minimum_up_z=float(min_up),fallen=bool(min_up<.5),
                 heading_change_deg=math.degrees(heading(s.get_floating_base_qpos(s.data.qpos))-yaw0))
        rows.append(row)
        print(json.dumps(row),flush=True)
out=root/'outputs/showcase_20260928_r22/forward_speed_ab.json'
with out.open('x') as f:
    json.dump(dict(shared_start=True,changed_physics=False,rows=rows),f,indent=2)
