"""Check whether joint clamps or 50 Hz slew blocks the attempted balance action."""
import argparse
import json
from pathlib import Path
import numpy as np
from diagnostics.backward_phase_search import NativeRollout, body_pitch


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('trajectory',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--matched-end-s',type=float)
    a=p.parse_args()
    engine=NativeRollout('/data/shijinsheng/open_duck')
    t=np.load(a.trajectory,allow_pickle=False)
    times=t['time']
    names=[engine.sim.model.actuator(i).name for i in range(14)]
    qpos=t['qpos']
    pitch=np.asarray([body_pitch(q) for q in qpos])
    clamp=np.abs(t['joint_limit_target' if 'joint_limit_target' in t else 'joint_target']-t['unclipped_target'])
    slew=np.abs(t['motor_target']-t['joint_target'])
    force=t['actuator_force']
    model=engine.sim.model
    limits=np.maximum(np.abs(model.actuator_forcerange[:,0]),np.abs(model.actuator_forcerange[:,1]))
    windows={}
    masks=[('after_startup',times>=3),('last_1s',times>=times[-1]-1)]
    if a.matched_end_s is not None:
        masks.append(('matched_1s',(times>=a.matched_end_s-1)&(times<=a.matched_end_s+1e-7)))
    for label,keep in masks:
        if not np.any(keep):
            windows[label]=dict(samples=0)
            continue
        windows[label]={name:dict(joint_clamp_fraction=float(np.mean(clamp[keep,j]>1e-6)),maximum_joint_clamp_rad=float(clamp[keep,j].max()),slew_fraction=float(np.mean(slew[keep,j]>1e-6)),maximum_slew_gap_rad=float(slew[keep,j].max()),maximum_abs_actuator_force=float(np.max(abs(force[keep,j]))),force_limit=float(limits[j]) if model.actuator_forcelimited[j] else None,force_limit_fraction=float(np.mean(abs(force[keep,j])>=limits[j]-1e-5)) if model.actuator_forcelimited[j] else None) for j,name in enumerate(names)}
    samples=[dict(time_s=float(times[i]),pitch_deg=float(np.degrees(pitch[i])),contacts=t['feet_contacts'][i].tolist(),maximum_joint_clamp_rad=float(clamp[i].max()),maximum_slew_gap_rad=float(slew[i].max())) for i in range(max(0,len(times)-76),len(times),5)]
    result=dict(trajectory=str(a.trajectory),simulator_actuator_force_not_measured_motor_torque=True,windows=windows,last_1p5s=samples)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
