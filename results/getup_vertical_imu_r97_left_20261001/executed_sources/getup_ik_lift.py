"""R9: test leg IK targets for lifting physically replayed, knee-supported states.

Desired root states exist ONLY in a separate kinematic scratch MjData. The
dynamic robot receives joint targets only, via unchanged motor limits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from diagnostics.audit_getup_load_support import LoadSupportSim
from diagnostics.getup_load_search import loaded_entry
from diagnostics.getup_anatomical_search import safe
from diagnostics.getup_beam_reference import primitive
from diagnostics.getup_independent_native import digest, DT, sustained_tail


def plan_target(sim, scratch, foot_z, home_rotations, height_step, shift_fraction, alignment):
    current = sim.data.qpos.copy()
    source = Rotation.from_quat(current[[4,5,6,3]])
    mat = source.as_matrix()
    yaw = np.arctan2(mat[1,0],mat[0,0])
    heading = Rotation.from_euler('z',yaw)
    destination = Slerp([0.,1.],Rotation.concatenate([source,heading]))([alignment])[0]
    quat = destination.as_quat()
    root = current[:3].copy()
    root[2] = min(root[2]+height_step,.17)
    feet = sim.data.xpos[sim.feet].copy()
    root[:2] += shift_fraction*(feet[:,:2].mean(axis=0)-root[:2])
    feet[:,2] = foot_z
    scratch.qpos[:] = current
    scratch.qpos[:3] = root
    scratch.qpos[3:7] = quat[[3,0,1,2]]
    leg = np.array([0,1,2,3,4,9,10,11,12,13])
    lower,upper = sim.lower[leg].copy(),sim.upper[leg].copy()
    lower[[3,8]] = 0.  # Normal-knee kinematic proposal; physical limits unchanged.
    desired_rot = [heading.as_matrix() @ r for r in home_rotations]
    def residual(q):
        scratch.qpos[sim.qadr[leg]] = q
        mujoco.mj_kinematics(sim.model,scratch)
        out=[]
        for i,b in enumerate(sim.feet):
            out.extend((scratch.xpos[b]-feet[i])*100.)
            out.extend(((scratch.xmat[b].reshape(3,3)-desired_rot[i])*.7).ravel())
        out.extend(.08*(q-sim.home[leg]))
        return np.asarray(out)
    start=np.clip(sim.data.qpos[sim.qadr[leg]],lower+1e-7,upper-1e-7)
    solution=least_squares(residual,start,bounds=(lower,upper),max_nfev=45,ftol=1e-5,xtol=1e-5,gtol=1e-5)
    target=sim.prev.copy()
    target[leg]=solution.x
    # Re-center the head slowly rather than use it as an external support.
    target[5:9]=.7*target[5:9]+.3*sim.home[5:9]
    return target,dict(cost=float(solution.cost),nfev=int(solution.nfev),
                        scratch_root_height_m=float(root[2]),scratch_only=True)


def run_case(job):
    root,experiment,variant,seed,record=job
    root,experiment=Path(root),Path(experiment)
    metadata=json.loads((experiment/'results.json').read_text())
    scene=Path(metadata['scene_path'])
    sim=LoadSupportSim(scene,root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    sim.prepare('standing')
    for _ in range(150):
        sim.step_target(sim.home+.25*sim.stand_action())
    foot_z=sim.data.xpos[sim.feet,2].copy()
    home_rotations=[sim.data.xmat[b].reshape(3,3).copy() for b in sim.feet]
    scratch=mujoco.MjData(sim.model)
    reference=np.load(experiment/'best_reference.npz',allow_pickle=False)
    sim.prepare(metadata['pose'],25000+seed,seed>0)
    initial=sim.measure()
    trace,metrics,plans=[],[],[]
    valid=True
    for target,duration in zip(reference['targets'],reference['durations_s']):
        r=primitive(sim,sim.snapshot(),target,float(duration),record)
        trace.extend(r['trace']); metrics.extend(r['measurements'])
        valid=valid and safe(r)
    entry=sim.measure()
    height_step,shift,alignment,duration=variant
    reached=False
    for _ in range(12):
        target,plan=plan_target(sim,scratch,foot_z,home_rotations,height_step,shift,alignment)
        plans.append(plan)
        r=primitive(sim,sim.snapshot(),target,duration,record)
        trace.extend([(t,q,v,c,'ik_lift') for t,q,v,c,phase in r['trace']])
        metrics.extend(r['measurements'])
        valid=valid and safe(r)
        if not valid:
            break
        if loaded_entry(r['metric']):
            reached=True
            break
    before_handoff=sim.measure()
    from_target=sim.prev.copy()
    flags=[]
    for i in range(250):
        a=min(i*DT,1.); a=a*a*(3.-2.*a)
        sim.step_target((1-a)*from_target+a*(sim.home+.25*sim.stand_action()))
        m=sim.measure(); metrics.append(m); flags.append(m['stable'])
        joints=sim.data.qpos[sim.qadr]
        overshoot=float(np.maximum(sim.lower-joints,joints-sim.upper).max())
        valid=valid and m['finite'] and m['self_penetration_m']<.004 and m['floor_penetration_m']<.01 and overshoot<.08
        if record:
            trace.append((float(sim.data.time),sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),'stand_handoff'))
    tail=sustained_tail(flags)
    joint_samples=np.asarray([m['joint_positions_rad'] for m in metrics])
    joint_overshoot=float(np.maximum(sim.lower-joint_samples,joint_samples-sim.upper).max())
    result=dict(seed=seed,variant=list(variant),initial=initial,prefix_end=entry,before_handoff=before_handoff,
                final=metrics[-1],entry_gate_reached=reached,valid=bool(valid),
                success=bool(initial['up_z']<.5 and initial['torso_contact'] and valid and tail>=2.-1e-8),
                sustained_final_stand_s=tail,plans=plans,
                maximum_floor_penetration_m=max(m['floor_penetration_m'] for m in metrics),
                maximum_self_penetration_m=max(m['self_penetration_m'] for m in metrics),
                maximum_joint_overshoot_rad=joint_overshoot,
                simulation_only=True,hardware_readiness=False,root_pose_edits_after_initialization=0)
    return result,trace


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    variants=[(.015,.05,.2,.4),(.03,.08,.3,.4),(.015,.15,.2,.7),(.03,.15,.4,.7)]
    jobs=[(str(args.root),str(args.experiment),v,s,s==0) for v in variants for s in range(3)]
    rows,selected=[],None
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result,trace in pool.map(run_case,jobs,chunksize=1):
            rows.append(result)
            print(json.dumps(dict(seed=result['seed'],variant=result['variant'],success=result['success'],
                                  valid=result['valid'],height=result['before_handoff']['height_m'],
                                  up=result['before_handoff']['up_z'],foot_load=result['before_handoff']['foot_load_fraction'])),flush=True)
            if trace and (selected is None or result['success']):
                selected=result
                np.savez_compressed(args.output/'best_trajectory.npz',time=[r[0] for r in trace],qpos=[r[1] for r in trace],
                                    qvel=[r[2] for r in trace],ctrl=[r[3] for r in trace],phase=[r[4] for r in trace])
    metadata=json.loads((args.experiment/'results.json').read_text())
    summary=dict(method='scratch-data leg IK proposals executed by native joint targets',pose=metadata['pose'],
                 scene_path=metadata['scene_path'],successful_validation_runs=sum(r['success'] for r in rows),
                 validation_runs=len(rows),results=rows,simulation_only=True,hardware_readiness=False,
                 root_pose_edits_after_initialization=0,
                 hashes={str(f):digest(f) for f in (Path(__file__),args.experiment/'best_reference.npz')})
    (args.output/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('IK COMPLETE:',summary['successful_validation_runs'],'/',len(rows),flush=True)


if __name__=='__main__':
    main()
