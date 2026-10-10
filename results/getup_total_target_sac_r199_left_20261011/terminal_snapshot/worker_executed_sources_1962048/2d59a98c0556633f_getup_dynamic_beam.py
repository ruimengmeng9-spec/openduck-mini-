"""R4 state-diverse, variable-duration search on decomposed self-collision CAD.

References remain simulation candidates. No pretrained locomotion model is
replaced and no external forces or runtime root-state edits are used.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time

import mujoco
import numpy as np

from diagnostics.getup_independent_native import RecoverySim, feature_targets, digest, POSES
from diagnostics.getup_beam_reference import primitive
from diagnostics.getup_feedback_reference import ready_to_handoff
from diagnostics.getup_independent_native import DT, sustained_tail


class SelfCollisionSim(RecoverySim):
    def measure(self):
        result = super().measure()
        self_penetration = 0.
        for c in self.data.contact:
            if self.floor not in c.geom:
                self_penetration = max(self_penetration, float(-c.dist))
        result['floor_penetration_m'] = result['penetration_m']
        result['self_penetration_m'] = self_penetration
        result['penetration_m'] = max(result['penetration_m'], self_penetration)
        result['stable'] = result['stable'] and self_penetration < .002
        return result


_sim = None


def init_worker(scene, stand):
    global _sim
    _sim = SelfCollisionSim(scene, stand)


def evaluate(job):
    state, target, duration = job
    result = primitive(_sim, state, target, duration)
    ms = result['measurements']
    # Do not punish angular motion as if recovery were a static standing task.
    score = np.mean([2.7*m['up_z']+min(m['height_m']/.17,1.2)
                     +.3*sum(m['feet'])-.25*m['torso_contact'] for m in ms[-min(8,len(ms)):]])
    result['score'] = float(score+3.*ready_to_handoff(result['metric'])
                            -80.*result['maximum_penetration_m'])
    return result


def replay(sim, pose, path, seed=0, perturb=False, record=False):
    initial = sim.prepare(pose,seed,perturb)
    init = sim.measure()
    trace, measurements, max_force, max_slew, max_joint = [], [], 0., 0., 0.
    for target, duration in path:
        result = primitive(sim,sim.snapshot(),target,duration,record)
        trace.extend(result['trace'])
        measurements.extend(result['measurements'])
        max_force = max(max_force,result['max_sampled_actuator_force_nm'])
        max_slew = max(max_slew,result['max_command_slew_rad_s'])
        max_joint = max(max_joint,result['maximum_joint_overshoot_rad'])
        if ready_to_handoff(result['metric']):
            break
    entry = sim.prev.copy()
    flags=[]
    previous=sim.prev.copy()
    for i in range(250):
        a=min(i*DT,1.)
        a=a*a*(3.-2.*a)
        target=(1-a)*entry+a*(sim.home+.25*sim.stand_action())
        applied=sim.step_target(target)
        max_slew=max(max_slew,float(np.abs(applied-previous).max()/DT))
        previous=applied.copy()
        max_force=max(max_force,float(np.abs(sim.data.actuator_force).max()))
        joints=sim.data.qpos[sim.qadr]
        max_joint=max(max_joint,float(np.maximum(sim.lower-joints,joints-sim.upper).max()))
        m=sim.measure()
        measurements.append(m)
        flags.append(m['stable'])
        if record:
            trace.append((float(sim.data.time),sim.data.qpos.copy(),sim.data.qvel.copy(),applied.copy(),'stand_handoff'))
    stable=sustained_tail(flags)
    max_floor=max(m['floor_penetration_m'] for m in measurements)
    max_self=max(m['self_penetration_m'] for m in measurements)
    valid=(all(m['finite'] for m in measurements) and max_floor<.01 and max_self<.004
           and max_joint<.08 and max_force<=3.23+1e-8 and max_slew<=5.24+1e-8)
    success=init['up_z']<.5 and init['torso_contact'] and valid and stable>=2.-1e-8
    return dict(success=bool(success),seed=seed,initial=init,final=measurements[-1],
                sustained_final_stand_s=stable,maximum_up_z=max(m['up_z'] for m in measurements),
                maximum_floor_penetration_m=max_floor,maximum_self_penetration_m=max_self,
                maximum_joint_overshoot_rad=max_joint,max_sampled_actuator_force_nm=max_force,
                max_command_slew_rad_s=max_slew,simulation_only=True,hardware_readiness=False),trace


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pose',choices=POSES,default='supine')
    p.add_argument('--depth',type=int,default=12)
    p.add_argument('--width',type=int,default=6)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=151)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    audit=json.loads((args.scene.parent/'audit.json').read_text())
    qualification=json.loads((args.scene.parent/'qualification.json').read_text())
    if qualification['standing_passed_runs'] != qualification['standing_runs']:
        raise RuntimeError('derived collision model failed normal-standing precheck')
    stand=args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim=SelfCollisionSim(args.scene,stand)
    rows=json.loads((args.workspace/'results.json').read_text())['rows']
    workspace_targets=feature_targets([r['features_rad'] for r in rows],sim.home,sim.lower,sim.upper)
    initial=sim.prepare(args.pose)
    beam=[dict(state=initial,path=[],score=0.,metric=sim.measure())]
    rng=np.random.default_rng(args.seed)
    history,started,best=[],time.time(),beam[0]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                             initargs=(str(args.scene),str(stand))) as pool:
        for depth in range(args.depth):
            jobs,paths=[],[]
            for parent in beam:
                global_random=rng.uniform(sim.lower,sim.upper,(20,sim.model.nu))
                local_random=np.clip(rng.normal(parent['state']['prev'],.45,(20,sim.model.nu)),sim.lower,sim.upper)
                global_random[:,[7,8]]=0.
                local_random[:,[7,8]]=0.
                library=np.concatenate([workspace_targets,global_random,local_random,
                                        sim.home[None],parent['state']['prev'][None]])
                for target in library:
                    for duration in (.24,.54):
                        jobs.append((parent['state'],target,duration))
                        paths.append(parent['path']+[(target.copy(),duration)])
            results=list(pool.map(evaluate,jobs,chunksize=4))
            candidates=[]
            for i,result in enumerate(results):
                if (not all(m['finite'] for m in result['measurements'])
                        or result['maximum_joint_overshoot_rad']>.08
                        or max(m['floor_penetration_m'] for m in result['measurements'])>.01
                        or max(m['self_penetration_m'] for m in result['measurements'])>.004):
                    continue
                result['path']=paths[i]
                candidates.append(result)
            candidates.sort(key=lambda r:-r['score'])
            if not candidates:
                raise RuntimeError('no contact-valid nodes remain')
            beam,bins=[],set()
            for candidate in candidates:
                q=candidate['state']['qpos'][3:7]
                key=(*np.round(q/.13).astype(int),round(candidate['metric']['height_m']/.02),
                     *np.round(candidate['state']['prev'][[2,3,4,5]]/.45).astype(int))
                if key not in bins:
                    bins.add(key)
                    beam.append(candidate)
                if len(beam)==args.width:
                    break
            best=beam[0]
            np.savez_compressed(args.output/'best_reference.npz',targets=[t for t,d in best['path']],
                                durations_s=[d for t,d in best['path']])
            row=dict(depth=depth+1,expanded=len(jobs),contact_valid_nodes=len(candidates),best_score=best['score'],
                     best_metric=best['metric'],ready_nodes=sum(ready_to_handoff(r['metric']) for r in candidates),
                     wall_seconds=time.time()-started)
            history.append(row)
            (args.output/'search_progress.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
            if ready_to_handoff(best['metric']):
                break
    results=[]
    for seed in range(20):
        result,trace=replay(sim,args.pose,best['path'],16000+seed,seed>0,seed==0)
        results.append(result)
        if trace:
            np.savez_compressed(args.output/'best_trajectory.npz',time=[r[0] for r in trace],
                                qpos=[r[1] for r in trace],qvel=[r[2] for r in trace],
                                ctrl=[r[3] for r in trace],phase=[r[4] for r in trace])
    summary=dict(pose=args.pose,method='variable-duration beam search on decomposed self-collision model',
                 successful_validation_runs=sum(r['success'] for r in results),validation_runs=len(results),
                 results=results,simulation_only=True,hardware_readiness=False,collision_approximation_validated=False,
                 root_state_edits_after_initialization=0,parameters_preserved=audit['unchanged_parameters'],
                 scene_path=str(args.scene),seed=args.seed,depth=args.depth,width=args.width,
                 hashes={str(f):digest(f) for f in (Path(__file__),args.scene,args.scene.parent/'robot_decomposed.xml',stand)})
    (args.output/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('VALIDATION:',summary['successful_validation_runs'],'/20',flush=True)


if __name__=='__main__':
    main()
