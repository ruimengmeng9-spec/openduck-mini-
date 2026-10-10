"""R5: extend a physically replayed recovery prefix into supported standing.

The prefix is replayed from the fallen start, never replaced by a teleported
crouch pose. Stand-height shaping changes search only, not success criteria.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from diagnostics.getup_dynamic_beam import SelfCollisionSim, replay
from diagnostics.getup_beam_reference import primitive
from diagnostics.getup_independent_native import digest
from diagnostics.getup_feedback_reference import ready_to_handoff


_sim=None


def init_worker(scene,stand):
    global _sim
    _sim=SelfCollisionSim(scene,stand)


def evaluate(job):
    state,target,duration=job
    result=primitive(_sim,state,target,duration)
    quality=[]
    for m in result['measurements'][-min(8,len(result['measurements'])):]:
        upright_weight=float(np.clip((m['up_z']-.45)/.5,0.,1.))
        lift=float(np.clip((m['height_m']-.04)/.12,0.,1.3))
        quality.append(2.*m['up_z']+4.*upright_weight*lift+.3*sum(m['feet'])-.4*m['torso_contact'])
    result['score']=float(np.mean(quality)+5.*ready_to_handoff(result['metric'])-100.*result['maximum_penetration_m'])
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--depth',type=int,default=12)
    p.add_argument('--width',type=int,default=6)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=162)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    prefix_result=json.loads((args.prefix.parent/'results.json').read_text())
    pose=prefix_result['pose']
    reference=np.load(args.prefix,allow_pickle=False)
    prefix=list(zip(reference['targets'],reference['durations_s']))
    stand=args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim=SelfCollisionSim(args.scene,stand)
    qualification=json.loads((args.scene.parent/'qualification.json').read_text())
    if qualification['standing_passed_runs']!=qualification['standing_runs']:
        raise RuntimeError('normal-standing qualification failed')
    sim.prepare(pose)
    for target,duration in prefix:
        result=primitive(sim,sim.snapshot(),target,float(duration))
        if (max(m['self_penetration_m'] for m in result['measurements'])>.004
                or result['maximum_joint_overshoot_rad']>.08):
            raise RuntimeError('prefix safety gate failed')
    initial=sim.snapshot()
    print('REPLAYED PREFIX:',json.dumps(sim.measure()),flush=True)
    beam=[dict(state=initial,path=prefix,metric=sim.measure(),score=0.)]
    rng,history,started=np.random.default_rng(args.seed),[],time.time()
    best=beam[0]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                             initargs=(str(args.scene),str(stand))) as pool:
        for depth in range(args.depth):
            jobs,paths=[],[]
            for parent in beam:
                posture=[]
                for hip in (-1.0,-.63,-.2):
                    for knee in (-.6,0.,.6,1.37):
                        target=sim.home.copy()
                        target[[2,11]]=[hip,-hip]
                        target[[3,12]]=knee
                        posture.append(target)
                local=np.clip(rng.normal(parent['state']['prev'],.4,(24,sim.model.nu)),sim.lower,sim.upper)
                standing=np.clip(rng.normal(sim.home,.22,(16,sim.model.nu)),sim.lower,sim.upper)
                local[:,[7,8]]=0.
                standing[:,[7,8]]=0.
                library=np.concatenate([posture,local,standing,sim.home[None],parent['state']['prev'][None]])
                for target in library:
                    for duration in (.54,1.08):
                        jobs.append((parent['state'],target,duration))
                        paths.append(parent['path']+[(target.copy(),duration)])
            evaluated=list(pool.map(evaluate,jobs,chunksize=4))
            candidates=[]
            for i,result in enumerate(evaluated):
                ms=result['measurements']
                if (not all(m['finite'] for m in ms) or result['maximum_joint_overshoot_rad']>.08
                        or max(m['self_penetration_m'] for m in ms)>.004
                        or max(m['floor_penetration_m'] for m in ms)>.01):
                    continue
                result['path']=paths[i]
                candidates.append(result)
            candidates.sort(key=lambda r:-r['score'])
            if not candidates:
                raise RuntimeError('no valid crouch-extension nodes')
            bins,beam=set(),[]
            for candidate in candidates:
                q=candidate['state']['qpos'][3:7]
                key=(*np.round(q/.13).astype(int),round(candidate['metric']['height_m']/.015),
                     *np.round(candidate['state']['prev'][[2,3,4,5]]/.35).astype(int))
                if key not in bins:
                    bins.add(key)
                    beam.append(candidate)
                if len(beam)==args.width:
                    break
            best=beam[0]
            np.savez_compressed(args.output/'best_reference.npz',targets=[t for t,d in best['path']],durations_s=[d for t,d in best['path']])
            row=dict(depth=depth+1,expanded=len(jobs),valid_nodes=len(candidates),best_score=best['score'],
                     best_metric=best['metric'],ready_nodes=sum(ready_to_handoff(r['metric']) for r in candidates),
                     wall_seconds=time.time()-started)
            history.append(row)
            (args.output/'search_progress.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
            if ready_to_handoff(best['metric']):
                break
    results=[]
    for i in range(20):
        result,trace=replay(sim,pose,best['path'],18000+i,i>0,i==0)
        results.append(result)
        if trace:
            np.savez_compressed(args.output/'best_trajectory.npz',time=[r[0] for r in trace],qpos=[r[1] for r in trace],
                                qvel=[r[2] for r in trace],ctrl=[r[3] for r in trace],phase=[r[4] for r in trace])
    summary=dict(pose=pose,method='replayed-prefix crouch-to-stand extension with height shaping',
                 successful_validation_runs=sum(r['success'] for r in results),validation_runs=len(results),results=results,
                 simulation_only=True,hardware_readiness=False,root_pose_edits_after_initialization=0,
                 scene_path=str(args.scene),seed=args.seed,depth=args.depth,width=args.width,
                 hashes={str(f):digest(f) for f in (Path(__file__),args.scene,args.prefix,stand)})
    (args.output/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('VALIDATION:',summary['successful_validation_runs'],'/20',flush=True)


if __name__=='__main__':
    main()
