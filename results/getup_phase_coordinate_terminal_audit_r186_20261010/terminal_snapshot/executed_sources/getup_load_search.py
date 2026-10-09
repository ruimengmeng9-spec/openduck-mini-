"""R8: loaded-support-aware recovery search, preserving all physical limits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from diagnostics.audit_getup_load_support import LoadSupportSim
from diagnostics.getup_aligned_extension import strict_entry
from diagnostics.getup_anatomical_search import anatomical_library, safe
from diagnostics.getup_beam_reference import primitive
from diagnostics import getup_dynamic_beam as dynamic
from diagnostics.getup_independent_native import digest


def loaded_entry(m):
    return (strict_entry(m) and min(m['foot_up_alignment_to_home']) > .85
            and min(m['foot_normal_forces_n']) > .1 and m['foot_load_fraction'] > .9)


def quality(m):
    u=float(np.clip((m['up_z']-.35)/.6,0.,1.))
    lift=float(np.clip((m['height_m']-.04)/.12,0.,1.2))
    flat=float(np.mean(m['foot_up_alignment_to_home']))
    home=float(m['joint_home_error_mean_rad'])
    return (2.*m['up_z']+u*(5.*lift+flat+1.5*m['foot_load_fraction']-.5*home)
            -.25*m['torso_contact'])


_sim=None


def init_worker(scene,stand):
    global _sim
    _sim=LoadSupportSim(scene,stand)


def evaluate(job):
    state,target,duration=job
    r=primitive(_sim,state,target,duration)
    tail=r['measurements'][-min(8,len(r['measurements'])):]
    r['score']=float(np.mean([quality(m) for m in tail])+10.*loaded_entry(r['metric'])
                     -100.*r['maximum_penetration_m'])
    return r


def targets_for(sim,parent,rng):
    # Retain normal-knee families, but allow recovery intermediates to explore
    # the original full limits. Loaded, aligned standing remains the goal.
    normal=anatomical_library(sim,parent['state']['prev'],rng)
    local=np.clip(rng.normal(parent['state']['prev'],.55,(12,sim.model.nu)),sim.lower,sim.upper)
    global_targets=rng.uniform(sim.lower,sim.upper,(12,sim.model.nu))
    library=np.concatenate([normal,local,global_targets])
    library[:,[7,8]]=0.
    return library


def choose_beam(candidates,width):
    # Keep a home-alignment lane as well as height/support lanes to avoid
    # pruning every temporary regression needed to unfold the legs.
    by_score=sorted(candidates,key=lambda r:-r['score'])
    by_home=sorted(candidates,key=lambda r:-(r['score']-1.2*r['metric']['joint_home_error_mean_rad']))
    by_load=sorted(candidates,key=lambda r:-(r['score']+r['metric']['foot_load_fraction']))
    beam,bins=[],set()
    for i in range(len(candidates)):
        for lane in (by_score,by_home,by_load):
            c=lane[i]
            q=c['state']['qpos'][3:7]
            key=(*np.round(q/.13).astype(int),round(c['metric']['height_m']/.015),
                 *np.round(c['state']['prev'][[2,3,4,12]]/.35).astype(int))
            if key not in bins:
                bins.add(key)
                beam.append(c)
            if len(beam)==width:
                return beam
    return beam


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--pose',choices=('prone','supine'),default='supine')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--depth',type=int,default=10)
    p.add_argument('--width',type=int,default=6)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=227)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    scene=args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand=args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    qual=json.loads((scene.parent/'qualification.json').read_text())
    if qual['standing_passed_runs']!=qual['standing_runs']:
        raise RuntimeError('model standing qualification failed')
    sim=LoadSupportSim(scene,stand)
    state=sim.prepare(args.pose)
    beam=[dict(state=state,path=[],score=0.,metric=sim.measure())]
    best=beam[0]
    rng,history,started=np.random.default_rng(args.seed),[],time.time()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                             initargs=(str(scene),str(stand))) as pool:
        for depth in range(args.depth):
            jobs,paths=[],[]
            for parent in beam:
                for target in targets_for(sim,parent,rng):
                    for duration in (.36,.90):
                        jobs.append((parent['state'],target,duration))
                        paths.append(parent['path']+[(target.copy(),duration)])
            candidates=[]
            for i,r in enumerate(pool.map(evaluate,jobs,chunksize=4)):
                if safe(r):
                    r['path']=paths[i]
                    candidates.append(r)
            if not candidates:
                raise RuntimeError('no safe load-aware nodes')
            beam=choose_beam(candidates,args.width)
            best=max(beam,key=lambda r:r['score'])
            np.savez_compressed(args.output/'best_reference.npz',targets=[t for t,d in best['path']],
                                durations_s=[d for t,d in best['path']])
            row=dict(depth=depth+1,expanded=len(jobs),safe_nodes=len(candidates),best_score=best['score'],
                     metric=best['metric'],ready_nodes=sum(loaded_entry(r['metric']) for r in candidates),
                     wall_seconds=time.time()-started)
            history.append(row)
            (args.output/'search_progress.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            brief={k:v for k,v in row.items() if k!='metric'}
            brief['metric']={k:best['metric'][k] for k in ('height_m','up_z','foot_load_fraction','foot_up_alignment_to_home','joint_home_error_max_rad')}
            print(json.dumps(brief),flush=True)
            if loaded_entry(best['metric']):
                break
    dynamic.ready_to_handoff=loaded_entry
    rows=[]
    for seed in range(20):
        result,trace=dynamic.replay(sim,args.pose,best['path'],23000+seed,seed>0,seed==0)
        rows.append(result)
        print('VALIDATING',seed,result['success'],flush=True)
        if trace:
            np.savez_compressed(args.output/'best_trajectory.npz',time=[r[0] for r in trace],qpos=[r[1] for r in trace],
                                qvel=[r[2] for r in trace],ctrl=[r[3] for r in trace],phase=[r[4] for r in trace])
    summary=dict(method='load-aware, foot-orientation and home-alignment search lanes',pose=args.pose,
                 successful_validation_runs=sum(r['success'] for r in rows),validation_runs=len(rows),results=rows,
                 simulation_only=True,hardware_readiness=False,root_pose_edits_after_initialization=0,
                 scene_path=str(scene),seed=args.seed,depth=args.depth,width=args.width,
                 hashes={str(f):digest(f) for f in (Path(__file__),Path(__file__).with_name('audit_getup_load_support.py'),scene,stand)})
    (args.output/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('COMPLETE:',summary['successful_validation_runs'],'/20',flush=True)


if __name__=='__main__':
    main()
