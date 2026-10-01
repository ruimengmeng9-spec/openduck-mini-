"""Test verified near-standing command sequences AFTER actual fallen prefixes.

No initial-state transplant and no assumption that a near-standing teacher
generalizes to a fallen start. Each composed path is replayed from the ground.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np

from diagnostics.getup_independent_native import DT, POSES, digest
from diagnostics.search_getup_contact_archive_r30 import prefix_trial
from diagnostics.search_getup_rescue_r22 import target_at
from diagnostics.train_getup_fullpath_r27 import rollout, save_trace
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def teacher_targets(sim, knots):
    # Explicit 50Hz targets, NOT a mean of incompatible teacher trajectories.
    q=np.asarray([target_at(sim,knots,index*DT) for index in range(60)])
    return np.concatenate([q,sim.home[None]]),np.r_[np.full(60,DT),.6]


def bridge_priority(row):
    m=row['metric']
    return float(m['up_z']+min(m['height_m']/.165,1.)*.5
                 +m['foot_load_fraction']*.25-m['joint_home_error_mean_rad']*.15)


_sim=None
def init_worker(scene,stand):
    global _sim
    _sim=StrictSim(scene,stand)


def endpoint(job):
    return prefix_trial(_sim,*job)


def evaluate(job):
    pose,q,d=job
    return rollout(_sim,pose,q,d,hold_s=4.)[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--pose',choices=POSES,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prefixes',type=int,default=24)
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    if not (args.archive/'results.json').exists():
        raise RuntimeError('Only inspect a terminal archive, not a live partially written checkpoint')
    args.output.mkdir(parents=True,exist_ok=False)
    scene=args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand=args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    library_file=args.root/'training/getup_transfer_r25/trajectory_library.npz'
    sim=StrictSim(scene,stand)
    with np.load(library_file,allow_pickle=False) as z:
        teachers=z['knots'][z['rescue']].copy()
        teacher_seeds=z['seed'][z['rescue']].copy().tolist()
    # Include the unchanged home baseline as a paired continuation control.
    teachers=np.concatenate([np.zeros((1,2,8)),teachers])
    teacher_seeds=[None,*teacher_seeds]
    source=args.output/'executed_sources';source.mkdir()
    for name in ('bridge_getup_contact_prefixes_r31.py','search_getup_contact_archive_r30.py',
                 'train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py','search_getup_rescue_r22.py'):
        shutil.copy2(Path(__file__).with_name(name),source/name)
    contract=dict(simulation_only=True,hardware_readiness=False,default_controller_replaced=False,
        method='full fallen prefixes followed by separate previously verified near-standing sequences',
        pose=args.pose,root_edits_after_initialization=0,mechanics_and_completion_gate_unchanged=True,
        worker_start_method='spawn',teacher_seeds=teacher_seeds,
        hashes={str(f):digest(f) for f in (Path(__file__),scene,stand,library_file,args.archive/'archive_paths.npz')})
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    with np.load(args.archive/'archive_paths.npz',allow_pickle=False) as z:
        count=sum(name.startswith('targets_') for name in z.files)
        paths=[(z[f'targets_{i}'].copy(),z[f'durations_{i}'].copy()) for i in range(count)]
    started=time.monotonic();trials=[];best=None
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),
                             initializer=init_worker,initargs=(str(scene),str(stand))) as pool:
        endpoints=list(pool.map(endpoint,[(args.pose,q,d) for q,d in paths],chunksize=1))
        (args.output/'endpoints.json').write_text(json.dumps(endpoints,indent=2))
        eligible=[i for i,r in enumerate(endpoints) if r['valid'] and r['metric']['up_z']>.55
                  and r['metric']['height_m']>.08]
        selected=sorted(eligible,key=lambda i:-bridge_priority(endpoints[i]))[:args.prefixes]
        scan=dict(total_prefixes=count,physically_valid=sum(r['valid'] for r in endpoints),
                  eligible_prefixes=len(eligible),selected_prefixes=selected,
                  maximum_up=max(r['metric']['up_z'] for r in endpoints if r['valid']),
                  control_steps=sum(r['control_steps'] for r in endpoints))
        print('PREFIX_SCAN',json.dumps(scan),flush=True)
        for i in selected:
            prefix_q,prefix_d=paths[i]
            combinations=[]
            for teacher in teachers:
                q,d=teacher_targets(sim,teacher)
                combinations.append((np.concatenate([prefix_q,q]),np.r_[prefix_d,d]))
            rows=list(pool.map(evaluate,[(args.pose,q,d) for q,d in combinations],chunksize=1))
            # All comparisons start from the identical actual fallen state.
            if len({r['initial_hash'] for r in rows})!=1:
                raise RuntimeError('Paired fallen starts do not match')
            for j,(r,(q,d)) in enumerate(zip(rows,combinations)):
                trials.append(dict(prefix_index=i,teacher_seed=teacher_seeds[j],result=r))
                if best is None or r['score']>best[0]['score']:
                    best=r,q,d
                    np.savez_compressed(args.output/'best_reference.npz',targets=q,durations_s=d)
            print('BRIDGE_PREFIX',i,'successes',sum(r['success'] for r in rows),'/',len(rows),flush=True)
            (args.output/'progress.json').write_text(json.dumps(trials,indent=2))
    validation=None
    if best:
        r,q,d=best
        nominal,trace=rollout(sim,args.pose,q,d,hold_s=31.,record=True)
        save_trace(args.output/'nominal_trajectory.npz',trace)
        n,hold=(20,31.) if nominal['success'] else (3,4.)
        heldout=[rollout(sim,args.pose,q,d,1100000+i,True,hold)[0] for i in range(n)]
        validation=dict(nominal=nominal,trials=heldout,trial_count=n,hold_s=hold,
                        successes=sum(r['success'] for r in heldout))
    summary=dict(contract=contract,scan=scan,trials=trials,validation=validation,
                  search_control_steps=scan['control_steps']+sum(t['result']['control_steps'] for t in trials),
                  full_task_completed=False,elapsed_seconds=time.monotonic()-started)
    (args.output/'results.json').write_text(json.dumps(summary,indent=2))
    print('BRIDGE_TERMINAL',args.pose,'nominal_success',bool(validation and validation['nominal']['success']),flush=True)


if __name__=='__main__':
    main()
