"""Contact/orientation-diverse prefix search from real fallen starts.

Unlike a small score-ranked beam, retain intermediate low-quality postures in
distinct contact/orientation cells. Each proposal replays its ENTIRE prefix from
the canonical fallen initialization: no snapshot transplant, root edits, added
forces or changes to the mechanics. Search progress is not recovery success.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from diagnostics.getup_independent_native import DT, POSES, digest
from diagnostics.train_getup_fullpath_r27 import progress, rollout, save_trace, state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def contact_cell(m, gravity, knee_angles):
    """Keep low/sideways postures and knee support rather than pruning them."""
    return (int(np.floor(gravity[0]/.3)), int(np.floor(gravity[1]/.3)),
            int(np.floor(gravity[2]/.2)), int(np.floor(m['height_m']/.02)),
            *map(int, m['feet']), int(m['torso_contact']),
            int(np.floor(m['foot_load_fraction']/.25)),
            *map(int, np.asarray(knee_angles) >= 0.))


def prefix_trial(sim, pose, q, d):
    sim.prepare(pose)
    initial = sim.measure()
    initial_hash = state_hash(sim)
    sim.clear_audit()
    controls, reached, measures = 0, False, []
    for target, duration in zip(q, d):
        start = sim.prev.copy()
        n = max(1, round(float(duration)/DT))
        for i in range(n):
            a = min((i+1)/(.7*n), 1.); a = a*a*(3-2*a)
            sim.step_target((1-a)*start+a*target)
            controls += 1
            m = sim.measure(); measures.append(m)
            if not sim.physical_valid():
                break
        if not sim.physical_valid():
            break
        # Loaded strict standing is needed before actor handoff. Never count
        # upright torso-supported crouching as a stand.
        if m['stable']:
            reached = True
            break
    m = sim.measure()
    valid = bool(sim.physical_valid() and initial['up_z'] < .5 and initial['torso_contact'])
    gravity = sim.data.xmat[sim.model.body('trunk_assembly').id].reshape(3,3)[2].copy()
    cell = contact_cell(m, gravity, sim.data.qpos[sim.qadr[[3,12]]])
    tail = measures[-min(5,len(measures)):]
    quality = progress(m) if not tail else float(np.mean([progress(x) for x in tail]))
    return dict(valid=valid, entry=reached, metric=m, score=quality,
                cell=list(cell), previous_target=sim.prev.tolist(),
                actual_joint_angles=sim.data.qpos[sim.qadr].tolist(),
                gravity=gravity.tolist(), peaks=sim.peaks.copy(),
                initial_hash=initial_hash, control_steps=controls)


def mutate_target(sim, parent, rng, kind):
    current = np.asarray(parent['result']['actual_joint_angles'])
    previous = np.asarray(parent['result']['previous_target'])
    scale = np.array([.6,.6,1.,1.,1.,.8,.6,1.2,.5,.6,.6,1.,1.,1.])
    if kind == 0:
        q = sim.home.copy()
    elif kind == 1:
        q = current+rng.normal(0,scale*.4)
    elif kind == 2:
        q = previous+rng.normal(0,scale*.8)
    elif kind == 3:
        q = rng.uniform(sim.lower,sim.upper)
    elif kind == 4:
        # Symmetric sagittal lever followed later by independent leg mutations.
        q = sim.home.copy()
        hip, knee, ankle = rng.uniform(-.5,1.45), rng.uniform(-1.5,1.5), rng.uniform(-1.5,1.5)
        q[[2,11]], q[[3,12]], q[[4,13]] = [-hip,hip], knee, ankle
        q[5:9] = rng.uniform(sim.lower[5:9],sim.upper[5:9])
    elif kind == 5:
        q = previous.copy()
        indices = np.array([0,1,2,3,4,5,6,7,8]) if rng.random()<.5 else np.array([5,6,7,8,9,10,11,12,13])
        q[indices] += rng.normal(0,scale[indices])
    else:
        # Head/neck counterweight exploration; head yaw/roll are NOT frozen.
        q = previous.copy()
        q[5:9] = rng.uniform(sim.lower[5:9],sim.upper[5:9])
    return np.clip(q,sim.lower,sim.upper)


_sim = None
def init_worker(scene,stand):
    global _sim
    _sim = StrictSim(scene,stand)


def evaluate(job):
    return prefix_trial(_sim,*job)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pose',choices=POSES,required=True)
    p.add_argument('--generations',type=int,default=48)
    p.add_argument('--population',type=int,default=32)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--max-depth',type=int,default=14)
    p.add_argument('--seed',type=int,default=1030)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    scene=args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand=args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim=StrictSim(scene,stand)
    qualification=json.loads((scene.parent/'qualification.json').read_text())
    assert qualification['standing_passed_runs']==qualification['standing_runs']
    contract=dict(method='contact/orientation archive of full dynamic prefixes',
        simulation_only=True,hardware_readiness=False,default_controller_replaced=False,
        root_edits_after_initialization=0,substep_audits=True,
        final_gate='strict loaded home-standing continuous tail >=30s; >=18/20 held-out starts for EACH of four fall orientations',
        proposal_budget=args.generations*args.population,pose=args.pose,seed=args.seed,
        max_depth=args.max_depth,workers=args.workers,
        source_hash=digest(__file__),scene_hash=digest(scene),standing_actor_hash=digest(stand))
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    for f in (Path(__file__),Path(__file__).with_name('train_getup_fullpath_r27.py'),
              Path(__file__).with_name('validate_getup_fullpath_r27.py')):
        shutil.copy2(f,args.output/f.name)
    rng=np.random.default_rng(args.seed)
    q0=np.empty((0,14)); d0=np.empty(0)
    r0=prefix_trial(sim,args.pose,q0,d0)
    root_node=dict(q=q0,d=d0,result=r0,visits=0)
    archive={tuple(r0['cell']):root_node}
    best=root_node; total=0; rows=[]; found=None; start_time=time.monotonic()
    # Existing failed prefixes are seeds only; no teacher is assumed successful.
    priors=[]
    for file in (args.root/f'training/getup_fullpath_r27/{args.pose}/best_reference.npz',
                 args.root/f'training/getup_reverse_path_r29/{args.pose}_best_reference.npz'):
        if file.exists():
            z=np.load(file,allow_pickle=False)
            priors.extend((z['targets'][:i],z['durations_s'][:i]) for i in range(1,min(len(z['targets']),args.max_depth)+1))
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                             initargs=(str(scene),str(stand))) as pool:
        for gen in range(args.generations):
            candidates=[]
            eligible=[node for node in archive.values() if len(node['d'])<args.max_depth]
            for index in range(args.population):
                if priors:
                    q,d=priors.pop(0)
                else:
                    if index==0 and len(best['d'])<args.max_depth:
                        parent=best
                    elif index%8==0:
                        parent=root_node
                    else:
                        # A novel low-quality cell remains expandable. Less
                        # expanded cells get priority without score pruning.
                        draws=[eligible[int(rng.integers(len(eligible)))] for _ in range(3)]
                        parent=min(draws,key=lambda n:n['visits'])
                    parent['visits']+=1
                    target=mutate_target(sim,parent,rng,int(rng.integers(7)))
                    q=np.concatenate([parent['q'],target[None]])
                    d=np.concatenate([parent['d'],[rng.choice([.16,.26,.4,.6,.9])]])
                candidates.append((q.copy(),d.copy()))
            results=list(pool.map(evaluate,[(args.pose,q,d) for q,d in candidates],chunksize=1))
            total+=sum(r['control_steps'] for r in results)
            for (q,d),r in zip(candidates,results):
                if not r['valid']:
                    continue
                node=dict(q=q,d=d,result=r,visits=0)
                key=tuple(r['cell'])
                old=archive.get(key)
                if old is None or r['score']>old['result']['score']+.005:
                    archive[key]=node
                if r['score']>best['result']['score']:
                    best=node
                if r['entry']:
                    qualified,_=rollout(sim,args.pose,q,d,hold_s=31.)
                    if qualified['success']:
                        found=node; break
            chosen=found or best
            np.savez_compressed(args.output/'best_reference.npz',targets=chosen['q'],durations_s=chosen['d'])
            np.savez_compressed(args.output/'archive_paths.npz',
                **{f'targets_{i}':n['q'] for i,n in enumerate(archive.values())},
                **{f'durations_{i}':n['d'] for i,n in enumerate(archive.values())})
            row=dict(generation=gen+1,cells=len(archive),control_steps_total=total,
                     physically_valid=sum(r['valid'] for r in results),strict_entries=sum(r['entry'] for r in results),
                     best=chosen['result'],found_nominal_30s=found is not None,
                     wall_seconds=time.monotonic()-start_time)
            rows.append(row); (args.output/'progress.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps({k:row[k] for k in ('generation','cells','control_steps_total','physically_valid','strict_entries','found_nominal_30s','wall_seconds')} |
                             dict(best_up=chosen['result']['metric']['up_z'],best_height=chosen['result']['metric']['height_m'],best_load=chosen['result']['metric']['foot_load_fraction'])),flush=True)
            if found:
                break
    chosen=found or best
    q,d=chosen['q'],chosen['d']
    nominal,trace=rollout(sim,args.pose,q,d,hold_s=31.,record=True)
    save_trace(args.output/'nominal_trajectory.npz',trace)
    count,hold=(20,31.) if nominal['success'] else (3,4.)
    trials=[rollout(sim,args.pose,q,d,1000000+i,True,hold)[0] for i in range(count)]
    summary=dict(contract=contract,search_controls=total,archive_cells=len(archive),
        nominal=nominal,trials=trials,trial_count=count,validation_hold_s=hold,
        successes=sum(r['success'] for r in trials),
        orientation_passed=nominal['success'] and count==20 and sum(r['success'] for r in trials)>=18,
        full_task_completed=False,elapsed_seconds=time.monotonic()-start_time)
    (args.output/'results.json').write_text(json.dumps(summary,indent=2))
    print('ORIENTATION VALIDATION',args.pose,summary['successes'],'/',count,flush=True)


if __name__=='__main__':
    main()
