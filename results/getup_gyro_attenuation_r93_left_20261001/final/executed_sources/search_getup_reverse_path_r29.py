"""Generate actual stand-to-ground paths, test inverse targets from real falls.

This is a proposal generator, NOT an assumption that dissipative dynamics are
time-reversible. Every inverse candidate starts at the prescribed fallen pose;
no forward terminal state, root pose, velocity or force is transplanted.
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from diagnostics.getup_independent_native import DT, digest, POSES
from diagnostics.train_getup_fullpath_r27 import rollout, save_trace
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def fallen_orientation(quaternion):
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, np.asarray(quaternion))
    gravity = rotation.reshape(3, 3)[2]
    if gravity[2] >= .5 or max(abs(gravity[0]), abs(gravity[1])) < .6:
        return None
    if abs(gravity[0]) >= abs(gravity[1]):
        return 'prone' if gravity[0] < 0 else 'supine'
    return 'left_side' if gravity[1] < 0 else 'right_side'


def forward_fall(sim, targets, durations):
    sim.prepare('standing')
    sim.clear_audit()
    for target, duration in zip(targets, durations):
        start = sim.prev.copy()
        n = round(duration/DT)
        for i in range(n):
            a = min((i+1)/(.7*n), 1.); a = a*a*(3-2*a)
            sim.step_target((1-a)*start+a*target)
            if not sim.physical_valid():
                return None, dict(valid=False, peaks=sim.peaks.copy())
    m = sim.measure()
    pose = fallen_orientation(sim.data.qpos[3:7]) if m['torso_contact'] else None
    return pose, dict(valid=True, final=m, peaks=sim.peaks.copy(),
                      terminal_target=sim.prev.tolist())


def proposals(sim, rng, count):
    scale = np.array([.25, .35, .65, .85, .75, .45, .5, .5, .35,
                      .25, .35, .65, .85, .75])
    for index in range(count):
        q = np.tile(sim.home, (4, 1))
        if index % 3 == 0:
            # Bilateral sagittal family, with independently timed head leverage.
            hip = rng.uniform(-.4, 1.15)
            knee = rng.uniform(0., 1.55)
            ankle = rng.uniform(-1.3, 1.3)
            q[1:, 2], q[1:, 11] = -hip, hip
            q[1:, [3, 12]] = knee
            q[1:, [4, 13]] = ankle
            q[1:, 5] = rng.uniform(-.3, 1.1)
            q[1:, 6] = rng.uniform(-.7, .7)
            q[0] = .5*q[1]+.5*sim.home
        else:
            # Asymmetric legs can unload one foot and create a lateral fall.
            q += rng.normal(0., scale, q.shape)
            q[:, [3, 12]] = np.maximum(q[:, [3, 12]], 0.)
            q[-1] = q[-2]
        q = np.clip(q, sim.lower, sim.upper)
        d = rng.choice([.2, .4, .7], 4).astype(float)
        d[-1] = 1.
        yield q, d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--candidates',type=int,default=256)
    parser.add_argument('--seed',type=int,default=929)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    scene=args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand=args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim=StrictSim(scene,stand)
    for source in (Path(__file__),Path(__file__).with_name('train_getup_fullpath_r27.py'),
                   Path(__file__).with_name('validate_getup_fullpath_r27.py')):
        shutil.copy2(source,args.output/source.name)
    rng, rows, best = np.random.default_rng(args.seed),[],{}
    start=time.monotonic()
    qualified={pose:0 for pose in POSES}
    for index,(q,d) in enumerate(proposals(sim,rng,args.candidates)):
        pose, fall=forward_fall(sim,q,d)
        row=dict(index=index,forward=fall,pose=pose)
        if pose:
            qualified[pose]+=1
            # Start at canonical ground pose; align joint targets dynamically,
            # then replay the proposed reversed command sequence. No teleport.
            reverse=np.concatenate([q[-1:],q[::-1],sim.home[None]])
            durations=np.concatenate([[.8],d[::-1],[1.]])
            for timing in (.5,1.):
                times=durations.copy(); times[1:-1]*=timing
                result,_=rollout(sim,pose,reverse,times,hold_s=4.)
                row.setdefault('inverse_trials',[]).append(dict(timing_scale=timing,result=result))
                if pose not in best or result['score']>best[pose][0]['score']:
                    best[pose]=(result,reverse.copy(),times.copy())
                    np.savez_compressed(args.output/f'{pose}_best_reference.npz',targets=reverse,durations_s=times)
        rows.append(row)
        if (index+1)%16==0:
            (args.output/'progress.json').write_text(json.dumps(dict(completed=index+1,qualified_falls=qualified,
                best={pose:item[0] for pose,item in best.items()},wall_seconds=time.monotonic()-start),indent=2))
            print('REVERSE PROPOSALS',index+1,'qualified',qualified,'best_success',
                  {pose:item[0]['success'] for pose,item in best.items()},flush=True)
    validation={}
    for pose,(r,q,d) in best.items():
        nominal,trace=rollout(sim,pose,q,d,hold_s=31.,record=True)
        save_trace(args.output/f'{pose}_nominal_trajectory.npz',trace)
        count,hold=(20,31.) if nominal['success'] else (3,4.)
        trials=[rollout(sim,pose,q,d,900000+i,True,hold)[0] for i in range(count)]
        validation[pose]=dict(nominal=nominal,trial_count=count,heldout_hold_s=hold,
                              successes=sum(t['success'] for t in trials),trials=trials)
    summary=dict(rows=rows,validation=validation,qualified_forward_falls=qualified,
        full_task_completed=len(validation)==4 and all(v['nominal']['success'] and v['trial_count']==20
                                                        and v['successes']>=18 for v in validation.values()),
        simulation_only=True,hardware_readiness=False,default_controller_replaced=False,
        root_edits_after_initialization=0,seed=args.seed,wall_seconds=time.monotonic()-start,
        warning='Inverse commands are proposals; success requires fresh actual fallen dynamics, not mathematical reversibility.',
        hashes={str(f):digest(f) for f in (Path(__file__),scene,stand)})
    (args.output/'results.json').write_text(json.dumps(summary,indent=2))
    print('REVERSE VALIDATION',{k:v['successes'] for k,v in validation.items()},flush=True)


if __name__=='__main__':
    main()
