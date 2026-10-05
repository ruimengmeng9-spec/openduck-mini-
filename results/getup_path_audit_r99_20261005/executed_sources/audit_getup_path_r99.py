"""R99 actual complete-path duration/rate audit, not a qualification claim."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import numpy as np

from diagnostics.getup_independent_native import DT,digest
from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.train_getup_head_feedback_r87 import path
from diagnostics.search_getup_reference_feedback_r64 import features,residual
from diagnostics.train_getup_fullpath_r27 import state_hash


ROOT=Path('/data/shijinsheng/open_duck')
TRAIN=(769000,769001,769003)


def replay(sim,source,peaks,targets,phases,ids,ref,gains,stride):
    # Restore only the trial's own initial fallen state, before any command.
    sim.restore(source)
    sim.peaks,sim.finite=peaks.copy(),True
    initial=sim.measure();initial_hash=state_hash(sim)
    assert initial['up_z']<.5 and initial['torso_contact']
    first=None;tail=0;trace=[];target=sim.home.copy()
    for i,command in enumerate(targets):
        if i%stride==0:
            error=features(sim)-ref[i]
            target=command.copy()
            target[ids]=np.clip(target[ids]+residual(error,gains[phases[i]]),sim.lower[ids],sim.upper[ids])
        sim.step_target(target)
        metric=sim.measure();valid=sim.physical_valid()
        strict=valid and metric['stable']
        tail=tail+1 if strict else 0
        if strict and first is None:first=(i+1)*DT
        trace.append((sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),strict))
        if not valid:break
    row=dict(initial=initial,initial_hash=initial_hash,initial_fallen=True,
             completed_steps=i+1,valid=sim.physical_valid(),
             first_strict_s=first,strict_tail_s=tail*DT,
             success=bool(sim.physical_valid() and i+1==len(targets) and tail*DT>=30.-1e-8),
             final=metric,peaks=sim.peaks.copy(),root_edits_after_initialization=0)
    return row,trace


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    scene=ROOT/'training/getup_decomposed_r4/model/scene.xml'
    stand=ROOT/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    checkpoint=ROOT/'outputs/getup_prefix_pose_r73_left_20260930/checkpoint.npz'
    with np.load(checkpoint,allow_pickle=False) as data:ck={k:data[k].copy() for k in data.files}
    base.init_worker(str(scene),str(stand),ck,TRAIN)
    sim,cases,_,ids=base._CTX
    targets,phases,ref,gains=path(35.)
    sources=args.output/'executed_sources';sources.mkdir()
    names=(Path(__file__).name,'train_getup_wait_pose_r74.py','train_getup_head_feedback_r87.py',
           'probe_getup_terminal_timing_r72.py','search_getup_reference_feedback_r64.py',
           'train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py','getup_fullfallen_env_r32.py')
    for name in names:shutil.copy2(Path(__file__).with_name(name),sources/name)
    np.savez_compressed(args.output/'frozen_path.npz',targets=targets,phases=phases,reference=ref,gains=gains)
    rows=[]
    for seed,case in cases.items():
        for stride in (1,5):
            row,trace=replay(sim,*case[:2],targets,phases,ids,ref,gains,stride)
            row.update(seed=seed,decision_stride=stride,decision_hz=50/stride)
            rows.append(row)
            np.savez_compressed(args.output/f'case_{seed}_stride_{stride}.npz',
                time=[r[0] for r in trace],qpos=[r[1] for r in trace],qvel=[r[2] for r in trace],
                applied=[r[3] for r in trace],strict=[r[4] for r in trace])
            print('PATH_RATE',json.dumps(row),flush=True)
    report=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        development_only=True,pose='left_side',training_seeds=list(TRAIN),
        hashes={name:digest(sources/name) for name in names},
        scene_sha256=digest(scene),standing_actor_sha256=digest(stand),checkpoint_sha256=digest(checkpoint),
        target_steps=len(targets),target_seconds=len(targets)*DT,
        phase_durations_s={str(k):v*DT for k,v in Counter(int(x) for x in phases).items()},
        ppo_episode_seconds=300*DT,ppo_decision_hz=10.,
        motor_slew_physics_strict_thresholds_unchanged=True,results=rows)
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('R99_PATH_AUDIT_TERMINAL',json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__=='__main__':main()
