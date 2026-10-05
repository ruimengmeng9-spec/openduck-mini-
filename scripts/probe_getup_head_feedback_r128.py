"""Bounded structural test: causal head-joint tracking, frozen R122 legs.

Only actual sensor angles minus nominal sensor angles drive four extra head
targets, not root truth, case keys or teacher recipes. No parameter training.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_history_program_r122 as previous
from diagnostics.getup_independent_native import digest

ROOT=previous.ROOT
OUTPUT=ROOT/'outputs/getup_head_feedback_r128_left_20261005'
HEAD=('neck_pitch','head_pitch','head_yaw','head_roll')
CASES=(None,769000,773004,3180019,3180038)
GAINS=(0.,1.,4.)


def sensor_feedback(current,nominal,indices,gain):
    current=np.asarray(current,dtype=np.float32);nominal=np.asarray(nominal,dtype=np.float32)
    if current.shape!=(55,) or nominal.shape!=(55,) or not np.isfinite(current).all() or not np.isfinite(nominal).all():
        raise ValueError('Only finite current and nominal sensor55 allowed')
    if gain not in GAINS:raise ValueError('Predeclared bounded gains only')
    indices=np.asarray(indices,dtype=int)
    if indices.shape!=(4,) or len(set(indices))!=4 or np.any(indices<0) or np.any(indices>=14):raise ValueError('Four distinct joint sensor indices required')
    return np.clip(-gain*(current[6+indices]-nominal[6+indices]).astype(float),-.18,.18)


def trial(job):
    case,gain,output=job;assert case in CASES and case in previous.audit.KNOWN
    file=previous.OUTPUT/'training/snapshot.npz'
    with np.load(file,allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
    with np.load(previous.OUTPUT/'snapshot/case_None/trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    env=previous.audit.HistoryReferenceEpisode(str(previous.program.SCENE),str(previous.program.STAND),previous.program.REFERENCE,228,True)
    env.reset(case);sim=env.sim
    actuator_ids=np.array([sim.model.actuator(n).id for n in HEAD])
    sensor_ids=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in HEAD])
    profile,knots,_=previous.scalar_program(model,env.preparation_sensors)
    original_step=sim.step_target;extras=[]
    def controlled(target):
        extra=sensor_feedback(env.observe(),nominal[env.controls],sensor_ids,gain) if env.controls<529 else np.zeros(4)
        extras.append(extra.copy())
        if np.any(extra!=0.):
            target=target.copy();target[actuator_ids]=np.clip(target[actuator_ids]+extra,sim.lower[actuator_ids],sim.upper[actuator_ids])
        return original_step(target)
    sim.step_target=controlled;records=[]
    try:
        while True:
            obs=env.observe();action=previous.program.execute_program(model,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    finally:sim.step_target=original_step
    row=step[5];row.update(gain=gain,success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12. and row['strict_tail_s']>=30.-1e-8),
        no_new_weight_training=True,no_case_metadata_in_control=True,head_feedback_max_abs_rad=float(np.abs(extras).max()))
    directory=Path(output);directory.mkdir(parents=True,exist_ok=False)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],
        qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],head_correction_rad=extras)
    olddir=previous.OUTPUT/'snapshot'/f'case_{case}'
    old=json.loads((olddir/'result.json').read_text());assert row['initial_hash']==old['initial_hash']
    if gain==0. or case is None:
        with np.load(olddir/'trajectory.npz',allow_pickle=False) as stored:
            parity={name:bool(np.array_equal(values,stored[name])) for name,values in arrays.items() if name!='head_correction_rad'}
        assert all(parity.values());row['original_complete_trace_bitwise_equal']=parity
        if case is None:np.testing.assert_array_equal(np.array(extras),np.zeros((2279,4)))
    np.savez_compressed(directory/'trajectory.npz',**arrays)
    (directory/'result.json').write_text(json.dumps(row,indent=2));return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_head_feedback_r128.py','launch_getup_head_feedback_r128.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    contract=dict(seed=228,smoke=args.smoke,workers=args.workers,head_joints=HEAD,predeclared_gains=GAINS,cases=CASES,
        hypothesis='Current ten-leg residual cannot directly correct observed head-trunk collision; test four bounded causal head joint corrections against frozen R122 snapshot',
        frozen_leg_policy=True,head_feedback_cap_rad=.18,head_home_feedback_unchanged=True,
        input_actual_joint_sensors_only=True,no_world_root_truth=True,no_context_lookup=True,
        no_new_parameter_training=True,physics_rewards_acceptance_unchanged=True,
        no_mid_episode_root_reset=True,independent_qualification_run=False,reserved_qualification_never_loaded=True,
        hashes={str(f):digest(f) for f in [previous.program.SCENE,previous.program.STAND,previous.program.REFERENCE,
            previous.OUTPUT/'training/snapshot.npz',previous.OUTPUT/'snapshot/case_None/trajectory.npz',*sources.iterdir()]})
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    jobs=[(s,g,str(args.output/f'gain_{g}/case_{s}')) for g in ((0.,1.) if args.smoke else GAINS) for s in ((None,3180019) if args.smoke else CASES)]
    rows=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for row in pool.map(trial,jobs):
            rows.append(row);(args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),total=len(jobs),last=row),indent=2))
            print('R128_TRIAL',row['gain'],row['case_seed'],row['success'],row['valid'],row['peaks'],flush=True)
    result=dict(rows=rows,smoke=args.smoke,not_training=True,not_independent_qualification=True,
        no_candidate_promotion=True,not_causal_proof_of_all_failures=True,hardware_readiness=False,full_task_completed=False)
    (args.output/'results.json').write_text(json.dumps(result,indent=2));print('R128_TERMINAL',len(rows),flush=True)


if __name__=='__main__':main()
