"""Single-factor common 10-control prefix, before any delayed selector training.

No new fitting or runtime case choice. Every frozen global program is tested
from every actual fallen start; original acceptance and actuator limits stay.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_local_hip_r130 as local
from diagnostics import probe_getup_complementary_feedback_r133 as prior
from diagnostics.getup_independent_native import digest

ROOT=local.ROOT
OUTPUT=ROOT/'outputs/getup_common_prefix_r136_left_20261006'
ORIGINAL_FEEDBACK=local.local_feedback
DELAY=0


def causal_feedback(current,nominal,ids,gains,delay):
    current=np.asarray(current,dtype=np.float32)
    if current.shape!=(55,) or not np.isfinite(current).all():raise ValueError('Actual sensor55 and phase required')
    if delay not in (0,10):raise ValueError('Only declared original or ten-control prefix')
    # The original observed causal control phase encodes the current step only.
    k=int(round(float(current[-1])*529))
    if k<delay:
        ORIGINAL_FEEDBACK(current,nominal,ids,gains)  # Still validate boundaries.
        return np.zeros(3)
    return ORIGINAL_FEEDBACK(current,nominal,ids,gains)


def init_worker(delay):
    global DELAY
    DELAY=delay;local.init_worker()
    local.local_feedback=lambda current,nominal,ids,gains:causal_feedback(current,nominal,ids,gains,DELAY)


def trial(job):
    index,gains,case,directory=job
    row=local.evaluate((gains,case,True,directory,index==0))
    file=Path(directory)/'trajectory.npz'
    with np.load(file,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    ref=prior.OUTPUT/f'program_{index:02d}'/f'case_{case}'
    assert row['initial_hash']==json.loads((ref/'result.json').read_text())['initial_hash']
    if DELAY==0 or index==0 or case is None:
        with np.load(ref/'trajectory.npz',allow_pickle=False) as z:parity={k:bool(np.array_equal(v,z[k])) for k,v in arrays.items()}
        assert all(parity.values());row['frozen_program_complete_parity']=parity
    if DELAY==10:
        original=local.prior.OUTPUT/'snapshot'/f'case_{case}'/'trajectory.npz'
        with np.load(original,allow_pickle=False) as z:equal={k:bool(np.array_equal(arrays[k][:10],z[k][:10])) for k in ('observations','normalized_residual','time','qpos','qvel','applied','strict')}
        assert all(equal.values());np.testing.assert_array_equal(arrays['local_hip_extra_rad'][:10],np.zeros((10,3)))
        row['common_ten_control_prefix_bitwise_equal']=equal
    row.update(prefix_controls=DELAY,program_index=index,new_weight_training=False)
    local.write_json(Path(directory)/'result.json',row);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    assert json.loads((prior.OUTPUT/'results.json').read_text())['complete_union_count']==24
    with np.load(prior.OUTPUT/'frozen_programs.npz',allow_pickle=False) as z:gains=z['gains'].copy()
    args.output.mkdir(exist_ok=False);executed=args.output/'executed_sources';executed.mkdir()
    for name in (Path(__file__).name,'test_getup_common_prefix_r136.py','launch_getup_common_prefix_r136.py','train_getup_local_hip_r130.py'):
        shutil.copy2(Path(__file__).with_name(name),executed/name)
    np.savez_compressed(args.output/'frozen_programs.npz',gains=gains)
    local.write_json(args.output/'contract.json',dict(smoke=args.smoke,workers=args.workers,initialization_seed=230,
        hypothesis='R134 has seven wrong initial decisions; test a common causal motion prefix and recollect complete program outcomes before considering later sensor decisions',
        prefix_controls=10,prefix_seconds=.2,no_selector_training_yet=True,no_case_lookup_or_privileged_input=True,
        no_mid_episode_state_edits=True,control_hz=50,physics_hz=500,full_controls=2279,
        original_entry_deadline_s=12,strict_tail_s=30,combined_feedback_cap_rad=.18,
        all_physical_limits_rewards_acceptance_unchanged=True,no_unseen_qualification=True,
        previous_success_labels_not_reused_for_delayed_programs=True,
        hashes={str(p):digest(p) for p in [prior.OUTPUT/'frozen_programs.npz',local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE,*executed.iterdir()]}))
    reports={}
    for delay in ((0,10) if args.smoke else (10,)):
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(delay,)) as pool:
            groups=[]
            for i,g in enumerate(gains):
                cases=[None,769000] if args.smoke else [None,*local.prior.program.TRAIN]
                directory=args.output/f'delay_{delay:02d}'/f'program_{i:02d}'
                rows=list(pool.map(trial,[(i,g.tolist(),s,str(directory/f'case_{s}')) for s in cases]))
                report=local.aggregate(rows);local.write_json(directory/'results.json',report);groups.append(report)
                print('R136_COMPLETE_PROGRAM',delay,i,report['successes'],report['physical_failures'],flush=True)
        reports[str(delay)]=groups
    result=dict(smoke=args.smoke,reports=reports,not_unified_controller_success=True,not_independent_qualification=True,
        no_selector_training_yet=True,hardware_readiness=False,full_task_completed=False)
    cases=[r['case_seed'] for r in reports['10'][0]['rows']]
    successes=np.array([[r['success'] for r in group['rows']] for group in reports['10']]).T
    result['complete_union_cases']=[s for s,ok in zip(cases,np.any(successes,axis=1)) if s is not None and ok]
    result['complete_union_count']=len(result['complete_union_cases'])
    local.write_json(args.output/'results.json',result);print('R136_TERMINAL',result['complete_union_count'],'NOT_UNIFIED',flush=True)


if __name__=='__main__':main()
