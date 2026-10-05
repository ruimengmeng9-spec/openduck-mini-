"""Frozen R102 feedback timing counterfactuals, original full qualification."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics.train_getup_joint_anchor_r102 import init_worker,evaluate,action_for,aggregate,write_json
import diagnostics.train_getup_joint_anchor_r102 as r102
from diagnostics.probe_getup_anchor_r101 import ROOT,SCENE,STAND,REFERENCE,MODEL
from diagnostics.getup_reference_env_r100 import ReferenceEpisode,TRAIN
from diagnostics.getup_independent_native import digest

WINDOWS={'early':(0.,2.6),'late':(2.6,10.58),'local':(1.6,2.6),'prefix':(0.,6.16)}


def enabled(name,control):
    a,b=WINDOWS[name]
    return a<=control*.02<b


def gated_trial(job):
    seed,name,gains,directory=job
    env=ReferenceEpisode(str(SCENE),str(STAND),REFERENCE,200,True);env.reset(seed)
    trace=[];actions=[]
    while True:
        action=(action_for(r102.WEIGHTS,env.observe(),r102.ANCHORS[env.controls],gains)
            if enabled(name,env.controls) else np.zeros(10))
        actions.append(action);row=env.step(action,auto_reset=False)
        trace.append((env.sim.data.time,env.sim.data.qpos.copy(),env.sim.data.qvel.copy(),env.sim.prev.copy(),env.tail>0))
        if row[2]:break
    result=row[5]
    result.update(success=bool(result['valid'] and result['controls']==len(env.targets)
        and result['entry_time_s'] is not None and result['entry_time_s']<=12.
        and result['strict_tail_s']>=30.-1e-8),window=name,reference_states_injected=False)
    np.savez_compressed(Path(directory)/f'case_{seed}.npz',time=[r[0] for r in trace],qpos=[r[1] for r in trace],
        qvel=[r[2] for r in trace],applied=[r[3] for r in trace],strict=[r[4] for r in trace],normalized_residual=actions)
    if seed is None:
        np.testing.assert_array_equal(np.asarray(actions),np.zeros((len(actions),10)))
        result['nominal_residual_exact_zero']=True
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);args=p.parse_args();args.output.mkdir(exist_ok=False)
    original=ROOT/'outputs/getup_joint_anchor_r102_left_20261005'
    previous=json.loads((original/'results.json').read_text())
    gains=np.array(previous['selected_gains'])
    with np.load(original/'anchor.npz') as d:anchors=d['actions'].copy()
    np.savez_compressed(args.output/'frozen_controller.npz',gains=gains,anchors=anchors)
    sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_temporal_gate_r104.py','train_getup_joint_anchor_r102.py',
        'getup_reference_env_r100.py','getup_fullfallen_env_r32.py','getup_fullfallen_contract_r32.py',
        'probe_getup_anchor_r101.py','search_getup_reference_feedback_r64.py',
        'search_getup_sustained_bridge_r42.py','getup_independent_native.py',
        'train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    contract=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        development_only=True,windows=WINDOWS,cases=[None,*TRAIN],workers=args.workers,
        hypothesis='Frozen global joint feedback may have incompatible effects at early and later phases',
        physics_and_rewards_and_acceptance_unchanged=True,no_root_edits_after_initialization=True,
        reference_states_never_injected=True,combined_target_residual_cap_rad=.18,
        qualification_controls=2279,standing_s=30,entry_deadline_s=12,
        hashes={str(f):digest(f) for f in [SCENE,STAND,REFERENCE,MODEL,original/'results.json',*sources.iterdir()]})
    write_json(args.output/'contract.json',contract);reports={}
    baseline=previous['baseline'];base_rows={r['case_seed']:r for r in baseline['rows']}
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(anchors,)) as pool:
        for name in WINDOWS:
            directory=args.output/name;directory.mkdir()
            rows=[]
            for row in pool.map(gated_trial,[(s,name,gains.tolist(),str(directory)) for s in [None,*TRAIN]]):
                assert row['initial_hash']==base_rows[row['case_seed']]['initial_hash']
                rows.append(row);print('R104_CASE',name,row['case_seed'],row['success'],row['valid'],flush=True)
            report=aggregate(rows)
            report['recoveries']=[r['case_seed'] for r in rows if r['success'] and not base_rows[r['case_seed']]['success']]
            report['regressions']=[r['case_seed'] for r in rows if not r['success'] and base_rows[r['case_seed']]['success']]
            write_json(directory/'results.json',report);reports[name]=report
            write_json(args.output/'partial_results.json',reports)
            print('R104_RESULT',name,report['successes'],report['nominal_success'],report['physical_failures'],flush=True)
        eligible=[name for name,r in reports.items() if r['nominal_success'] and r['physical_failures']==0
            and r['successes']>=22 and r['successes']>baseline['successes']]
        independent=None;passed=False
        if eligible:
            winner=max(eligible,key=lambda name:reports[name]['successes'])
            write_json(args.output/'frozen_selection.json',dict(window=winner,gains=gains.tolist()))
            candidate_dir=args.output/'independent_candidate';candidate_dir.mkdir()
            fresh=list(range(3160000,3160040))
            candidate=list(pool.map(gated_trial,[(s,winner,gains.tolist(),str(candidate_dir)) for s in fresh]))
            reference_dir=args.output/'independent_baseline';reference_dir.mkdir()
            reference=list(pool.map(evaluate,[(np.zeros(10).tolist(),s,True,str(reference_dir)) for s in fresh]))
            for a,b in zip(candidate,reference):assert a['initial_hash']==b['initial_hash']
            groups=[sum(r['success'] for r in candidate[i:i+20]) for i in (0,20)]
            passed=all(v>=18 for v in groups) and all(r['valid'] for r in candidate)
            independent=dict(candidate=candidate,baseline=reference,group_successes=groups)
            write_json(candidate_dir/'results.json',dict(rows=candidate,group_successes=groups))
            write_json(reference_dir/'results.json',dict(rows=reference))
        write_json(args.output/'results.json',dict(reports=reports,baseline_successes=baseline['successes'],
            original_r102_successes=previous['development']['successes'],independent=independent,
            left_stage_passed=passed,full_task_completed=False,simulation_only=True,hardware_readiness=False))
    print('R104_TERMINAL',flush=True)


if __name__=='__main__':main()
