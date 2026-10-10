"""Train bounded joint gains on frozen nominal-anchored R100 features.

This is controller parameter training, not new end-to-end network training.
Physics, reference, rewards and strict acceptance are imported unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np

from diagnostics.probe_getup_anchor_r101 import ROOT,SCENE,STAND,REFERENCE,MODEL,nominal_observations
from diagnostics.getup_reference_env_r100 import ReferenceEpisode,TRAIN,numpy_action
from diagnostics.getup_independent_native import digest

WEIGHTS=ANCHORS=None


def init_worker(anchors):
    global WEIGHTS,ANCHORS
    with np.load(MODEL) as data:WEIGHTS={k:data[k].copy() for k in data.files}
    ANCHORS=anchors


def action_for(weights,observation,anchor,gains):
    gains=np.asarray(gains,dtype=np.float64)
    if gains.shape!=(10,) or not np.isfinite(gains).all() or np.max(np.abs(gains))>4.:
        raise ValueError('Ten finite feedback gains bounded by four required')
    return np.clip(gains*(numpy_action(weights,observation)-anchor),-1.,1.)


def evaluate(job):
    gains,seed,qualification,directory=job
    env=ReferenceEpisode(str(SCENE),str(STAND),REFERENCE,200,qualification)
    env.reset(seed);trace=[];actions=[]
    while True:
        action=action_for(WEIGHTS,env.observe(),ANCHORS[env.controls],gains)
        result=env.step(action,auto_reset=False)
        if directory:
            actions.append(action)
            trace.append((env.sim.data.time,env.sim.data.qpos.copy(),env.sim.data.qvel.copy(),env.sim.prev.copy(),env.tail>0))
        if result[2]:break
    row=result[5]
    row['success']=bool(row['valid'] and row['controls']==len(env.targets)
        and row['entry_time_s'] is not None and row['entry_time_s']<=12.
        and row['strict_tail_s']>=30.-1e-8) if qualification else bool(row['training_success'])
    row.update(qualification=qualification,reference_states_injected=False,gains=list(gains))
    if directory:
        path=Path(directory)/f'case_{seed}.npz'
        np.savez_compressed(path,time=[r[0] for r in trace],qpos=[r[1] for r in trace],
            qvel=[r[2] for r in trace],applied=[r[3] for r in trace],strict=[r[4] for r in trace],normalized_residual=actions)
        if seed is None:
            np.testing.assert_array_equal(np.array(actions),np.zeros((len(trace),10)))
            row['nominal_residual_exact_zero']=True
    return row


def aggregate(rows):
    nominal=[r for r in rows if r['case_seed'] is None]
    return dict(rows=rows,successes=sum(r['success'] for r in rows if r['case_seed'] is not None),
        nominal_success=nominal[0]['success'] if nominal else None,
        physical_failures=sum(not r['valid'] for r in rows),
        return_sum=float(sum(r['return_sum'] for r in rows)))


def rank(report):
    # Nominal retention is mandatory; success is the existing short-tail label.
    # The unmodified environment return only breaks ties; no reward is changed.
    return (bool(report['nominal_success']),report['successes'],-report['physical_failures'],report['return_sum'])


def write_json(path,data):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def evaluate_group(pool,gains,seeds,qualification,directory=None):
    if directory:directory.mkdir(exist_ok=False)
    rows=list(pool.map(evaluate,[(gains.tolist(),s,qualification,str(directory) if directory else None) for s in seeds]))
    report=aggregate(rows)
    if directory:write_json(directory/'results.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--generations',type=int,default=24)
    p.add_argument('--population',type=int,default=12);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    if args.population<6:raise ValueError('At least six candidates required')
    observations=nominal_observations()
    with np.load(MODEL) as data:weights={k:data[k].copy() for k in data.files}
    anchors=np.stack([numpy_action(weights,row) for row in observations])
    np.savez_compressed(args.output/'anchor.npz',observations=observations,actions=anchors)
    sources=args.output/'executed_sources';sources.mkdir()
    names=[Path(__file__).name,'test_getup_joint_anchor_r102.py','probe_getup_anchor_r101.py',
        'getup_reference_env_r100.py','getup_fullfallen_env_r32.py','getup_fullfallen_contract_r32.py',
        'search_getup_reference_feedback_r64.py','search_getup_sustained_bridge_r42.py',
        'getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py']
    for name in names:shutil.copy2(Path(__file__).with_name(name),sources/name)
    contract=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        hypothesis='Joint-specific anchored feedback may avoid damage hidden by a shared scalar gain',
        training_method='CEM over ten bounded feedback gains; frozen R100 neural features',seed=202,
        generations=args.generations,population=args.population,workers=args.workers,
        normalized_action_cap=1.,combined_target_residual_cap_rad=.18,gain_bounds=[-4.,4.],
        physics_and_rewards_and_acceptance_unchanged=True,no_intermediate_root_reset=True,
        reference_states_never_injected=True,training_cases=[None,*TRAIN],
        training_controls=629,qualification_controls=2279,decision_hz=50,physics_hz=500,
        qualification_gate=dict(development=22,independent_groups=[18,18],standing_s=30),
        independent_seeds=list(range(3160000,3160040)),smoke=args.smoke,
        hashes={str(f):digest(f) for f in [SCENE,STAND,REFERENCE,MODEL,*sources.iterdir()]})
    write_json(args.output/'contract.json',contract)
    zero=np.zeros(10);rng=np.random.default_rng(202);mean=zero.copy();std=np.full(10,.8)
    selected=zero.copy();history=[];cache={};best=None
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(anchors,)) as pool:
        # Mandatory end-to-end parity before any optimization.
        parity=evaluate_group(pool,zero,[None,769000],True,args.output/'zero_parity')
        baseline=json.loads((ROOT/'outputs/getup_reference_residual_r100_left_20261005/baseline_development/results.json').read_text())
        baseline_rows={r['case_seed']:r for r in baseline['rows']}
        for row in parity['rows']:
            old=baseline_rows[row['case_seed']]
            assert row['initial_hash']==old['initial_hash'] and row['success']==old['success']
            with np.load(args.output/'zero_parity'/f"case_{row['case_seed']}.npz") as current:
                with np.load(ROOT/'outputs/getup_reference_residual_r100_left_20261005/baseline_development'/f"case_{row['case_seed']}.npz") as previous:
                    for field in ('qpos','qvel','applied'):np.testing.assert_array_equal(current[field],previous[field])
        print('R102_ZERO_PARITY_PASSED',flush=True)
        if args.smoke:
            probe=evaluate_group(pool,np.ones(10),[None,769000],False,args.output/'smoke_anchored')
            write_json(args.output/'results.json',dict(smoke=True,parity=parity,probe=probe,full_task_completed=False))
        else:
            seeds=[None,*TRAIN]
            for generation in range(args.generations):
                candidates=[zero.copy(),selected.copy(),*np.clip(rng.normal(mean,std,(args.population-2,10)),-4.,4.)]
                reports=[]
                for i,gains in enumerate(candidates):
                    key=gains.tobytes().hex()
                    if key not in cache:cache[key]=evaluate_group(pool,gains,seeds,False)
                    report=cache[key];reports.append(report)
                    print('R102_CANDIDATE',generation+1,i,report['successes'],report['nominal_success'],report['physical_failures'],flush=True)
                ordered=sorted(range(len(candidates)),key=lambda i:rank(reports[i]),reverse=True)
                winner=ordered[0]
                if best is None or rank(reports[winner])>rank(best):
                    selected=candidates[winner].copy();best=reports[winner]
                elite=np.stack([candidates[i] for i in ordered[:max(3,args.population//4)]])
                mean=.6*mean+.4*elite.mean(0);std=np.clip(.6*std+.4*elite.std(0),.1,1.5)
                history.append(dict(generation=generation+1,candidates=[g.tolist() for g in candidates],reports=reports,
                    best_gains=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
                np.savez_compressed(args.output/f'checkpoint_{generation+1:04d}.npz',gains=selected,mean=mean,std=std)
                write_json(args.output/'history.json',history)
                write_json(args.output/'progress.json',dict(completed_generations=generation+1,best_gains=selected.tolist(),
                    best_training_successes=best['successes'],full_task_completed=False,rng_state=rng.bit_generator.state))
                print('R102_GENERATION',generation+1,best['successes'],selected.tolist(),flush=True)
            development=evaluate_group(pool,selected,seeds,True,args.output/'development_candidate')
            base=evaluate_group(pool,zero,seeds,True,args.output/'development_baseline')
            promoted=bool(development['nominal_success'] and development['successes']>=22
                and development['successes']>base['successes'] and development['physical_failures']==0)
            qualification=None;passed=False
            if promoted:
                np.savez_compressed(args.output/'frozen_candidate.npz',gains=selected)
                fresh=list(range(3160000,3160040))
                candidate=evaluate_group(pool,selected,fresh,True,args.output/'independent_candidate')
                reference=evaluate_group(pool,zero,fresh,True,args.output/'independent_baseline')
                for a,b in zip(candidate['rows'],reference['rows']):assert a['initial_hash']==b['initial_hash']
                groups=[sum(r['success'] for r in candidate['rows'][i:i+20]) for i in (0,20)]
                passed=all(n>=18 for n in groups) and candidate['physical_failures']==0
                qualification=dict(candidate=candidate,baseline=reference,group_successes=groups)
            write_json(args.output/'results.json',dict(development=development,baseline=base,
                selected_gains=selected.tolist(),candidate_promoted=promoted,qualification=qualification,
                left_stage_passed=passed,full_task_completed=False,simulation_only=True,hardware_readiness=False))
    print('R102_TERMINAL',flush=True)


if __name__=='__main__':main()
