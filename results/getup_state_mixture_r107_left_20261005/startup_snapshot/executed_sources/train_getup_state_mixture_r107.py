"""Train a causal state-dependent mixture of zero and frozen R102 feedback.

The complete-development results contain 19 distinct successes across the two
profiles, but neither global profile exceeds 13. A small state gate tests whether
feedback benefit depends on the observed deviation, not merely clock time.
No seed IDs or outcome labels are policy inputs. This is parameter training,
not new neural feature training, and not proof of a causal failure mechanism.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
import diagnostics.train_getup_joint_anchor_r102 as r102
from diagnostics.train_getup_joint_anchor_r102 import aggregate,rank,write_json
from diagnostics.probe_getup_anchor_r101 import ROOT,SCENE,STAND,REFERENCE,MODEL
from diagnostics.getup_reference_env_r100 import ReferenceEpisode,TRAIN
from diagnostics.getup_independent_native import digest

OBSERVATIONS=GAINS=BASE_ROWS=None


def scale_for(parameters,observation,nominal):
    p=np.asarray(parameters,dtype=float)
    a=np.asarray(observation);b=np.asarray(nominal)
    if p.shape!=(5,) or not np.isfinite(p).all() or not (0<=p[0]<=1) or np.abs(p[1:]).max()>8:
        raise ValueError('Bounded five-dimensional causal mixture parameters required')
    if a.shape!=(55,) or b.shape!=(55,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite 55-dimensional current and nominal observations required')
    # Reference error components are [up_x,up_y,.15*gyro_y,.15*gyro_x].
    # Subtraction removes the shared stored-reference offset at this control step.
    delta=np.asarray(a[50:54],dtype=float)-np.asarray(b[50:54],dtype=float)
    return float(np.clip(p[0]+np.dot(p[1:],delta),0.,1.))


def init_worker(observations,anchors,gains):
    global OBSERVATIONS,GAINS,BASE_ROWS
    r102.init_worker(anchors);OBSERVATIONS=observations;GAINS=np.asarray(gains)
    prior=json.loads((ROOT/'outputs/getup_joint_anchor_r102_left_20261005/results.json').read_text())
    BASE_ROWS={r['case_seed']:r for r in prior['baseline']['rows']}


def evaluate(job):
    parameters,seed,qualification,directory=job
    env=ReferenceEpisode(str(SCENE),str(STAND),REFERENCE,207,qualification);env.reset(seed)
    trace=[];actions=[];scales=[]
    while True:
        observation=env.observe();k=env.controls
        scale=scale_for(parameters,observation,OBSERVATIONS[k])
        action=scale*r102.action_for(r102.WEIGHTS,observation,r102.ANCHORS[k],GAINS)
        row=env.step(action,auto_reset=False)
        if directory:
            actions.append(action);scales.append(scale)
            trace.append((env.sim.data.time,env.sim.data.qpos.copy(),env.sim.data.qvel.copy(),env.sim.prev.copy(),env.tail>0))
        if row[2]:break
    result=row[5]
    if seed in BASE_ROWS:assert result['initial_hash']==BASE_ROWS[seed]['initial_hash']
    result.update(parameters=list(parameters),reference_states_injected=False,
        policy_uses_case_identity=False,qualification=qualification,
        success=bool(result['valid'] and result['controls']==len(env.targets)
            and result['entry_time_s'] is not None and result['entry_time_s']<=12.
            and result['strict_tail_s']>=30.-1e-8) if qualification else bool(result['training_success']))
    if directory:
        np.savez_compressed(Path(directory)/f'case_{seed}.npz',time=[r[0] for r in trace],qpos=[r[1] for r in trace],
            qvel=[r[2] for r in trace],applied=[r[3] for r in trace],strict=[r[4] for r in trace],
            normalized_residual=actions,mixture_scale=scales)
        if seed is None:
            np.testing.assert_array_equal(np.array(actions),np.zeros((len(actions),10)))
            result['nominal_residual_exact_zero']=True
    return result


def group(pool,parameters,seeds,qualification,directory=None):
    if directory:directory.mkdir(exist_ok=False)
    rows=list(pool.map(evaluate,[(parameters.tolist(),s,qualification,str(directory) if directory else None) for s in seeds]))
    report=aggregate(rows)
    if directory:write_json(directory/'results.json',report)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--generations',type=int,default=24)
    p.add_argument('--population',type=int,default=12);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    if args.workers<1 or args.generations<1 or args.population<6:raise ValueError('Invalid search budget')
    args.output.mkdir(exist_ok=False)
    previous=ROOT/'outputs/getup_joint_anchor_r102_left_20261005'
    prior=json.loads((previous/'results.json').read_text());gains=np.array(prior['selected_gains'])
    with np.load(previous/'anchor.npz') as z:observations=z['observations'].copy();anchors=z['actions'].copy()
    np.savez_compressed(args.output/'frozen_profiles.npz',observations=observations,anchors=anchors,gains=gains)
    sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_state_mixture_r107.py','train_getup_joint_anchor_r102.py',
        'getup_reference_env_r100.py','getup_fullfallen_env_r32.py','getup_fullfallen_contract_r32.py',
        'probe_getup_anchor_r101.py','search_getup_reference_feedback_r64.py','search_getup_sustained_bridge_r42.py',
        'getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    base_success={r['case_seed'] for r in prior['baseline']['rows'] if r['case_seed'] is not None and r['success']}
    candidate_success={r['case_seed'] for r in prior['development']['rows'] if r['case_seed'] is not None and r['success']}
    contract=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,smoke=args.smoke,
        hypothesis='State deviation may discriminate when the frozen feedback profile helps versus damages recovery',
        method='CEM over 5 causal mixture parameters; neural features and joint profile frozen',
        motivating_profile_successes=[len(base_success),len(candidate_success)],
        profile_union_successes=len(base_success|candidate_success),union_is_not_a_policy_result=True,
        policy_input='current minus nominal four IMU error features and scalar bias; no seed or outcome input',
        mixture_bounds=[0.,1.],parameter_bias_bounds=[0.,1.],parameter_slope_bounds=[-8.,8.],
        seed=207,generations=args.generations,population=args.population,workers=args.workers,
        cases=[None,*TRAIN],training_controls=629,qualification_controls=2279,
        physics_hz=500,decision_hz=50,combined_target_residual_cap_rad=.18,
        physics_and_rewards_and_acceptance_unchanged=True,no_intermediate_root_reset=True,
        reference_states_never_injected=True,home_has_no_extra_network_feedback=True,
        gate=dict(development=22,independent_groups=[18,18],standing_s=30,entry_deadline_s=12),
        independent_seeds=list(range(3160000,3160040)),
        hashes={str(f):digest(f) for f in [SCENE,STAND,REFERENCE,MODEL,previous/'results.json',previous/'anchor.npz',*sources.iterdir()]})
    write_json(args.output/'contract.json',contract)
    zero=np.zeros(5);one=np.array([1.,0.,0.,0.,0.]);rng=np.random.default_rng(207)
    selected=one.copy();mean=np.array([.5,0.,0.,0.,0.]);std=np.array([.3,2.,2.,2.,2.])
    history=[];cache={};best=None
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,
        initargs=(observations,anchors,gains)) as pool:
        for name,parameters,old_directory in [('zero',zero,'development_baseline'),('one',one,'development_candidate')]:
            report=group(pool,parameters,[None,769000],True,args.output/('parity_'+name))
            for row in report['rows']:
                with np.load(args.output/('parity_'+name)/f"case_{row['case_seed']}.npz") as current:
                    with np.load(previous/old_directory/f"case_{row['case_seed']}.npz") as original:
                        for field in ('qpos','qvel','applied'):np.testing.assert_array_equal(current[field],original[field])
            print('R107_PROFILE_PARITY',name,'PASS',flush=True)
        if args.smoke:
            report=group(pool,mean,[None,769000],False,args.output/'smoke_mixture')
            write_json(args.output/'results.json',dict(smoke=True,report=report,full_task_completed=False))
        else:
            seeds=[None,*TRAIN]
            for generation in range(args.generations):
                random=rng.normal(mean,std,(args.population-3,5));random[:,0]=np.clip(random[:,0],0.,1.)
                random[:,1:]=np.clip(random[:,1:],-8.,8.)
                candidates=[zero.copy(),one.copy(),selected.copy(),*random]
                reports=[]
                for i,parameters in enumerate(candidates):
                    key=parameters.tobytes().hex()
                    if key not in cache:cache[key]=group(pool,parameters,seeds,False)
                    report=cache[key];reports.append(report)
                    print('R107_CANDIDATE',generation+1,i,report['successes'],report['nominal_success'],report['physical_failures'],flush=True)
                ordered=sorted(range(len(candidates)),key=lambda i:rank(reports[i]),reverse=True)
                winner=ordered[0]
                if best is None or rank(reports[winner])>rank(best):selected=candidates[winner].copy();best=reports[winner]
                elite=np.stack([candidates[i] for i in ordered[:max(3,args.population//4)]])
                mean=.6*mean+.4*elite.mean(0);std=.6*std+.4*elite.std(0)
                std=np.clip(std,[.05,.15,.15,.15,.15],[.4,3.,3.,3.,3.])
                history.append(dict(generation=generation+1,candidates=[c.tolist() for c in candidates],reports=reports,
                    best_parameters=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
                np.savez_compressed(args.output/f'checkpoint_{generation+1:04d}.npz',parameters=selected,mean=mean,std=std)
                write_json(args.output/'history.json',history)
                write_json(args.output/'progress.json',dict(completed_generations=generation+1,best_parameters=selected.tolist(),
                    best_training_successes=best['successes'],rng_state=rng.bit_generator.state,full_task_completed=False))
                print('R107_GENERATION',generation+1,best['successes'],selected.tolist(),flush=True)
            development=group(pool,selected,seeds,True,args.output/'development_candidate')
            baseline=group(pool,zero,seeds,True,args.output/'development_baseline')
            previous_profile=group(pool,one,seeds,True,args.output/'development_r102')
            promoted=bool(development['nominal_success'] and development['successes']>=22
                and development['successes']>max(baseline['successes'],previous_profile['successes'])
                and development['physical_failures']==0)
            independent=None;passed=False
            if promoted:
                np.savez_compressed(args.output/'frozen_candidate.npz',parameters=selected)
                fresh=list(range(3160000,3160040))
                candidate=group(pool,selected,fresh,True,args.output/'independent_candidate')
                reference=group(pool,zero,fresh,True,args.output/'independent_baseline')
                for a,b in zip(candidate['rows'],reference['rows']):assert a['initial_hash']==b['initial_hash']
                groups=[sum(r['success'] for r in candidate['rows'][i:i+20]) for i in (0,20)]
                passed=all(v>=18 for v in groups) and candidate['physical_failures']==0
                independent=dict(candidate=candidate,baseline=reference,group_successes=groups)
            write_json(args.output/'results.json',dict(development=development,baseline=baseline,
                previous_profile=previous_profile,selected_parameters=selected.tolist(),candidate_promoted=promoted,
                independent=independent,left_stage_passed=passed,full_task_completed=False,
                simulation_only=True,hardware_readiness=False))
    print('R107_TERMINAL',flush=True)


if __name__=='__main__':main()
