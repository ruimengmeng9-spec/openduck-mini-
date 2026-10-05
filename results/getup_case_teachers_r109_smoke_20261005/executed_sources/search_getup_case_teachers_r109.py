"""Generate complete-fall teacher trajectories, not a seed-indexed policy.

R108 demonstrates interacting initial-state sensitivities. Instead of another
global frozen-feature gain search, optimize a bounded smooth residual path for
each of the five development starts missed by both existing controllers. Every
proposal is replayed from its own original complete fallen start. These are
offline teacher records; no library lookup is allowed as a deployed policy.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_joint_anchor_r102 as r102
from diagnostics.train_getup_joint_anchor_r102 import write_json
from diagnostics.getup_reference_env_r100 import ReferenceEpisode, TRAIN
from diagnostics.probe_getup_anchor_r101 import ROOT, SCENE, STAND, REFERENCE, MODEL
from diagnostics.getup_independent_native import digest

NODES=np.array([0,50,100,150,200,300,400,529])
GAINS=BASE_ROWS=None


def knot_action(knots, control):
    x=np.asarray(knots,dtype=float)
    if x.shape!=(6,10) or not np.isfinite(x).all() or np.abs(x).max()>1.:
        raise ValueError('Six bounded ten-joint normalized residual knots required')
    if control>=529:return np.zeros(10)
    values=np.vstack((np.zeros((1,10)),x,np.zeros((1,10))))
    return np.array([np.interp(control,NODES,values[:,j]) for j in range(10)])


def init_worker(anchors,gains):
    global GAINS,BASE_ROWS
    r102.init_worker(anchors);GAINS=np.asarray(gains)
    prior=json.loads((ROOT/'outputs/getup_state_mixture_r107_left_20261005/results.json').read_text())
    BASE_ROWS={r['case_seed']:r for r in prior['baseline']['rows']}


def rollout(job):
    seed,profile,knots,full,directory,parity_folder=job
    if profile not in (0,1):raise ValueError('Fixed feedback profile must be zero or R102')
    env=ReferenceEpisode(str(SCENE),str(STAND),REFERENCE,209,full);env.reset(seed)
    assert env.initial_hash==BASE_ROWS[seed]['initial_hash']
    records=[]
    while True:
        obs=env.observe();k=env.controls
        base=np.zeros(10) if profile==0 else r102.action_for(r102.WEIGHTS,obs,r102.ANCHORS[k],GAINS)
        extra=knot_action(knots,k)
        action=np.clip(base+extra,-1.,1.)
        step=env.step(action,auto_reset=False)
        if directory:records.append((obs.copy(),action.copy(),env.sim.data.time,env.sim.data.qpos.copy(),
            env.sim.data.qvel.copy(),env.sim.prev.copy(),env.tail>0))
        if step[2]:break
    row=step[5]
    row.update(profile=profile,teacher_data_only=True,qualified_teacher=False,
        unified_policy_success=False,full_path=full,
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None
            and row['entry_time_s']<=12. and row['strict_tail_s']>=30.-1e-8) if full else bool(row['training_success']))
    if full:row['qualified_teacher']=row['success']
    if directory:
        dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
        arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],
            time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],
            applied=[r[5] for r in records],strict=[r[6] for r in records])
        if parity_folder:
            with np.load(Path(parity_folder)/f'case_{seed}.npz') as old:
                for key in ('qpos','qvel','applied','strict','normalized_residual'):
                    np.testing.assert_array_equal(np.array(arrays[key]),old[key])
            row['original_controller_full_path_bitwise_parity']=True
        np.savez_compressed(dest/'trajectory.npz',**arrays)
        np.savez_compressed(dest/'teacher_parameters.npz',knots=knots,profile=profile)
        write_json(dest/'result.json',row)
    return row


def rank(row):
    return bool(row['training_success']),bool(row['valid']),row['return_sum']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=6);parser.add_argument('--generations',type=int,default=24)
    parser.add_argument('--population',type=int,default=14);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if not 1<=args.workers<=6 or args.population<6 or args.generations<1:raise ValueError('Bounded fresh search required')
    args.output.mkdir(exist_ok=False)
    prior_dir=ROOT/'outputs/getup_state_mixture_r107_left_20261005'
    prior=json.loads((prior_dir/'results.json').read_text())
    successes={p:{r['case_seed'] for r in prior[name]['rows'] if r['success']}
        for p,name in [(0,'baseline'),(1,'previous_profile')]}
    missing=[s for s in TRAIN if s not in successes[0]|successes[1]]
    assert missing==[773004,773005,773007,773011,773015]
    with np.load(prior_dir/'frozen_profiles.npz') as data:anchors=data['anchors'].copy();gains=data['gains'].copy()
    np.savez_compressed(args.output/'frozen_profiles.npz',anchors=anchors,gains=gains)
    sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_case_teachers_r109.py','train_getup_joint_anchor_r102.py',
        'getup_reference_env_r100.py','getup_independent_native.py','validate_getup_fullpath_r27.py',
        'train_getup_fullpath_r27.py','probe_getup_anchor_r101.py','getup_fullfallen_env_r32.py',
        'getup_fullfallen_contract_r32.py','search_getup_reference_feedback_r64.py','search_getup_sustained_bridge_r42.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    write_json(args.output/'contract.json',dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        teacher_data_only=True,not_a_seed_indexed_deployed_policy=True,
        hypothesis='Case-specific complete-fall path optimization may supply missing teacher coverage after global feedback plateaus',
        missing_development_cases=missing,existing_teacher_coverage=19,standard_teacher=True,
        fixed_seed=209,seed_per_case='209+case seed',workers=args.workers,generations=args.generations,population=args.population,
        residual_knot_controls=NODES.tolist(),learned_interior_knots=6,learned_joints=10,normalized_bounds=[-1,1],
        combined_residual_cap_rad=.18,unchanged_home_hold=True,physics_hz=500,control_hz=50,
        short_training_tail_s=1,teacher_full_path_controls=2279,teacher_strict_tail_s=30,entry_deadline_s=12,
        physics_rewards_acceptance_unchanged=True,no_intermediate_root_reset=True,reference_states_never_injected=True,
        independent_qualification_not_run=True,independent_seeds_unused=True,smoke=args.smoke,
        hashes={str(f):digest(f) for f in [SCENE,STAND,REFERENCE,MODEL,prior_dir/'results.json',
            prior_dir/'frozen_profiles.npz',*sources.iterdir()]}))
    zero=np.zeros((6,10));teachers=[];history=[]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,
        initargs=(anchors,gains)) as pool:
        jobs=[(s,p,zero,True,str(args.output/'parity'/str(p)/f'case_{s}'),
            str(prior_dir/('development_baseline' if p==0 else 'development_r102')))
            for p in (0,1) for s in (None,769000)]
        parity=list(pool.map(rollout,jobs));write_json(args.output/'parity.json',parity)
        print('R109_ZERO_KNOT_FULL_PATH_PARITY_PASS',flush=True)
        if args.smoke:
            knots=zero.copy();knots[1,8]=.03
            smoke=list(pool.map(rollout,[(773004,0,knots,False,str(args.output/'nonzero_smoke'),None)]))
            write_json(args.output/'results.json',dict(smoke=True,rows=smoke,teacher_data_only=True,full_task_completed=False))
        else:
            jobs=[]
            for s in (None,*TRAIN):
                if s in missing:continue
                profile=0 if s in successes[0] else 1
                jobs.append((s,profile,zero,True,str(args.output/'existing_teachers'/f'case_{s}'),
                    str(prior_dir/('development_baseline' if profile==0 else 'development_r102'))))
            teachers=list(pool.map(rollout,jobs))
            assert all(t['qualified_teacher'] for t in teachers)
            state={s:dict(rng=np.random.default_rng(209+s),mean=zero.copy(),std=np.full((6,10),.10),
                best=None,profile=0,knots=zero.copy(),cache={},qualified=None) for s in missing}
            for gen in range(1,args.generations+1):
                active=[s for s in missing if state[s]['qualified'] is None]
                if not active:break
                jobs=[];candidates={}
                for s in active:
                    st=state[s]
                    proposals=[(0,zero.copy()),(1,zero.copy()),(st['profile'],st['knots'].copy())]
                    while len(proposals)<args.population:
                        profile=int(st['rng'].integers(2))
                        center=st['knots'] if len(proposals)%3==0 else st['mean']
                        knots=np.clip(center+st['rng'].normal(0.,st['std']),-1.,1.)
                        proposals.append((profile,knots))
                    candidates[s]=proposals
                    for i,(profile,knots) in enumerate(proposals):jobs.append((s,profile,knots,False,None,None))
                reports=list(pool.map(rollout,jobs));cursor=0;generation=[]
                for s in active:
                    st=state[s];proposals=candidates[s];rows=reports[cursor:cursor+len(proposals)];cursor+=len(proposals)
                    order=sorted(range(len(rows)),key=lambda i:rank(rows[i]),reverse=True);winner=order[0]
                    improved=st['best'] is None or rank(rows[winner])>rank(st['best'])
                    if improved:st.update(best=rows[winner],profile=proposals[winner][0],knots=proposals[winner][1].copy())
                    elite=np.stack([proposals[i][1] for i in order[:max(3,args.population//4)]])
                    st['mean']=.6*st['mean']+.4*elite.mean(0);st['std']=np.clip(.6*st['std']+.4*elite.std(0),.025,.35)
                    if improved and st['best']['training_success']:
                        full_dir=args.output/'complete_teacher_checks'/f'case_{s}'/f'generation_{gen:04d}'
                        verified=list(pool.map(rollout,[(s,st['profile'],st['knots'],True,str(full_dir),None)]))[0]
                        if verified['qualified_teacher']:
                            st['qualified']=verified;teachers.append(verified)
                        print('R109_FULL_TEACHER_CHECK',s,gen,verified['success'],verified['valid'],flush=True)
                    case_dir=args.output/'checkpoints'/f'case_{s}';case_dir.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(case_dir/f'generation_{gen:04d}.npz',knots=st['knots'],profile=st['profile'],
                        mean=st['mean'],std=st['std'])
                    generation.append(dict(case_seed=s,candidates=[dict(profile=p,knots=k.tolist()) for p,k in proposals],
                        reports=rows,best=st['best'],best_profile=st['profile'],qualified=st['qualified'],
                        rng_state=st['rng'].bit_generator.state))
                    print('R109_CASE',gen,s,st['best']['training_success'],st['best']['valid'],
                        st['best']['return_sum'],'TEACHER',st['qualified'] is not None,flush=True)
                history.append(dict(generation=gen,cases=generation));write_json(args.output/'history.json',history)
                write_json(args.output/'progress.json',dict(completed_generations=gen,
                    missing_cases=missing,teachers=teachers,new_qualified_teachers=sum(state[s]['qualified'] is not None for s in missing),
                    teacher_data_only=True,unified_policy_success=False,full_task_completed=False))
            write_json(args.output/'results.json',dict(teachers=teachers,missing_cases=missing,
                uncovered=[s for s in missing if state[s]['qualified'] is None],completed_generations=len(history),
                teacher_data_only=True,unified_policy_success=False,independent_qualification_run=False,
                simulation_only=True,hardware_readiness=False,full_task_completed=False))
    print('R109_TERMINAL',flush=True)


if __name__=='__main__':main()
