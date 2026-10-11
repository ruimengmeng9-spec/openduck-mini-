"""Independent saved R203 decisions, no environment, production kernel decisions, or learning updates."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import traceback
import numpy as np
import mujoco
from diagnostics import train_getup_total_target_lstdq_r203b as run

ROOT=Path('/data/shijinsheng/open_duck')
REPO=ROOT/'github/openduck-mini-'
BASE='8f19fcfb0ac6922818a4079f058d23cecd565e5c'
DEST=ROOT/'outputs/getup_total_target_lstdq_r203_terminal_verification_20261011'
PROOF=ROOT/'tmp/getup_total_target_lstdq_r203_terminal_verification_20261011.json'
A_PRE=ROOT/'outputs/getup_total_target_lstdq_r203_preflight_20261011'
A_LAUNCH=ROOT/'outputs/getup_total_target_lstdq_launcher_r203_20261011'
DIRECTORIES=(run.NUMERIC,run.NUMERIC_SOURCE,A_PRE,run.PREFLIGHT,A_LAUNCH,run.LAUNCH,run.SMOKE,run.OUTPUT)
CASES=[None,*range(769000,769008),*range(773000,773016)]
SPEC=[('zero_parity',[None,769000,773004]),('initial_standard',[None]),('initial_sample',[769002,773004])]
MJB='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
PHYSICS='4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae'
GAMMA=.99**.2
RIDGE=.001

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def eq(a,b,label=''):np.testing.assert_array_equal(a,b,err_msg=label)
def ah(a):
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def inventory(p):return {str(f.relative_to(p)):digest(f) for f in sorted(p.rglob('*')) if f.is_file()}
def sources(m,folder):
    for p,v in m.items():assert digest(p)==v['sha256']==digest(folder/v['copy']),p
    return len(m)
def idle():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
        pid,args=line.strip().split(None,1)
        if int(pid)==os.getpid():continue
        first=Path(args.split()[0]).name
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in args and any(t in args for t in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_','verify_getup_'))),args
        assert 'publish_getup' not in args and 'git push' not in args,args
def resource(additional=0):
    files=set()
    for directory in (ROOT/'tmp',ROOT/'outputs',Path(run.__file__).parent):
        for item in directory.iterdir():
            if 'r203' in item.name.lower():
                if item.is_file():files.add(item)
                elif item.is_dir():files.update(f for f in item.rglob('*') if f.is_file())
    live=sum(p.stat().st_size for p in files);forecast=(640+640+128)*1024**2
    assert live+forecast+additional<=2*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
    return dict(live_bytes=live,future_archive_git_extra_bytes=forecast,additional_bound_bytes=additional,
        live_plus_forecast_and_additional_bytes=live+forecast+additional,free_bytes=shutil.disk_usage(ROOT).free)
def feature(o,initial,k):return np.r_[np.tanh(o[:50].astype(float)),np.tanh(initial[:50].astype(float)),min(k/528.,1.),1.]
def greedy(theta,x):return np.clip(x@theta[102:].reshape(102,10),-1.,1.)
def state(theta,count):return b'R203LSTDQv1\0'+struct.pack('<Q',count)+theta.astype('<f8').tobytes()
def returns(r):
    result=np.empty_like(r,dtype=float);tail=0.
    for i in range(len(r)-1,-1,-1):tail=float(r[i])+GAMMA*tail;result[i]=tail
    return result
def basis(x,a):return np.concatenate((x,(x[:,:,None]*a[:,None,:]).reshape(len(x),-1)),axis=1)
def checkpoint(theta,count):
    stem=run.OUTPUT/f'checkpoint_{count:04d}'
    w=load(str(stem)+'.npz');assert set(w)=={'theta','solve_count'}
    eq(w['theta'],theta);assert int(w['solve_count'])==count
    raw=state(theta,count);assert raw==Path(str(stem)+'.state').read_bytes()
    saved=read(str(stem)+'_sampling_rng.json')
    assert not saved['learner_rng_exists'] and not saved['learner_rng_consumed'] and saved['no_synthetic_checkpoint_reused']
    paths=[run.OUTPUT/name/f'case_{c}/rng.json' for name,cases in SPEC for c in cases] if count==0 else [run.OUTPUT/f'training_{count:04d}/case_{c}/rng.json' for c in CASES]
    expected={str(p):dict(sha256=digest(p),actual_saved_state=read(p)) for p in paths}
    assert expected==saved['actual_episode_sampling_rng_records']
    return dict(solve_count=count,theta=ah(theta),state_bytes_sha256=hashlib.sha256(raw).hexdigest(),full_state_exact=True,actual_sampling_rng_records_exact=True,learner_RNG_absent=True)
def report(folder,cases):
    saved=read(folder/'results.json');rows=[read(folder/f'case_{c}/result.json') for c in cases]
    assert rows==saved['rows']
    expected=dict(successes=sum(bool(r['success']) for r in rows if r['case_seed'] is not None),
        nominal_success=next((bool(r['success']) for r in rows if r['case_seed'] is None),None),
        physical_failures=sum(not r['valid'] for r in rows),return_sum=sum(r['return_sum'] for r in rows))
    for k,v in expected.items():assert saved[k]==v,k
    return saved
def good(report,base):
    old={r['case_seed']:r for r in base['rows']}
    return bool(report['nominal_success'] and report['physical_failures']==0 and report['successes']>base['successes'] and all(not old[r['case_seed']]['success'] or r['success'] for r in report['rows']))

def learn(iteration,before):
    path=run.OUTPUT/f'training_{iteration:04d}';episodes=[]
    assert (path/'learner_state_before_solve.state').read_bytes()==state(before,iteration-1)
    eq(load(path/'frozen_before_solve.npz')['theta'],before)
    for case in CASES:
        with np.load(path/f'case_{case}/trajectory.npz',allow_pickle=False) as z:
            obs=z['observations'][:,:50];n=len(obs);r=z['original_step_reward'].astype(float)
            mask=(np.arange(n)>0)&(np.arange(n)<529);eq(mask,z['policy_active'])
            ids=np.flatnonzero(mask);x=np.stack([feature(obs[i],obs[0],int(i)) for i in ids])
            nx=np.zeros_like(x);boot=np.zeros(len(ids));backup=r[ids].copy();ret=returns(r)
            for j,i in enumerate(ids):
                eq(x[j],z['policy_features'][i])
                eq(greedy(before,x[j]),z['policy_mean'][i])
                if i+1<n and mask[i+1]:nx[j]=feature(obs[i+1],obs[0],int(i+1));boot[j]=1.
                elif i+1<n:backup[j]+=GAMMA*ret[i+1]
            episodes.append(dict(phi=x,next_phi=nx,action=z['policy_requested_action'][ids].copy(),
                reward_backup=backup,bootstrap=boot,control=ids,original_reward=r.copy(),original_full_returns=ret))
    combined={k:np.concatenate([e[k] for e in episodes]) for k in episodes[0]}
    combined['episode_offsets']=np.r_[0,np.cumsum([len(e['phi']) for e in episodes])]
    combined['original_episode_offsets']=np.r_[0,np.cumsum([len(e['original_reward']) for e in episodes])]
    captured=load(path/'complete_new_original_transitions.npz');assert captured.keys()==combined.keys()
    for k,v in combined.items():eq(v,captured[k],k)
    contract=read(path/'transition_contract.json')
    assert contract['current_round_new_training_only'] and not contract['old_data_used'] and not contract['evaluation_data_used']
    assert contract['original_home_tail_included'] and contract['invalid_original_terminal'] and contract['fixed_Q_offset_not_modified_reward']
    assert contract['active_samples']==len(combined['phi']) and contract['full_reward_samples']==len(combined['original_reward'])
    assert contract['terminal_bootstrap_count']==int((combined['bootstrap']==0).sum())==25
    x=combined['phi'];a=combined['action'];nx=combined['next_phi'];r=combined['reward_backup'];boot=combined['bootstrap'];n=len(x)
    nxt=greedy(before,nx);nxt[boot==0]=0.
    A=np.zeros((1122,1122));b=np.zeros(1122)
    # Independent equations; same documented block256 accumulation order for exact original-byte comparisons.
    for start in range(0,n,256):
        sl=slice(start,min(start+256,n));psi=basis(x[sl],a[sl]);nextpsi=basis(nx[sl],nxt[sl])
        delta=psi-GAMMA*boot[sl,None]*nextpsi
        offset_next=-.5*np.sum(nxt[sl]**2,axis=-1);offset_now=-.5*np.sum(a[sl]**2,axis=-1)
        target=r[sl]+GAMMA*boot[sl]*offset_next-offset_now
        A+=psi.T@delta;b+=psi.T@target
    A/=n;b/=n;A.flat[::1123]+=RIDGE
    original=load(path/'original_pre_solve_system.npz')
    for k,v in [('matrix',A),('rhs',b),('next_action',nxt)]:eq(v,original[k],k)
    theta=np.linalg.solve(A,b);residual=A@theta-b
    scale=max(1.,float(np.linalg.norm(A,np.inf)*np.linalg.norm(theta,np.inf)),float(np.linalg.norm(b,np.inf)))
    relative=float(np.linalg.norm(residual,np.inf)/scale);assert np.isfinite(theta).all() and relative<=1e-10
    decisions=dict(theta=theta,matrix=A,rhs=b,residual=residual,next_action=nxt,
        action_before=greedy(before,x),action_after=greedy(theta,x))
    saved=load(path/'full_lstdq_decisions.npz');assert saved.keys()==decisions.keys()
    for k,v in decisions.items():eq(v,saved[k],k)
    update=read(path/'lstdq_update.json')
    assert update['solve_count']==iteration and update['relative_residual']==relative and update['fixed_ridge']==.001 and update['fixed_action_curvature']==1.
    assert update['nonsymmetric_projected_Bellman_equation'] and update['no_actor_or_optimizer'] and update['fixed_Q_offset_not_modified_reward']
    result=dict(iteration=iteration,active_samples=n,original_full_reward_samples=len(combined['original_reward']),
        terminal_bootstrap_count=25,full_transition_arrays_independently_exact={k:ah(v) for k,v in combined.items()},
        full_pre_solve_A_b_next_greedy_exact=True,full_saved_decisions_independently_exact={k:ah(v) for k,v in decisions.items()},
        relative_residual=relative,max_saved_action_change=float(np.abs(decisions['action_after']-decisions['action_before']).max()),
        checkpoint=checkpoint(theta,iteration),current_round_only=True,original_home_discount_tail_and_invalid_terminal_exact=True,
        fixed_quadratic_offset_both_ends_not_reward_change=True,production_decision_functions_invoked=False,
        same_numpy_linear_algebra_not_cross_implementation_bitwise_guarantee=True)
    print('R203_LSTDQ_SAVED_SOLVE_VERIFIED',iteration,n,relative,flush=True)
    return theta,result

def verify():
    before={str(d):inventory(d) for d in DIRECTORIES};manifest_counts=[]
    ne=read(ROOT/'tmp/getup_total_target_lstdq_r203_numeric_exit_20261011.json')
    assert ne['natural_exit'] and ne['exit_code']==0 and ne['source_hashes_unchanged']
    nr=read(run.NUMERIC/'result.json');assert nr['passed'] and nr['tests_run']==24 and nr['skipped']==nr['failures']==nr['errors']==0
    manifest_counts.append([str(run.NUMERIC),sources(read(run.NUMERIC/'source_manifest.json'),run.NUMERIC/'executed_sources')])
    for p,sha in ne['source_hashes'].items():assert digest(run.NUMERIC_SOURCE/p)==sha
    for folder in (A_LAUNCH,run.LAUNCH,run.SMOKE,run.OUTPUT):
        c=read(folder/'contract.json');manifest_counts.append([str(folder),sources(c['sources'],folder/'executed_sources')])
        if folder in (run.SMOKE,run.OUTPUT):
            for p,sha in c['model_files'].items():assert digest(p)==sha
            for p,sha in c['hashes'].items():assert digest(p)==sha
            assert read(folder/'input_hashes_after.json')['source_and_model_files_unchanged']
        for f in folder.glob('worker_source_manifest_*.json'):
            dst=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''))
            manifest_counts.append([str(f),sources(read(f),dst)])
    for folder in (A_PRE,run.PREFLIGHT,A_LAUNCH/'initial',run.LAUNCH/'initial',run.LAUNCH/'formal'):
        c=read(folder/'interface_result.json')
        assert c['tests_run']==16 and c['skipped']==c['errors']==c['failures']==0 and c['source_hashes_unchanged']
        log=(folder/'regression.log').read_text();assert 'Ran 16 tests' in log and '\nOK\n' in log and 'skipped' not in log
        manifest_counts.append([str(folder),sources(read(folder/'source_manifest.json'),folder/'executed_sources')])
        for f in (folder/'worker_fixture').glob('worker_source_manifest_*.json'):
            dst=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''))
            manifest_counts.append([str(f),sources(read(f),dst)])
    for p in (run.LAUNCH/'initial/regression_natural_exit.json',run.LAUNCH/'formal/regression_natural_exit.json'):
        c=read(p);assert c['natural_exit'] and c['exit_code']==0 and c['tests_run']==16 and c['skipped']==c['errors']==c['failures']==0
    for p in (run.LAUNCH/'smoke_natural_exit.json',run.LAUNCH/'formal_natural_exit.json'):
        c=read(p);assert c['natural_exit'] and c['exit_code']==0 and c['actual_subprocess_return_observed']
    launcher_exit=read(ROOT/'tmp/getup_total_target_lstdq_r203b_launcher_exit_20261011.json')
    assert launcher_exit['natural_exit'] and launcher_exit['exit_code']==0 and launcher_exit['actual_launcher_subprocess_return_observed'] and launcher_exit['source_hashes_unchanged']
    for p,sha in launcher_exit['source_hashes'].items():assert digest(p)==sha
    assert not (run.OUTPUT/'failure.json').exists() and not (run.LAUNCH/'failure.json').exists()
    old_exit=read(ROOT/'tmp/getup_total_target_lstdq_r203_launcher_exit_20261011.json')
    assert old_exit['natural_exit'] and old_exit['exit_code']==1 and (A_LAUNCH/'failure.json').exists()
    assert not (A_LAUNCH/'initial/regression_natural_exit.json').exists()
    assert not (ROOT/'tmp/getup_total_target_lstdq_r203_launch_request_exit_20261011.json').exists()
    weights=[np.zeros(1122)];initial=checkpoint(weights[0],0)
    for p in (run.SMOKE/'initial.npz',run.OUTPUT/'initial.npz'):eq(load(p)['theta'],weights[0])
    learning=[]
    for iteration in (1,2):
        theta,evidence=learn(iteration,weights[-1]);weights.append(theta);learning.append(evidence)
    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    size=mujoco.mj_sizeModel(model);buffer=np.empty(size,dtype=np.uint8);mujoco.mj_saveModel(model,buffer=buffer)
    assert hashlib.sha256(buffer.tobytes()).hexdigest()==MJB
    legs=np.array([model.actuator(s+'_'+part).id for s in ('left','right') for part in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
    head=np.array([i for i in range(14) if i not in legs]);joints=model.actuator_trnid[:,0]
    lower,upper=model.jnt_range[joints].T.copy();home=model.keyframe('home').qpos[model.jnt_qposadr[joints]]
    with np.load(run.local.prior.program.REFERENCE,allow_pickle=False) as z:reference=z['targets'].copy()
    selector=load(run.SMOKE/'frozen/initial_selector.npz')
    with np.load(run.SMOKE/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    result=read(run.OUTPUT/'results.json');assert result['selected_checkpoint'] is None and result['selected_R157_fallback']
    count=controls=baseline_count=0;rows=[]
    for folder in (run.SMOKE,run.OUTPUT):
        for path in sorted(folder.glob('*/case_*')):
            row=read(path/'result.json');mode=row['policy_mode'];n=row['controls'];count+=1;controls+=n
            group=path.parent.name
            wi=1 if group in ('training_0002','check_0001') else 2 if group=='check_0002' else 0
            theta=weights[wi]
            weightfile=run.OUTPUT/f'checkpoint_{wi:04d}.npz' if group in ('training_0001','training_0002','check_0001','check_0002') else folder/'initial.npz'
            assert digest(weightfile)==row['policy_model_sha256']
            assert row['physics_unchanged'] and row['compiled_model_unchanged'] and row['physics_sha256']==PHYSICS and row['compiled_model_sha256']==MJB
            rng=np.random.default_rng(row['sampling_seed'])
            with np.load(path/'trajectory.npz',allow_pickle=False) as archive:
                z={k:archive[k] for k in archive.files if k not in ('qpos','qvel')};obs=z['observations'];assert len(obs)==n
                assert z['construction_preparation_sensors'].shape==z['preparation_sensors'].shape==(40,50)
                assert z['construction_preparation_times'].shape==(40,)
                eq(obs[1:,34:48],(z['applied'][:-1]-home).astype(np.float32))
                sensor=z['preparation_sensors'][-1].astype(float)
                logits=(sensor-selector['mean'])/selector['std']@selector['weight']+selector['bias']
                choice=int(np.argmax(logits));gains=selector['global_gains'][choice].copy()
                assert logits.tolist()==row['initial_sensor_logits'] and choice==row['base_choice'] and gains.tolist()==row['base_gains']
                for k in range(n):
                    ids=np.array([9,10,11]);current=obs[k].astype(np.float32);nom=nominal[k].astype(np.float32)
                    position=(current[6+ids]-nom[6+ids]).astype(float)/.05;velocity=(current[20+ids]-nom[20+ids]).astype(float)/.05
                    right=-.18*np.tanh(gains[:3]*position+gains[3:]*velocity) if k<529 else np.zeros(3)
                    eq(right,z['local_hip_extra_rad'][k],'right_feedback')
                    active=mode!='baseline' and 0<k<529;assert bool(z['policy_active'][k])==active
                    x=np.zeros(102);mean=np.zeros(10);noise=np.zeros(10);proposal=np.zeros(10);requested=np.zeros(10)
                    if active:
                        x=feature(obs[k],obs[0],k);mean=greedy(theta,x)
                        if mode=='sample':noise=rng.normal(size=10);proposal=mean+1e-4*noise;requested=np.clip(proposal,-1.,1.)
                        else:proposal=mean.copy();requested=mean.copy()
                    for key,val in [('policy_features',x),('policy_mean',mean),('policy_noise',noise),('policy_unclipped_proposal',proposal),('policy_requested_action',requested)]:eq(val,z[key][k],key)
                    fixed=z['original_double_pre_slew_target_rad'][k];adjusted=fixed.copy()
                    if active:adjusted[legs]=np.clip(reference[k,legs]+.18*requested,lower[legs],upper[legs])
                    eq(adjusted,z['adjusted_double_pre_slew_target_rad'][k]);eq(adjusted[head],fixed[head])
                    if k:
                        prev=z['applied'][k-1]
                        planned=np.clip(np.clip(adjusted,lower,upper),prev-5.24*.02,prev+5.24*.02)
                        base=np.clip(np.clip(fixed,lower,upper),prev-5.24*.02,prev+5.24*.02)
                        eq(planned,z['planned_before_integration_rad'][k]);eq(base,z['same_state_unperturbed_planned_rad'][k])
                eq(z['planned_before_integration_rad'],z['applied'])
                assert float(np.abs(z['adjusted_double_pre_slew_target_rad']-reference[:n]).max())==row['maximum_combined_14_joint_correction_rad']
                assert abs(sum(z['original_step_reward'])-row['return_sum'])<1e-10
                assert row['success']==bool(row['valid'] and n==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8)
                oldpath=run.selector.OUTPUT/'candidate'/f"case_{row['case_seed']}";oldrow=read(oldpath/'result.json')
                assert row['initial_hash']==oldrow['initial_hash']
                with np.load(oldpath/'trajectory.npz',allow_pickle=False) as old:
                    eq(z['applied'][0],old['applied'][0])
                    if mode=='baseline':
                        baseline_count+=1
                        for key in old.files:eq(archive[key] if key in ('qpos','qvel') else z[key],old[key],key)
                        for key in oldrow:
                            if key in row:assert row[key]==oldrow[key],key
            assert rng.bit_generator.state==read(path/'rng.json')
            rows.append(dict(group=folder.name+'/'+group,case=row['case_seed'],mode=mode,controls=n,success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],original_peaks=row['peaks'],combined_max=row['maximum_combined_14_joint_correction_rad']))
            print('R203_SAVED_PATH_VERIFIED',count,group,row['case_seed'],n,flush=True)
    assert count==162 and controls==365248 and baseline_count==56
    for folder in (run.SMOKE,run.OUTPUT):
        startup=read(folder/'startup_closed.json');assert startup['smoke_gate_passed'] and startup['attempts']==6
        for name,cases in SPEC:assert report(folder/name,cases)==startup['reports'][name]
    for name,cases in SPEC:
        for case in cases:
            a=run.OUTPUT/name/f'case_{case}';b=run.SMOKE/name/f'case_{case}'
            with np.load(a/'trajectory.npz',allow_pickle=False) as za,np.load(b/'trajectory.npz',allow_pickle=False) as zb:
                assert za.files==zb.files
                for key in za.files:eq(za[key],zb[key],key) # isolated full-field equality, never control inference
            assert read(a/'result.json')==read(b/'result.json') and read(a/'rng.json')==read(b/'rng.json')
    baseline=report(run.OUTPUT/'development_baseline',CASES);candidate=report(run.OUTPUT/'candidate',CASES)
    assert result['baseline']==baseline and result['candidate']==candidate
    assert baseline['successes']==candidate['successes']==18 and baseline['nominal_success'] and candidate['nominal_success'] and baseline['physical_failures']==candidate['physical_failures']==0
    for c in CASES:
        a=run.OUTPUT/'development_baseline'/f'case_{c}';b=run.OUTPUT/'candidate'/f'case_{c}'
        with np.load(a/'trajectory.npz',allow_pickle=False) as za,np.load(b/'trajectory.npz',allow_pickle=False) as zb:
            assert za.files==zb.files
            for key in za.files:eq(za[key],zb[key],key)
        ra=read(a/'result.json');rb=read(b/'result.json')
        for key in ra:
            if key!='sampling_seed':assert ra[key]==rb[key],key
    history=read(run.OUTPUT/'training_closed_history.json');assert history==result['history'] and len(history)==2
    eligible=[]
    for i,h in enumerate(history,1):
        train=report(run.OUTPUT/f'training_{i:04d}',CASES);check=report(run.OUTPUT/f'check_{i:04d}',CASES)
        assert train==h['training'] and check==h['check']
        assert h['eligible']==good(check,baseline)
        assert not h['eligible'] and check['successes']==4 and not check['nominal_success']
        assert train['successes']==(8 if i==1 else 7) and train['nominal_success']==(i==1) and train['physical_failures']==(0 if i==1 else 1)
        assert check['physical_failures']==(1 if i==1 else 0)
        assert h['solve_count']==i and h['active_samples']==learning[i-1]['active_samples'] and h['complete_reward_samples']==learning[i-1]['original_full_reward_samples']
        assert h['relative_residual']==learning[i-1]['relative_residual'] and h['max_saved_action_change']==learning[i-1]['max_saved_action_change']
        eligible.append(h['eligible'])
    assert not result['original_development_gate'] and result['actual_dynamic_attempts']==156
    for k in ('expanded_development_run','independent_qualification_run','qualification_passed','full_task_completed','hardware_readiness'):assert not result[k]
    assert before=={str(d):inventory(d) for d in DIRECTORIES}
    idle();resources=resource(8*1024**2)
    return dict(audit_complete=True,new_dynamic_attempts=0,verifier_source_sha256=digest(__file__),
        closed_saved_paths=count,controls=controls,baseline_paths_all_original_fields_exact=baseline_count,
        two_complete_current_round_LSTDQ_solves_independently_recomputed=learning,fresh_zero_initial_checkpoint=initial,
        all_causal102_greedy_iid_noise_unclipped_proposal_boxrequest_sampling_RNG_exact=True,det_no_RNG_consumption=True,
        selector_right_feedback_independently_exact=True,post_IMU_right_double_boundary_ten_leg_replacement_head_exact=True,
        planning_after_first_recursive_joint_slew_previous_exact=True,first_applied_R157_boundary_exact=True,all_planned_equals_applied=True,
        constructor_and_reset_each40x50=True,formal_independent_startup_full_arrays_results_rng_exact=True,
        final_fallback_full_25_arrays_old_fields_peaks_exact=True,baseline_successes=18,candidate_R157_fallback_successes=18,
        deterministic_check_successes=[4,4],deterministic_standard_successes=[False,False],common_eligible=eligible,
        all_original_invalid_labels_peaks_retained=True,formal_and_launcher_actual_natural_exit_zero=True,
        original_launcher_a_recording_TypeError_and_request_AttributeError_preserved=True,no_original_request_exit_receipt_fabricated=True,
        source_manifest_counts=manifest_counts,source_inventory=before,source_evidence_hashes_unchanged=True,
        full_model_MJB_sha256=MJB,original_physics_sha256=PHYSICS,
        original_before_request_full_IMU_double_reconstructed=False,first_joint_slew_planning_independently_reconstructed=False,
        environment_MjData_forward_integrate_contact_force_relabel_invoked=False,production_kernel_decision_or_update_functions_invoked=False,
        same_numpy_not_cross_implementation_bitwise_or_physical_safety_guarantee=True,
        all_existing_original_arrays_present_not_rewritten=True,independent_recomputed_full_arrays_compared_and_hashes_saved_without_duplicate_large_NPZ=True,
        dynamic_budget162_closed=True,additional_dynamic_budget_allowed=False,rows=rows,resources=resources,
        archive_publisher_not_yet_implemented=True,best_remains_R157_18_of_24=True,
        full_task_completed=False,hardware_readiness=False,qualification_passed=False)

def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6]);idle()
    assert not DEST.exists() and not PROOF.exists()
    resource(32*1024**2)
    DEST.mkdir();copydir=DEST/'executed_sources';copydir.mkdir();manifest={}
    paths={Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).is_file():paths.add(Path(p).resolve())
    for p in sorted(paths):
        name=hashlib.sha256(str(p).encode()).hexdigest()[:16]+'_'+p.name
        shutil.copy2(p,copydir/name);manifest[str(p)]=dict(sha256=digest(p),copy=name)
    with (DEST/'source_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    try:
        result=verify();sources(manifest,copydir)
        with PROOF.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        resource()
        print('R203_TERMINAL_VERIFICATION_COMPLETE',result['closed_saved_paths'],result['controls'],digest(PROOF),flush=True)
    except Exception:
        with (DEST/'failure.json').open('x') as f:json.dump(dict(traceback=traceback.format_exc(),new_dynamic_attempts=0,proof_saved=PROOF.exists()),f,indent=2)
        raise
if __name__=='__main__':main()

