"""Fresh finite model-free CACLA positive-TD self-action reinforcement; unchanged simulation envelope."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np
from diagnostics import train_getup_adaptive_sensor_mpc_r195 as utility
from diagnostics import total_target_cacla_kernel_r205 as kernel

prior=utility.prior; local=utility.local; selector=utility.selector; planner=utility.planner
ROOT=utility.ROOT; CASES=utility.CASES; MODEL=None; WORKER_SOURCE_DIRECTORY=None; WORKER_SOURCE_MANIFEST=None
OUTPUT=ROOT/'outputs/getup_total_target_cacla_r205_left_20261011'
SMOKE=ROOT/'outputs/getup_total_target_cacla_r205_smoke_20261011'
LAUNCH=ROOT/'outputs/getup_total_target_cacla_launcher_r205_20261011'
PREFLIGHT=ROOT/'outputs/getup_total_target_cacla_r205_preflight_20261011'
NUMERIC=ROOT/'outputs/getup_total_target_cacla_numeric_r205_20261011'
NUMERIC_SOURCE=ROOT/'tmp/getup_total_target_cacla_numeric_r205_20261011'
EXPECTED_HEAD='d485cfc1c8eb31a4762ad5b711f8ae4d6fde6866'
KERNEL_SHA='582a059b384c2610b35469aa997a04890e2f12e5eb7db5952efcb98fa4d1eed0'
SMOKE_SPEC=[('zero_parity','baseline',[None,769000,773004],True),
            ('initial_standard','deterministic',[None],False),
            ('initial_sample','sample',[769002,773004],False)]


def capture_sources(destination,existing=None):
    if existing is None:destination.mkdir()
    paths={Path(__file__).resolve(),Path(kernel.__file__).resolve()}
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).is_file():paths.add(Path(p).resolve())
    for name in ('test_getup_total_target_cacla_r205.py','launch_getup_total_target_cacla_r205.py'):
        paths.add(Path(__file__).with_name(name).resolve())
    manifest={} if existing is None else dict(existing)
    for path in sorted(paths):
        sha=prior.digest(path); name=hashlib.sha256(str(path).encode()).hexdigest()[:16]+'_'+path.name
        if str(path) in manifest:
            assert manifest[str(path)]['sha256']==sha==prior.digest(destination/manifest[str(path)]['copy'])
            continue
        assert not (destination/name).exists()
        shutil.copy2(path,destination/name); assert prior.digest(destination/name)==sha
        manifest[str(path)]=dict(sha256=sha,copy=name)
    return manifest


def init_worker(frozen):
    global MODEL,WORKER_SOURCE_DIRECTORY,WORKER_SOURCE_MANIFEST
    path=Path(frozen)
    # Direct frozen loading: never call R195 init_worker or append old evidence.
    with np.load(path/'snapshot.npz',allow_pickle=False) as z:local.WEIGHTS={k:z[k].copy() for k in z.files}
    with np.load(path/'initial_selector.npz',allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}
    with np.load(path/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:local.NOMINAL=z['observations'].copy()
    assert local.NOMINAL.shape==(2279,55) and 'feature_mode' in local.WEIGHTS
    WORKER_SOURCE_DIRECTORY=path.parent/f'worker_executed_sources_{os.getpid()}'
    WORKER_SOURCE_MANIFEST=capture_sources(WORKER_SOURCE_DIRECTORY)
    local.write_json(path.parent/f'worker_source_manifest_{os.getpid()}.json',WORKER_SOURCE_MANIFEST)


def evaluate(job):
    global WORKER_SOURCE_MANIFEST
    mode,case,directory,weightfile,sampling_seed,parity=job
    assert mode in ('baseline','deterministic','sample') and case in CASES
    with np.load(weightfile,allow_pickle=False) as z:weights={k:z[k].copy() for k in z.files}
    rng=np.random.default_rng(305000000+sampling_seed)
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    captured=dict(frames=[],times=[]); records=[]; sampling_records=[]; signals={}; last_inputs={}; env=sim=original=None
    prepare_module=local.prior.audit; old_prepare=prepare_module.record_preparation
    def recorder(s,p,observe=prepare_module.native_observation):
        return utility.record_preparation_partial(s,p,observe,captured)
    prepare_module.record_preparation=recorder
    keys=('local_hip_extra_rad','original_double_pre_slew_target_rad',
          'adjusted_double_pre_slew_target_rad','planned_before_integration_rad',
          'same_state_unperturbed_planned_rad','policy_features','policy_mean',
          'policy_noise','policy_unclipped_proposal',
          'policy_requested_action','policy_active','policy_value','policy_raw_mean')
    signals={k:[] for k in keys}; rewards=[]; maxima=[]
    try:
        env=prepare_module.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,kernel.SEED,True)
        env.reset(case);sim=env.sim
        fingerprint=prior.physics_hash(sim); full_fingerprint=utility.model_digest(sim.model)
        mapping=utility.kernel.Mapping(sim.model)
        ids=np.array([sim.model.actuator(n).id for n in local.JOINTS]); sensors=np.array([9,10,11])
        gains,choice,logits=selector.old.predict(MODEL,env.preparation_sensors[-1])
        profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
        original=sim.step_target; initial=env.observe().copy()
        def controlled(target):
            k=env.controls; observed=env.observe().copy()
            np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
            frozen=local.local_feedback(observed,local.NOMINAL[k],sensors,gains) if k<529 else np.zeros(3)
            fixed=local.merge_target(target,env.targets[k],frozen,ids,sim.lower,sim.upper)
            active=mode!='baseline' and 0<k<529
            x=np.zeros(101);mean=np.zeros(10);noise=np.zeros(10);proposal=np.zeros(10);requested=np.zeros(10)
            value=np.asarray(0.);raw_mean=np.zeros(10)
            if active:
                x=kernel.features(observed[:50],initial[:50],k)
                decision=kernel.request(weights,x,rng if mode=='sample' else None)
                mean=decision['mean'];noise=decision['noise'];proposal=decision['proposal'];requested=decision['boxrequest']
                value=kernel.forward(weights,x[None,:],'value')[0][0,0];raw_mean=decision['rawmean']
                sampling_records.append(dict(control=k,mode=mode,rng_before=decision['rng_before'],rng_after=decision['rng_after']))
            # Box request REPLACES old leg feedback; no second tanh or residual addition.
            adjusted=replace_legs(fixed,env.targets[k],requested,mapping.legs,sim.lower,sim.upper,active)
            np.testing.assert_array_equal(adjusted[mapping.head],fixed[mapping.head])
            base=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
            wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
            assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
            if not active:assert adjusted is fixed
            last_inputs.update(control=np.asarray(k),current_native55=observed,initial_native55=initial,previous_applied=sim.prev.copy(),original_reference=env.targets[k].copy(),post_IMU_right_boundary=fixed.copy(),requested_action=requested)
            maxima.append(float(np.abs(adjusted-env.targets[k]).max()))
            applied=original(adjusted);np.testing.assert_array_equal(applied,wanted);np.testing.assert_array_equal(sim.prev,wanted)
            for key,value in zip(keys,(frozen,fixed,adjusted,wanted,base,x,mean,noise,proposal,requested,active,value,raw_mean)):
                signals[key].append(np.asarray(value).copy())
            return applied
        sim.step_target=controlled
        while True:
            obs=env.observe(); action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False);rewards.append(float(step[1]))
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
        row=step[5]
        arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],preparation_sensors=env.preparation_sensors,construction_preparation_sensors=captured['completed_batches'][0][0],construction_preparation_times=captured['completed_batches'][0][1],original_step_reward=rewards,**signals)
        previous=selector.OUTPUT/'candidate'/f'case_{case}'
        oldrow=json.loads((previous/'result.json').read_text());assert row['initial_hash']==oldrow['initial_hash']
        with np.load(previous/'trajectory.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(arrays['applied'][0],z['applied'][0])
            if parity:
                equality={key:bool(np.array_equal(np.asarray(arrays[key]),z[key])) for key in z.files}
                assert all(equality.values()),equality
                fields=('pose','case_seed','initial_hash','actual_fallen_start','valid','training_success','controls','strict_tail_s','entry_time_s','return_sum','final','peaks','prefix_start','root_edits_after_initialization')
                field_equality={key:row[key]==oldrow[key] for key in fields}
                assert all(field_equality.values()),field_equality
                row['complete_R157_environment_field_parity']=field_equality
                row['complete_R157_parity']=equality
        assert abs(sum(rewards)-row['return_sum'])<1e-10
        row.update(policy_mode=mode,policy_model_sha256=prior.digest(weightfile),sampling_seed=sampling_seed,
            success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),
            recovery_leg_feedback_replaced=mode!='baseline',old_feedback_added_to_policy=False,
            standard_R157_parity_required=mode=='baseline',base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),
            maximum_combined_14_joint_correction_rad=max(maxima),physics_sha256=fingerprint,
            physics_unchanged=fingerprint==prior.physics_hash(sim),compiled_model_sha256=full_fingerprint,
            compiled_model_unchanged=full_fingerprint==utility.model_digest(sim.model),no_root_or_case_input_to_value_policy=True,root_edits_during_recovery=0,
            clipped_request_has_boundary_atoms_not_Gaussian_density=True,
            positive_TD_self_action_reinforcement_not_external_teacher_BC=True,bootstrapped_TD_not_convergence_or_improvement=True)
        assert row['physics_unchanged'] and row['compiled_model_unchanged']
        np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'rng.json',rng.bit_generator.state);local.write_json(dest/'policy_sampling_rng_records.json',sampling_records);local.write_json(dest/'result.json',row)
        runtime_sources=capture_sources(WORKER_SOURCE_DIRECTORY,WORKER_SOURCE_MANIFEST)
        WORKER_SOURCE_MANIFEST=runtime_sources
        local.write_json(dest/'runtime_source_manifest.json',runtime_sources)
        print('R205_PATH',mode,case,row['controls'],row['success'],row['valid'],flush=True)
        return row
    except Exception:
        np.savez_compressed(dest/'partial_trajectory.npz',observations=[r[0] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],original_step_reward=rewards,preparation_sensors=np.asarray(captured['frames']),preparation_times=np.asarray(captured['times']),completed_preparation_sensors=np.asarray([b[0] for b in captured.get('completed_batches',[])]),completed_preparation_times=np.asarray([b[1] for b in captured.get('completed_batches',[])]),**signals)
        np.savez_compressed(dest/'failed_causal_inputs.npz',**last_inputs)
        local.write_json(dest/'partial_sampling_rng_records.json',sampling_records)
        local.write_json(dest/'partial_rng.json',rng.bit_generator.state)
        local.write_json(dest/'failure.json',dict(controls_completed=len(records),terminal_result_saved=False,traceback=traceback.format_exc()))
        raise
    finally:
        prepare_module.record_preparation=old_prepare
        if sim is not None and original is not None:sim.step_target=original


def storage_gate(start=False):
    free=shutil.disk_usage(ROOT).free
    assert free>=(12 if start else 10)*1024**3,'Resource gate; no cleanup or bypass'
    files=set()
    for directory in (ROOT/'tmp',ROOT/'outputs',Path(__file__).parent):
        for p in directory.iterdir():
            if 'r205' in p.name.lower():
                if p.is_file():files.add(p)
                elif p.is_dir():files.update(q for q in p.rglob('*') if q.is_file())
    live=sum(p.stat().st_size for p in files)
    forecast=(640+640+128)*1024**2
    assert live<=640*1024**2 and live+forecast<=2*1024**3,'Fixed complete storage cap exhausted'
    return dict(live_bytes=live,future_archive_git_extra_bytes=forecast,
                complete_actual_plus_forecast_bytes=live+forecast,free_bytes=free,
                external_tmp_logs_sources_included=True,all_R205_paths_included=True,no_cleanup=True)


def group(pool,mode,cases,path,weightfile,seedbase,parity=False):
    storage_gate()
    report=local.aggregate(list(pool.map(evaluate,[(mode,c,str(path/f'case_{c}'),str(weightfile),seedbase+i,parity) for i,c in enumerate(cases)])))
    local.write_json(path/'results.json',report);storage_gate();return report


def smoke_groups(pool,output,initial):
    result={}
    for name,mode,cases,parity in SMOKE_SPEC:
        result[name]=group(pool,mode,cases,output/name,initial,3051000,parity)
    return result


def smoke_gate(reports):
    # None is only checked for groups that actually contain the standard case.
    for name,mode,cases,parity in SMOKE_SPEC:
        report=reports[name]
        if report['physical_failures'] or any(r['controls']!=2279 for r in report['rows']):return False
        if None in cases and not report['nominal_success']:return False
    return all(all(r['complete_R157_parity'].values()) and all(r['complete_R157_environment_field_parity'].values()) for r in reports['zero_parity']['rows'])


def compare_smoke(output):
    for name,mode,cases,parity in SMOKE_SPEC:
        for case in cases:
            a=output/name/f'case_{case}';b=SMOKE/name/f'case_{case}'
            with np.load(a/'trajectory.npz',allow_pickle=False) as za,np.load(b/'trajectory.npz',allow_pickle=False) as zb:
                assert za.files==zb.files
                for key in za.files:np.testing.assert_array_equal(za[key],zb[key],err_msg=key)
            assert json.loads((a/'result.json').read_text())==json.loads((b/'result.json').read_text())
            assert json.loads((a/'rng.json').read_text())==json.loads((b/'rng.json').read_text())
            assert (a/'policy_sampling_rng_records.json').read_bytes()==(b/'policy_sampling_rng_records.json').read_bytes()


def replace_legs(fixed,reference,request,legs,lower,upper,active):
    if not active:return fixed
    adjusted=fixed.copy()
    adjusted[legs]=np.clip(reference[legs]+.18*np.asarray(request),lower[legs],upper[legs])
    return adjusted


def episode_batch(observations,rewards,mask,requests,invalid):
    observations=np.asarray(observations);mask=np.asarray(mask)
    n=len(observations)
    if observations.shape!=(n,50) or mask.dtype!=np.bool_ or mask.shape!=(n,):
        raise ValueError('Exact native50 and boolean active mask required')
    expected=(np.arange(n)>0)&(np.arange(n)<529)
    if not np.array_equal(mask,expected):raise ValueError('Exact active controls required')
    phi=np.asarray([kernel.features(observations[c],observations[0],c) for c in range(n)])
    return kernel.episode_transitions(phi,requests,rewards,invalid)


def dataset(path,weights):
    episodes=[]
    for case in CASES:
        directory=path/f'case_{case}'
        row=json.loads((directory/'result.json').read_text())
        with np.load(directory/'trajectory.npz',allow_pickle=False) as z:
            e=episode_batch(z['observations'][:,:50],z['original_step_reward'],z['policy_active'],z['policy_requested_action'],not row['valid'])
            for j,c in enumerate(e['controls']):
                x=e['phi'][j];d=kernel.request(weights,x)
                np.testing.assert_array_equal(x,z['policy_features'][c])
                np.testing.assert_array_equal(d['mean'],z['policy_mean'][c])
                np.testing.assert_array_equal(d['rawmean'],z['policy_raw_mean'][c])
                np.testing.assert_array_equal(kernel.forward(weights,x[None,:],'value')[0][0,0],z['policy_value'][c])
                proposal=d['mean']+z['policy_noise'][c]
                np.testing.assert_array_equal(proposal,z['policy_unclipped_proposal'][c])
                np.testing.assert_array_equal(np.clip(proposal,-1,1),z['policy_requested_action'][c])
            episodes.append(e)
    arrays=kernel.merge_episodes(episodes)
    # train_round saves the sole complete replay; no duplicate large matrices.
    local.write_json(path/'replay_contract.json',dict(current_round_new_training_only=True,old_data_used=False,evaluation_data_used=False,
        original_home_tail_included=True,invalid_original_terminal=True,active_samples=len(arrays['phi']),full_reward_samples=len(arrays['full_rewards']),
        terminal_bootstrap_count=int((arrays['bootstrap']==0).sum()),active_backup_terminal_not_physics_relabel=True,reward_unchanged=True))
    return arrays

def save_checkpoint_sampling_records(output,stem,paths,learner_rng):
    entries={}
    for path in paths:
        if path.exists():
            entries[str(path)]=dict(sha256=prior.digest(path),actual_saved_state=json.loads(path.read_text()))
    local.write_json(output/f'{stem}_sampling_rng.json',dict(actual_learning_rng_state=learner_rng.bit_generator.state,
        actual_episode_sampling_rng_records=entries,no_synthetic_checkpoint_reused=True))

def select_checkpoint(candidates):
    return min(c[3] for c in candidates) if candidates else None


def eligible(report,baseline):
    oldrows={r['case_seed']:r for r in baseline['rows']}
    return bool(report['nominal_success'] and not report['physical_failures'] and
                report['successes']>baseline['successes'] and
                all(not oldrows[r['case_seed']]['success'] or r['success'] for r in report['rows']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    output=SMOKE if args.smoke else OUTPUT
    assert not output.exists();storage_gate(start=True)
    assert prior.digest(kernel.__file__)==KERNEL_SHA
    learner=kernel.Learner(kernel.initial_weights()) # Fresh production initialization, never synthetic checkpoint.
    learner_rng=learner.rng
    learner.state() # Plain SGD full counts/RNG; no Adam or target.
    if not args.smoke:
        completed=json.loads((LAUNCH/'smoke_natural_exit.json').read_text())
        assert completed['natural_exit'] and completed['exit_code']==0
        reg=json.loads((LAUNCH/'formal/regression_natural_exit.json').read_text())
        assert reg['natural_exit'] and reg['exit_code']==0
        assert reg['tests_run']==16 and reg['skipped']==reg['failures']==reg['errors']==0
        assert smoke_gate(json.loads((SMOKE/'results.json').read_text())['reports'])
    output.mkdir();sources=capture_sources(output/'executed_sources');assets=utility.model_files(local.prior.program.SCENE)
    plan=NUMERIC_SOURCE/'GETUP_TOTAL_TARGET_CACLA_PLAN_R205_20261011.json'
    assert json.loads(plan.read_text())['budgets']['dynamic_total_cap']==162
    shutil.copy2(plan,output/'design_plan.json');assert prior.digest(plan)==prior.digest(output/'design_plan.json')
    frozen=output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:
        shutil.copy2(source,frozen/name);assert prior.digest(source)==prior.digest(frozen/name)
    hashes={str(p):prior.digest(p) for p in [*frozen.iterdir(),local.prior.program.STAND,local.prior.program.REFERENCE]}
    initial=output/'initial.npz'
    learner.checkpoint(output/'initial')
    if not args.smoke:
        assert (output/'initial.json').read_bytes()==(SMOKE/'initial.json').read_bytes()
        with np.load(SMOKE/'initial.npz',allow_pickle=False) as z:
            for k,v in learner.weights.items():np.testing.assert_array_equal(z[k],v)
    with np.load(initial,allow_pickle=False) as z:
        for k,v in kernel.initial_weights().items():np.testing.assert_array_equal(z[k],v)
    local.write_json(output/'contract.json',dict(simulation_only=True,smoke=args.smoke,seed=305,workers=6,
        total_attempt_cap=162,smoke_cap=6,formal_cap=156,training_rounds=2,training_attempts=50,
        value_updates_per_round=64,total_value_update_cap=128,total_actor_update_cap=128,policy_features=101,actor_parameters=3962,value_parameters=3809,total_parameters=7771,
        learning_rate=kernel.LR,sampling_sigma=kernel.NOISE,plain_SGD_no_Adam_momentum_target=True,
        only_current_round_new_R205_training_used=True,strict_positive_TD_self_requested_action_gate=True,
        only_current_round_replay=True,no_learned_dynamics=True,
        own_action_regression_not_external_BC=True,clipped_request_not_Gaussian_density=True,
        original_reward_unchanged=True,complete_episode_controls=2279,
        policy_replaces_old_recovery_leg_feedback=True,head_and_home_original=True,control0_original=True,
        frozen_reference_cap_rad=.18,storage_budget_gib=2,reserve_gib=10,
        unseen_320_never_loaded=True,full_task_completed=False,hardware_readiness=False,
        sources=sources,model_files=assets,hashes=hashes))
    try:
        with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
            reports=smoke_groups(pool,output,initial)
            gate=smoke_gate(reports)
            local.write_json(output/'startup_closed.json',dict(reports=reports,smoke_gate_passed=gate,attempts=6,terminal_result_saved=True))
            if args.smoke:
                local.write_json(output/'results.json',dict(smoke=True,reports=reports,smoke_gate_passed=gate,actual_dynamic_attempts=6,terminal_result_saved=True,training_run=False,qualification_passed=False,full_task_completed=False,hardware_readiness=False));return
            assert gate;compare_smoke(output)
            baseline=group(pool,'baseline',CASES,output/'development_baseline',initial,3052000,True)
            assert baseline['successes']==18 and baseline['nominal_success'] and not baseline['physical_failures']
            history=[];candidates=[]
            learner.checkpoint(output/'checkpoint_0000')
            save_checkpoint_sampling_records(output,'checkpoint_0000',
                [p for name,_,cases,_ in SMOKE_SPEC for c in cases
                 for p in [output/name/f'case_{c}/rng.json']],learner_rng)
            for iteration in (1,2):
                before=output/f'checkpoint_{iteration-1:04d}.npz'
                trainpath=output/f'training_{iteration:04d}'
                train=group(pool,'sample',CASES,trainpath,before,3053000+iteration*100)
                before_policy={n:v.copy() for n,v in learner.weights.items()}
                arrays=dataset(trainpath,before_policy)
                storage_gate()
                evidence=learner.train_round(arrays,trainpath/'learning')
                storage_gate()
                learner.checkpoint(output/f'checkpoint_{iteration:04d}')
                save_checkpoint_sampling_records(output,f'checkpoint_{iteration:04d}',[trainpath/f'case_{c}/rng.json' for c in CASES],learner_rng)
                check=group(pool,'deterministic',CASES,output/f'check_{iteration:04d}',
                    output/f'checkpoint_{iteration:04d}.npz',3055000)
                good=eligible(check,baseline)
                history.append(dict(iteration=iteration,training=train,check=check,eligible=good,
                    value_updates_this_round=64,total_value_updates=learner.value_count,total_actor_updates=learner.actor_count,
                    active_samples=len(arrays['phi']),complete_reward_samples=len(arrays['full_rewards']),
                    max_saved_action_change=float(np.abs(kernel.forward(learner.weights,arrays['phi'],'actor')[0]-kernel.forward(before_policy,arrays['phi'],'actor')[0]).max()),
                    improvement_or_safety_not_implied_by_TD_update=True))
                if good:candidates.append((check['successes'],min(r['return_sum'] for r in check['rows']),
                    sum(r['return_sum'] for r in check['rows']),iteration))
                local.write_json(output/'training_closed_history.json',history)
                print('R205_UPDATE_CLOSED',iteration,check['successes'],good,flush=True)
            selected=select_checkpoint(candidates)
            model=initial if selected is None else output/f'checkpoint_{selected:04d}.npz'
            mode='baseline' if selected is None else 'deterministic'
            candidate=group(pool,mode,CASES,output/'candidate',model,3055000,selected is None)
            assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
            gate=bool(candidate['nominal_success'] and not candidate['physical_failures'] and candidate['successes']>=22 and candidate['successes']>18)
            local.write_json(output/'results.json',dict(terminal_result_saved=True,actual_dynamic_attempts=156,baseline=baseline,candidate=candidate,selected_checkpoint=selected,selected_R157_fallback=selected is None,history=history,original_development_gate=gate,expanded_development_run=False,independent_qualification_run=False,qualification_passed=False,full_task_completed=False,hardware_readiness=False))
    except Exception:
        local.write_json(output/'failure.json',dict(traceback=traceback.format_exc(),terminal_result_saved=False,full_task_completed=False,hardware_readiness=False));raise
    finally:
        assert assets==utility.model_files(local.prior.program.SCENE)
        assert all(prior.digest(Path(p))==v['sha256'] for p,v in sources.items())
        assert all(prior.digest(Path(p))==sha for p,sha in hashes.items())
        runtime_sources=capture_sources(output/'executed_sources',sources)
        local.write_json(output/'runtime_source_manifest.json',runtime_sources)
        local.write_json(output/'input_hashes_after.json',dict(source_and_model_files_unchanged=True,hashes=hashes,model_files=assets))


if __name__=='__main__':main()

