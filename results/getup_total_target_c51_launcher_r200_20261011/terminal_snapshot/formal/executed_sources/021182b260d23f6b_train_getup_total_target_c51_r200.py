"""Fresh discrete categorical value control; unchanged simulation envelope and reward."""
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
from diagnostics import total_target_c51_kernel_r200 as kernel

prior=utility.prior; local=utility.local; selector=utility.selector; planner=utility.planner
ROOT=utility.ROOT; CASES=utility.CASES; MODEL=None
OUTPUT=ROOT/'outputs/getup_total_target_c51_r200_left_20261011'
SMOKE=ROOT/'outputs/getup_total_target_c51_r200_smoke_20261011'
LAUNCH=ROOT/'outputs/getup_total_target_c51_launcher_r200_20261011'
PREFLIGHT=ROOT/'outputs/getup_total_target_c51_r200_preflight_20261011'
SMOKE_SPEC=[('zero_parity','baseline',[None,769000,773004],True),
            ('initial_standard','deterministic',[None],False),
            ('initial_sample','sample',[769002,773004],False)]


def capture_sources(destination):
    destination.mkdir()
    paths={Path(__file__).resolve(),Path(kernel.__file__).resolve()}
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).is_file():paths.add(Path(p).resolve())
    for name in ('test_getup_total_target_c51_r200.py','launch_getup_total_target_c51_r200.py'):
        paths.add(Path(__file__).with_name(name).resolve())
    manifest={}
    for path in sorted(paths):
        sha=prior.digest(path); name=hashlib.sha256(str(path).encode()).hexdigest()[:16]+'_'+path.name
        shutil.copy2(path,destination/name); assert prior.digest(destination/name)==sha
        manifest[str(path)]=dict(sha256=sha,copy=name)
    return manifest


def init_worker(frozen):
    global MODEL
    path=Path(frozen)
    # Direct frozen loading: never call R195 init_worker or append old evidence.
    with np.load(path/'snapshot.npz',allow_pickle=False) as z:local.WEIGHTS={k:z[k].copy() for k in z.files}
    with np.load(path/'initial_selector.npz',allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}
    with np.load(path/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:local.NOMINAL=z['observations'].copy()
    assert local.NOMINAL.shape==(2279,55) and 'feature_mode' in local.WEIGHTS
    manifest=capture_sources(path.parent/f'worker_executed_sources_{os.getpid()}')
    local.write_json(path.parent/f'worker_source_manifest_{os.getpid()}.json',manifest)


def evaluate(job):
    mode,case,directory,weightfile,sampling_seed,parity=job
    assert mode in ('baseline','deterministic','sample') and case in CASES
    with np.load(weightfile,allow_pickle=False) as z:weights={k:z[k].copy() for k in z.files}
    rng=np.random.default_rng(sampling_seed)
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    captured=dict(frames=[],times=[]); records=[]; signals={}; last_inputs={}; env=sim=original=None
    prepare_module=local.prior.audit; old_prepare=prepare_module.record_preparation
    def recorder(s,p,observe=prepare_module.native_observation):
        return utility.record_preparation_partial(s,p,observe,captured)
    prepare_module.record_preparation=recorder
    keys=('local_hip_extra_rad','original_double_pre_slew_target_rad',
          'adjusted_double_pre_slew_target_rad','planned_before_integration_rad',
          'same_state_unperturbed_planned_rad','policy_requested_action','policy_active',
          'policy_code','policy_greedy_code','policy_expected_values','policy_selected_probabilities',
          'policy_exploration_gate','policy_exploration_code')
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
            requested=np.zeros(10);chosen=greedy=draw=0;gate_draw=0.
            expected_values=np.zeros(kernel.ACTIONS);selected_probabilities=np.zeros(kernel.ATOMS)
            if active:
                x=kernel.features(observed[:50],initial[:50],k)
                decision=kernel.select(weights,x,rng,mode=='sample')
                requested=decision['action'];chosen=decision['chosen'];greedy=decision['greedy']
                expected_values=decision['expected_values'];selected_probabilities=decision['selected_probabilities']
                gate_draw=decision['exploration_gate'];draw=decision['exploration_code']
            # Old leg feedback is REPLACED in this branch, never summed.
            adjusted=kernel.replace_legs(fixed,env.targets[k],requested,mapping.legs,sim.lower,sim.upper,active)
            np.testing.assert_array_equal(adjusted[mapping.head],fixed[mapping.head])
            base=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
            wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
            assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
            if not active:assert adjusted is fixed
            last_inputs.update(control=np.asarray(k),current_native55=observed,initial_native55=initial,previous_applied=sim.prev.copy(),original_reference=env.targets[k].copy(),post_IMU_right_boundary=fixed.copy(),requested_action=requested)
            maxima.append(float(np.abs(adjusted-env.targets[k]).max()))
            applied=original(adjusted);np.testing.assert_array_equal(applied,wanted);np.testing.assert_array_equal(sim.prev,wanted)
            for key,value in zip(keys,(frozen,fixed,adjusted,wanted,base,requested,active,chosen,greedy,expected_values,selected_probabilities,gate_draw,draw)):
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
                assert all(equality.values()),equality;assert row['peaks']==oldrow['peaks']
                row['complete_R157_parity']=equality
        assert abs(sum(rewards)-row['return_sum'])<1e-10
        row.update(policy_mode=mode,policy_model_sha256=prior.digest(weightfile),sampling_seed=sampling_seed,
            success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),
            recovery_leg_feedback_replaced=mode!='baseline',old_feedback_added_to_policy=False,
            standard_R157_parity_required=mode=='baseline',base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),
            maximum_combined_14_joint_correction_rad=max(maxima),physics_sha256=fingerprint,
            physics_unchanged=fingerprint==prior.physics_hash(sim),compiled_model_sha256=full_fingerprint,
            compiled_model_unchanged=full_fingerprint==utility.model_digest(sim.model),no_root_or_case_input_to_value_policy=True,root_edits_during_recovery=0,
            fixed_codebook_is_normalized_request_not_executed_action=True,greedy_is_expected_categorical_return=True,estimated_return_is_not_safety_probability=True)
        assert row['physics_unchanged'] and row['compiled_model_unchanged']
        np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'rng.json',rng.bit_generator.state);local.write_json(dest/'result.json',row)
        print('R200_PATH',mode,case,row['controls'],row['success'],row['valid'],flush=True)
        return row
    except Exception:
        np.savez_compressed(dest/'partial_trajectory.npz',observations=[r[0] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],original_step_reward=rewards,preparation_sensors=np.asarray(captured['frames']),preparation_times=np.asarray(captured['times']),completed_preparation_sensors=np.asarray([b[0] for b in captured.get('completed_batches',[])]),completed_preparation_times=np.asarray([b[1] for b in captured.get('completed_batches',[])]),**signals)
        np.savez_compressed(dest/'failed_causal_inputs.npz',**last_inputs)
        local.write_json(dest/'failure.json',dict(controls_completed=len(records),terminal_result_saved=False,traceback=traceback.format_exc()))
        raise
    finally:
        prepare_module.record_preparation=old_prepare
        if sim is not None and original is not None:sim.step_target=original


def storage_gate():
    assert shutil.disk_usage(ROOT).free>10*1024**3
    live=sum(p.stat().st_size for d in (OUTPUT,SMOKE,LAUNCH,PREFLIGHT) if d.exists() for p in d.rglob('*') if p.is_file())
    assert 3*live<2*1024**3,'Fixed storage budget exhausted; no cleanup or new attempts'


def group(pool,mode,cases,path,weightfile,seedbase,parity=False):
    storage_gate()
    report=local.aggregate(list(pool.map(evaluate,[(mode,c,str(path/f'case_{c}'),str(weightfile),seedbase+i,parity) for i,c in enumerate(cases)])))
    local.write_json(path/'results.json',report);storage_gate();return report


def smoke_groups(pool,output,initial):
    result={}
    for name,mode,cases,parity in SMOKE_SPEC:
        result[name]=group(pool,mode,cases,output/name,initial,3001000,parity)
    return result


def smoke_gate(reports):
    # None is only checked for groups that actually contain the standard case.
    for name,mode,cases,parity in SMOKE_SPEC:
        report=reports[name]
        if report['physical_failures'] or any(r['controls']!=2279 for r in report['rows']):return False
        if None in cases and not report['nominal_success']:return False
    return all(all(r['complete_R157_parity'].values()) for r in reports['zero_parity']['rows'])


def compare_smoke(output):
    for name,mode,cases,parity in SMOKE_SPEC:
        for case in cases:
            a=output/name/f'case_{case}';b=SMOKE/name/f'case_{case}'
            with np.load(a/'trajectory.npz',allow_pickle=False) as za,np.load(b/'trajectory.npz',allow_pickle=False) as zb:
                assert za.files==zb.files
                for key in za.files:np.testing.assert_array_equal(za[key],zb[key],err_msg=key)
            assert json.loads((a/'result.json').read_text())==json.loads((b/'result.json').read_text())
            assert json.loads((a/'rng.json').read_text())==json.loads((b/'rng.json').read_text())


def dataset(path,weights):
    episodes=[]
    for case in CASES:
        with np.load(path/f'case_{case}/trajectory.npz',allow_pickle=False) as z:
            obs=z['observations'][:,:50];mask=z['policy_active'].astype(bool)
            for k in np.flatnonzero(mask):
                x=kernel.features(obs[k],obs[0],int(k));q,p=kernel.values(weights,x)
                greedy=int(np.argmax(q));draw=int(z['policy_exploration_code'][k])
                chosen=draw if z['policy_exploration_gate'][k]<kernel.EPSILON else greedy
                assert chosen==z['policy_code'][k] and greedy==z['policy_greedy_code'][k]
                np.testing.assert_array_equal(q,z['policy_expected_values'][k])
                np.testing.assert_array_equal(p[chosen],z['policy_selected_probabilities'][k])
                np.testing.assert_array_equal(kernel.codebook()[chosen],z['policy_requested_action'][k])
            episodes.append(kernel.episode_replay(obs,z['policy_code'].astype(np.int64),z['original_step_reward'],mask))
    arrays={key:np.concatenate([e[key] for e in episodes]) for key in episodes[0]}
    np.savez_compressed(path/'new_episode_replay.npz',**arrays)
    return arrays


def eligible(report,baseline):
    oldrows={r['case_seed']:r for r in baseline['rows']}
    return bool(report['nominal_success'] and not report['physical_failures'] and
                report['successes']>baseline['successes'] and
                all(not oldrows[r['case_seed']]['success'] or r['success'] for r in report['rows']))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>12*1024**3
    if not args.smoke:
        assert smoke_gate(json.loads((SMOKE/'results.json').read_text())['reports'])
        learner=kernel.Learner(kernel.initial_weights()) # Import learning runtime before source capture.
    output.mkdir();sources=capture_sources(output/'executed_sources');assets=utility.model_files(local.prior.program.SCENE)
    plan=ROOT/'tmp/GETUP_TOTAL_TARGET_C51_PLAN_R200_20261011.json'
    assert json.loads(plan.read_text())['total_attempt_cap']==162
    shutil.copy2(plan,output/'design_plan.json');assert prior.digest(plan)==prior.digest(output/'design_plan.json')
    frozen=output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:
        shutil.copy2(source,frozen/name);assert prior.digest(source)==prior.digest(frozen/name)
    hashes={str(p):prior.digest(p) for p in [*frozen.iterdir(),local.prior.program.STAND,local.prior.program.REFERENCE]}
    initial=output/'initial.npz'
    if args.smoke:np.savez_compressed(initial,**kernel.initial_weights())
    else:shutil.copy2(SMOKE/'initial.npz',initial)
    with np.load(initial,allow_pickle=False) as z:
        for k,v in kernel.initial_weights().items():np.testing.assert_array_equal(z[k],v)
    local.write_json(output/'contract.json',dict(simulation_only=True,smoke=args.smoke,seed=300,workers=6,
        total_attempt_cap=162,smoke_cap=6,formal_cap=156,training_rounds=2,training_attempts=50,
        categorical_updates_per_round=kernel.UPDATES,learning_rate=kernel.RATE,batch=kernel.BATCH,
        policy_inputs=101,actions=kernel.ACTIONS,return_atoms=kernel.ATOMS,
        return_support=[kernel.VMIN,kernel.VMAX],support_clips_estimator_only=True,
        normalized_code_magnitude=kernel.CODE_MAGNITUDE,epsilon=kernel.EPSILON,
        action_book=kernel.codebook().tolist(),replay_only_new_R200_training=True,
        actor_used=False,twin_value_networks_used=False,entropy_used=False,
        target_hard_copies_per_round=1,target_frozen_within_round=True,
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
            baseline=group(pool,'baseline',CASES,output/'development_baseline',initial,3002000,True)
            assert baseline['successes']==18 and baseline['nominal_success'] and not baseline['physical_failures']
            rng=np.random.default_rng(300); history=[];candidates=[];replay_parts=[];learner.checkpoint(output/'checkpoint_0000');local.write_json(output/'checkpoint_0000_rng.json',rng.bit_generator.state)
            for iteration in (1,2):
                before=output/f'checkpoint_{iteration-1:04d}.npz'
                trainpath=output/f'training_{iteration:04d}'
                train=group(pool,'sample',CASES,trainpath,before,3003000+iteration*100)
                arrays=dataset(trainpath,learner.numpy_weights());replay_parts.append(arrays)
                replay={key:np.concatenate([p[key] for p in replay_parts]) for key in arrays}
                local.write_json(trainpath/'replay_manifest.json',dict(training_rounds=list(range(1,iteration+1)),transitions=len(replay['reward']),old_data_used=False))
                storage_gate();update=learner.update(replay,rng)
                np.savez_compressed(trainpath/'learner_update.npz',**{k:v for k,v in update.items() if isinstance(v,np.ndarray)},**{'last_gradient_'+k:v for k,v in update['last_gradient'].items()})
                learner.checkpoint(output/f'checkpoint_{iteration:04d}');local.write_json(output/f'checkpoint_{iteration:04d}_rng.json',rng.bit_generator.state)
                check=group(pool,'deterministic',CASES,output/f'check_{iteration:04d}',output/f'checkpoint_{iteration:04d}.npz',3005000)
                good=eligible(check,baseline)
                history.append(dict(iteration=iteration,training=train,check=check,eligible=good,changes=update['changes'],categorical_updates=update['updates'],hard_target_copies=update['hard_target_copies']))
                if good:candidates.append((check['successes'],min(r['return_sum'] for r in check['rows']),sum(r['return_sum'] for r in check['rows']),iteration))
                local.write_json(output/'training_closed_history.json',history)
                print('R200_UPDATE_CLOSED',iteration,check['successes'],good,flush=True)
            selected=max(candidates,key=lambda c:(c[0],c[1],c[2],-c[3]))[3] if candidates else None
            model=initial if selected is None else output/f'checkpoint_{selected:04d}.npz'
            mode='baseline' if selected is None else 'deterministic'
            candidate=group(pool,mode,CASES,output/'candidate',model,3005000,selected is None)
            assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
            gate=bool(candidate['nominal_success'] and not candidate['physical_failures'] and candidate['successes']>=22 and candidate['successes']>18)
            local.write_json(output/'results.json',dict(terminal_result_saved=True,actual_dynamic_attempts=156,baseline=baseline,candidate=candidate,selected_checkpoint=selected,selected_R157_fallback=selected is None,history=history,original_development_gate=gate,expanded_development_run=False,independent_qualification_run=False,qualification_passed=False,full_task_completed=False,hardware_readiness=False))
    except Exception:
        local.write_json(output/'failure.json',dict(traceback=traceback.format_exc(),terminal_result_saved=False,full_task_completed=False,hardware_readiness=False));raise
    finally:
        assert assets==utility.model_files(local.prior.program.SCENE)
        assert all(prior.digest(Path(p))==v['sha256'] for p,v in sources.items())
        assert all(prior.digest(Path(p))==sha for p,sha in hashes.items())
        local.write_json(output/'input_hashes_after.json',dict(source_and_model_files_unchanged=True,hashes=hashes,model_files=assets))


if __name__=='__main__':main()




