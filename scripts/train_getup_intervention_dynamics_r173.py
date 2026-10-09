"""Finite full-episode signed action interventions, then offline dynamics fitting.

Identification experiment, NOT an optimized recovery controller or deployment.
No mid-episode state cloning/reset; signs are frozen instruments, not case input.
"""
import argparse
import hashlib
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_local_hip_r130 as local
from diagnostics import train_getup_set_decision_r157 as selector
from diagnostics import train_getup_velocity_damping_r165 as planner
from diagnostics import train_getup_sensor_dynamics_r172b as dynamics
from diagnostics.getup_independent_native import digest

ROOT=local.ROOT if hasattr(local,'ROOT') else local.prior.ROOT
OUTPUT=ROOT/'outputs/getup_intervention_dynamics_r173_20261009'
SMOKE=ROOT/'outputs/getup_intervention_dynamics_r173_smoke_20261009'
CASES=[None,*local.prior.program.TRAIN]
AMPLITUDE=.0005
MODEL=PLANS=None

def instruments():
    rng=np.random.default_rng(273)
    signs=rng.choice([-1.,1.],(529,3));signs[0]=0
    plans=np.zeros((7,529,3))
    for axis in range(3):
        plans[1+2*axis,:,axis]=AMPLITUDE*signs[:,axis]
        plans[2+2*axis]=-plans[1+2*axis]
    return plans,rng.bit_generator.state

def intervention(current,nominal,k,plan):
    x=np.asarray(current,dtype=np.float32);n=np.asarray(nominal,dtype=np.float32);p=np.asarray(plan,dtype=float)
    if x.shape!=(55,) or n.shape!=(55,) or p.shape!=(529,3):raise ValueError('Sensor55 and fixed529x3 plan required')
    if not all(np.isfinite(v).all() for v in (x,n,p)) or np.abs(p).max()>AMPLITUDE:raise ValueError('Finite bounded instruments required')
    if type(k) is not int or k<0:raise ValueError('Integer causal control index required')
    if k==0 or k>=529:return np.zeros(3),0.
    error=(x[:34].astype(float)-n[:34].astype(float))/dynamics.SCALE[:34]
    activation=float(np.tanh(np.abs(error).max()))
    return p[k]*activation,activation

def init_worker(frozen):
    global MODEL,PLANS
    frozen=Path(frozen)
    with np.load(frozen/'initial_selector.npz',allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}
    with np.load(frozen/'snapshot.npz',allow_pickle=False) as z:local.WEIGHTS={k:z[k].copy() for k in z.files}
    with np.load(frozen/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:local.NOMINAL=z['observations'].copy()
    with np.load(frozen/'instruments.npz',allow_pickle=False) as z:PLANS=z['plans'].copy()

def evaluate(job):
    program,case,directory=job
    assert case in CASES and program in range(7)
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,273,True)
    env.reset(case);sim=env.sim
    def physical_fingerprint():
        h=hashlib.sha256()
        for name in ('body_mass','body_inertia','jnt_range','dof_damping','dof_frictionloss','dof_armature',
                     'actuator_forcerange','actuator_ctrlrange','actuator_gainprm','actuator_biasprm','geom_friction','geom_contype','geom_conaffinity'):
            h.update(np.asarray(getattr(sim.model,name)).tobytes())
        h.update(str((sim.model.opt.timestep,int(sim.model.opt.enableflags),int(sim.model.opt.disableflags))).encode())
        return h.hexdigest()
    physics_before=physical_fingerprint()
    ids=np.array([sim.model.actuator(n).id for n in local.JOINTS])
    sensors=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in local.JOINTS])
    np.testing.assert_array_equal(sensors,[9,10,11])
    gains,choice,logits=selector.old.predict(MODEL,env.preparation_sensors[-1])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    original=sim.step_target;records=[];observed_history=[];frozen_history=[];extra_history=[];activations=[]
    planned=[];base_planned=[];direct=[];maximum=[]
    def controlled(target):
        k=env.controls;observed=env.observe().copy()
        np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
        frozen=local.local_feedback(observed,local.NOMINAL[k],sensors,gains) if k<529 else np.zeros(3)
        fixed=local.merge_target(target,env.targets[k],frozen,ids,sim.lower,sim.upper)
        extra,activation=intervention(observed,local.NOMINAL[k],int(k),PLANS[program])
        adjusted=local.merge_target(fixed,env.targets[k],extra,ids,sim.lower,sim.upper)
        wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
        unperturbed=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
        assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
        maximum.append(float(np.abs(adjusted-env.targets[k]).max()))
        observed_history.append(observed);frozen_history.append(frozen);extra_history.append(extra);activations.append(activation)
        planned.append(wanted.copy());base_planned.append(unperturbed.copy());direct.append(wanted-unperturbed)
        applied=original(adjusted)
        np.testing.assert_array_equal(applied,wanted)
        np.testing.assert_array_equal(sim.prev,wanted)
        return applied
    sim.step_target=controlled
    try:
        while True:
            obs=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    finally:sim.step_target=original
    np.testing.assert_array_equal(observed_history,[r[0] for r in records])
    np.testing.assert_array_equal(extra_history[0],np.zeros(3))
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],
        qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],local_hip_extra_rad=frozen_history,
        preparation_sensors=env.preparation_sensors,intervention_requested_extra_rad=extra_history,intervention_activation=activations,
        planned_before_integration_rad=planned,same_state_unperturbed_planned_rad=base_planned,same_state_executed_target_difference_rad=direct)
    reference=selector.OUTPUT/'candidate'/f'case_{case}'
    old=json.loads((reference/'result.json').read_text());row=step[5];assert row['initial_hash']==old['initial_hash']
    with np.load(reference/'trajectory.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.asarray(arrays['applied'])[0],z['applied'][0])
        if program==0 or case is None:
            equal={k:bool(np.array_equal(np.asarray(arrays[k]),z[k])) for k in z.files}
            assert all(equal.values()),equal;assert row['peaks']==old['peaks'];row['complete_R157_parity']=equal
    if case is None:np.testing.assert_array_equal(extra_history,np.zeros((len(records),3)))
    row.update(instrument_program=program,base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),
        maximum_combined_14_joint_correction_rad=max(maximum),maximum_requested_instrument_rad=float(np.abs(extra_history).max()),
        online_planned_target_equals_actual_applied_bitwise=True,instrument_not_optimized_policy=True,root_edits_during_recovery=0,
        physics_sha256=physics_before,physics_unchanged=physics_before==physical_fingerprint())
    assert row['physics_unchanged']
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row

def compare_prefixes(root,cases):
    """All trajectories start independently; common state before FIRST intervention only."""
    rows=[]
    for case in cases:
        with np.load(root/'program_00'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as z:
            initial=z['observations'][:2].copy();first=z['applied'][0].copy()
        for axis in range(3):
            paths=[root/f'program_{p:02d}'/f'case_{case}' for p in (1+2*axis,2+2*axis)]
            with np.load(paths[0]/'trajectory.npz',allow_pickle=False) as a,np.load(paths[1]/'trajectory.npz',allow_pickle=False) as b:
                for z in (a,b):
                    np.testing.assert_array_equal(z['observations'][:2],initial)
                    np.testing.assert_array_equal(z['applied'][0],first)
                    np.testing.assert_array_equal(z['planned_before_integration_rad'],z['applied'])
                np.testing.assert_array_equal(a['same_state_unperturbed_planned_rad'][1],b['same_state_unperturbed_planned_rad'][1])
                np.testing.assert_array_equal(a['intervention_requested_extra_rad'][1],-b['intervention_requested_extra_rad'][1])
                rows.append(dict(case_seed=case,axis=axis,control1_common_state_exact=True,
                    requested_pair_separation_rad=float(np.abs(a['intervention_requested_extra_rad'][1]-b['intervention_requested_extra_rad'][1]).max()),
                    applied_pair_separation_rad=float(np.abs(a['applied'][1]-b['applied'][1]).max()),
                    next_native34_pair_separation=float(np.abs(a['observations'][2,:34]-b['observations'][2,:34]).max()),
                    subsequent_states_not_assumed_equal=True))
    local.write_json(root/'first_intervention_pairs.json',rows)
    return rows

def fit_interventional_model(output):
    """No root arrays/labels enter features; valid flag filters whole episodes offline."""
    with np.load(output/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:
        nominal=z['observations'][:,:50].copy();nominal_applied=z['applied'].copy()
    data={};manifest=[]
    for program in range(7):
        for case in CASES:
            path=output/'episodes'/f'program_{program:02d}'/f'case_{case}'
            row=json.loads((path/'result.json').read_text())
            meta=dict(program=program,case_seed=case,initial_hash=row['initial_hash'],valid=row['valid'],trajectory_sha256=digest(path/'trajectory.npz'))
            manifest.append(meta)
            if not row['valid']:continue  # All failures retained; disclose selection bias.
            with np.load(path/'trajectory.npz',allow_pickle=False) as z:
                obs=z['observations'][:,:50].copy();applied=z['planned_before_integration_rad'].copy()
            assert obs.shape==(2279,50) and applied.shape==(2279,14)
            data[program,case]=dynamics.samples(obs,applied,nominal,nominal_applied)
    training=[k for k in data if k[0]<5 and k[1] not in dynamics.HELD_CASES]
    f=np.concatenate([data[k][0] for k in training]);y=np.concatenate([data[k][1] for k in training])
    gram=f.T@f;rhs=f.T@y;w=np.linalg.solve(gram+np.eye(342),rhs)
    blindf=f.copy()
    for lo in (100,214,328):blindf[:,lo:lo+14]=0
    blind=np.linalg.solve(blindf.T@blindf+np.eye(342),blindf.T@y)
    dest=output/'learner_closed_01';dest.mkdir()
    np.savez_compressed(dest/'model.npz',weights=w,action_blind_weights=blind,sensor_scales=dynamics.SCALE,ridge=1.)
    np.savez_compressed(dest/'normal_equations.npz',gram=gram,rhs=rhs,weights=w,blind_gram=blindf.T@blindf,blind_rhs=blindf.T@y,blind_weights=blind)
    local.write_json(dest/'learner.json',dict(seed=273,ridge=1.,solver='float64 normal equations, one fixed solve and action-input ablation',
        training_keys=[list(k) for k in training],training_transitions=len(f),rng=np.random.default_rng(273).bit_generator.state,
        optimizer_random_sampling=False,root_state_inputs=False,invalid_episodes_excluded_from_fit=True,not_a_controller=True))
    local.write_json(dest/'input_manifest.json',manifest)
    signals=dest/'predictions';signals.mkdir();reports=[]
    for (program,case),(x,target,past,truth) in data.items():
        pred=past+x@w*dynamics.SCALE[:34];abl=past+x@blind*dynamics.SCALE[:34]
        if case is None:np.testing.assert_array_equal(pred,np.zeros_like(pred))
        split='training' if (program,case) in training else 'both_held' if program>=5 and case in dynamics.HELD_CASES else 'case_held' if case in dynamics.HELD_CASES else 'axis_held'
        np.savez_compressed(signals/f'program_{program:02d}_case_{case}.npz',predicted_native34_deviation=pred,action_blind_deviation=abl,
                            persistence_deviation=past,actual_next_native34_deviation=truth)
        reports.append(dict(program=program,case_seed=case,split=split,metrics=dynamics.stage_metrics(pred,abl,past,truth)))
    local.write_json(dest/'evaluation.json',reports)
    return dict(model_sha256=digest(dest/'model.npz'),training_trajectories=len(training),training_transitions=len(f),
                valid_input_trajectories=len(data),invalid_excluded=175-len(data),held_cases=list(dynamics.HELD_CASES),held_axis='right_hip_pitch',
                evaluation_trajectories=len(reports),one_step_model_not_predictive_controller=True)

def group(pool,program,cases,path):
    report=local.aggregate(list(pool.map(evaluate,[(program,c,str(path/f'case_{c}')) for c in cases])))
    local.write_json(path/'results.json',report)
    return report

def compare_smoke(output):
    for name,cases in [('zero_parity',[None,769000,773004]),('nonzero_smoke',[None,769002,773004])]:
        for case in cases:
            a=output/name/f'case_{case}';b=SMOKE/name/f'case_{case}'
            with np.load(a/'trajectory.npz',allow_pickle=False) as x,np.load(b/'trajectory.npz',allow_pickle=False) as y:
                assert x.files==y.files
                for k in x.files:np.testing.assert_array_equal(x[k],y[k])
            ra=json.loads((a/'result.json').read_text());rb=json.loads((b/'result.json').read_text())
            assert ra['initial_hash']==rb['initial_hash'] and ra['peaks']==rb['peaks'] and ra['success']==rb['success']

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    output=SMOKE if args.smoke else OUTPUT
    assert not output.exists();assert shutil.disk_usage(ROOT).free>8*1024**3
    output.mkdir();src=output/'executed_sources';src.mkdir()
    names=[Path(__file__).name,'test_getup_intervention_dynamics_r173.py','launch_getup_intervention_dynamics_r173.py',
           'train_getup_sensor_dynamics_r172b.py','train_getup_velocity_damping_r165.py','train_getup_set_decision_r157.py',
           'train_getup_success_selector_r134.py','train_getup_local_hip_r130.py','train_getup_history_program_r122.py',
           'probe_getup_sensor_history_r121.py','train_getup_program_r113.py','getup_reference_env_r100.py',
           'getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py']
    for name in names:shutil.copy2(Path(__file__).with_name(name),src/name)
    frozen=output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),
                        (local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),
                        (local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:
        shutil.copy2(source,frozen/name);assert digest(source)==digest(frozen/name)
    plans,rng=instruments();np.savez_compressed(frozen/'instruments.npz',plans=plans,amplitude=AMPLITUDE,seed=273)
    local.write_json(frozen/'instrument_rng.json',rng)
    hashes={str(f):digest(f) for f in [*src.iterdir(),*frozen.iterdir(),*[Path(__file__).with_name(n) for n in names],
            local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE]}
    local.write_json(output/'contract.json',dict(seed=273,smoke=args.smoke,simulation_only=True,
        method='Full-episode paired signed preplanned action interventions then one offline transition solve',
        hypothesis='R172 logged-action ablation showed weak action information; collect deliberately assigned signed instruments and verify online planned targets instead of expanding old feedback or MSE budgets',
        no_policy_parameter_search=True,no_candidate_promotion=True,instrument_amplitude_fixed_rad=AMPLITUDE,
        instrument_programs=7,formal_full_path_attempts=175,formal_startup_attempts=6,independent_smoke_attempts=6,
        instruments='Frozen axis-wise signed schedules shared across all cases; norm of current native34 deviation only ensures nominal zero',
        first_target_original=True,standard_scalar_exact_zero=True,current_planned_target_verified_before_integration=True,
        common_state_claim_only_before_first_intervention=True,later_divergence_not_instantaneous_causal_gain=True,
        inputs='Current native34, same-phase nominal sensor, frozen control-index instrument; R157 initial native50 selector',
        no_case_label_root_truth_future_or_lookup_controller_inputs=True,home_original=True,combined_14_joint_cap_rad=.18,
        full_controls=2279,control_hz=50,physics_hz=500,entry_deadline_s=12,strict_tail_s=30,
        physics_reward_acceptance_unchanged=True,no_extra_preparation_integration=True,no_mid_episode_root_reset=True,
        qualification_never_loaded=True,automatic_expanded_or_qualification=False,learner_features_unchanged_from_R172=True,
        learner_difference='New deliberately intervened physical data, not another ridge/feature grid on old passive data',
        fixed_ridge=1.,invalid_episodes_retained_but_excluded_from_fit=True,held_cases=list(dynamics.HELD_CASES),held_axis='right_hip_pitch',
        full_task_completed=False,hardware_readiness=False,hashes=hashes))
    reports=[]
    with ProcessPoolExecutor(3 if args.smoke else 6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
        parity=group(pool,0,[None,769000,773004],output/'zero_parity')
        nonzero=group(pool,1,[None,769002,773004],output/'nonzero_smoke')
        if not args.smoke:compare_smoke(output)
        local.write_json(output/'startup_closed.json',dict(parity=parity,nonzero=nonzero,independent_smoke_bitwise_equal=not args.smoke,
            formal_data_programs_closed=0,terminal_result_saved=False))
        print('R173_STARTUP_PARITY_AND_ONLINE_PLANNER_PASS',flush=True)
        if args.smoke:
            local.write_json(output/'results.json',dict(smoke=True,parity=parity,nonzero=nonzero,terminal_result_saved=True,full_task_completed=False));return
        for program in range(7):
            report=group(pool,program,CASES,output/'episodes'/f'program_{program:02d}');reports.append(report)
            if program==0:assert report['successes']==18 and report['nominal_success'] and not report['physical_failures']
            checkpoint=output/'checkpoints'/f'program_{program:02d}';checkpoint.mkdir(parents=True,exist_ok=False)
            local.write_json(checkpoint/'closed.json',dict(program=program,report=report,instrument_rng=rng,closed_episodes=25))
            local.write_json(output/'progress.json',dict(closed_program=program,programs=7,full_path_attempts_closed=25*(program+1),
                baseline_successes=reports[0]['successes'],reports=reports,no_candidate_promotion=True,terminal_result_saved=False))
            print('R173_INSTRUMENT_COMPLETE',program,report['successes'],report['physical_failures'],flush=True)
    pairs=compare_prefixes(output/'episodes',CASES)
    local.write_json(output/'training_data_closed.json',dict(programs=7,attempts=175,reports=reports,common_state_pairs=len(pairs),rng=rng))
    learner=fit_interventional_model(output)
    assert all(digest(Path(f))==h for f,h in hashes.items())
    local.write_json(output/'training_closed.json',dict(learner=learner,data_attempts=175,startup_attempts=6,all_failures_saved=True,rng=rng))
    local.write_json(output/'results.json',dict(smoke=False,terminal_result_saved=True,intervention_reports=reports,learner=learner,
        baseline_successes=18,current_best_unified_policy='R157',current_best_unified_successes=18,new_controller=False,
        original_development_gate=False,expanded_development_run=False,independent_qualification_run=False,
        full_task_completed=False,hardware_readiness=False,source_hashes_unchanged=True))
    print('R173_DATA_AND_MODEL_CLOSED_NOT_NEW_CONTROLLER',learner,flush=True)

if __name__=='__main__':main()
