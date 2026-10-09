"""Finite model-free recurrent sensor readout search; simulation only.

Fixed nonlinear recurrent features, trained coordinated 14-actuator readout.
No fitted forward model, teacher-action loss, gain changes or expert mixing.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import numpy as np
from diagnostics import train_getup_local_hip_r130 as local
from diagnostics import train_getup_set_decision_r157 as selector
from diagnostics import train_getup_velocity_damping_r165 as planner
from diagnostics.getup_independent_native import digest

ROOT=local.ROOT
OUTPUT=ROOT/'outputs/getup_recurrent_readout_r175_left_20261009'
SMOKE=ROOT/'outputs/getup_recurrent_readout_r175_smoke_20261009'
CASES=[None,*local.prior.program.TRAIN]
HIDDEN=16
SHAPE=(14,HIDDEN)
BOUND=.02
SCALE=np.array([1.]*3+[.05]*3+[.05]*14+[.05]*14+[.05]*14+[1.]*2)
MODEL=A=B=None


def fixed_features():
    rng=np.random.default_rng(1750275)
    a=rng.normal(size=(HIDDEN,50));a/=np.abs(a).sum(axis=1,keepdims=True)
    b=rng.normal(size=(HIDDEN,HIDDEN));b*=.75/np.abs(b).sum(axis=1,keepdims=True)
    probe=rng.choice([-1.,1.],size=SHAPE)*.0005
    return a,b,probe,rng.bit_generator.state


def recurrent_feedback(parameters,current,nominal,initial,initial_nominal,state,input_matrix,recurrent_matrix):
    w=np.asarray(parameters,dtype=float);h=np.asarray(state,dtype=float)
    a=np.asarray(input_matrix,dtype=float);b=np.asarray(recurrent_matrix,dtype=float)
    values=[np.asarray(x,dtype=np.float32) for x in (current,nominal,initial,initial_nominal)]
    if w.shape!=SHAPE or not np.isfinite(w).all() or np.any(np.abs(w)>BOUND):raise ValueError('Bounded14x16 readout required')
    if h.shape!=(HIDDEN,) or not np.isfinite(h).all() or np.any(np.abs(h)>1):raise ValueError('Bounded16 causal internal state required')
    if any(x.shape!=(55,) or not np.isfinite(x).all() for x in values):raise ValueError('Finite causal sensor55 snapshots required')
    if a.shape!=(HIDDEN,50) or b.shape!=(HIDDEN,HIDDEN) or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Finite fixed recurrent matrices required')
    if np.max(np.abs(b).sum(axis=1))>.75+1e-14:raise ValueError('Fixed recurrence contraction bound exceeded')
    x,n,x0,n0=values
    delta=((x[:50].astype(float)-n[:50].astype(float))-(x0[:50].astype(float)-n0[:50].astype(float)))/SCALE
    phi=np.tanh(delta)
    hidden=np.tanh(a@phi+b@h)
    extra=.18*np.tanh(w@hidden) if np.any(w) else np.zeros(14)
    return extra,hidden,phi,delta


def init_worker(frozen):
    global MODEL,A,B
    path=Path(frozen)
    with np.load(path/'initial_selector.npz',allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}
    with np.load(path/'snapshot.npz',allow_pickle=False) as z:local.WEIGHTS={k:z[k].copy() for k in z.files}
    with np.load(path/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:local.NOMINAL=z['observations'].copy()
    with np.load(path/'fixed_recurrence.npz',allow_pickle=False) as z:A=z['input_matrix'].copy();B=z['recurrent_matrix'].copy()


def physics_hash(sim):
    h=hashlib.sha256()
    for name in ('body_mass','body_inertia','jnt_range','dof_damping','dof_frictionloss','dof_armature','actuator_forcerange','actuator_ctrlrange','actuator_gainprm','actuator_biasprm','geom_friction','geom_contype','geom_conaffinity'):
        h.update(np.asarray(getattr(sim.model,name)).tobytes())
    h.update(str((sim.model.opt.timestep,int(sim.model.opt.enableflags),int(sim.model.opt.disableflags))).encode())
    return h.hexdigest()


def evaluate(job):
    parameters,case,directory,parity=job
    w=np.asarray(parameters,dtype=float).reshape(SHAPE)
    assert case in CASES
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,275,True)
    env.reset(case);sim=env.sim;fingerprint=physics_hash(sim)
    ids=np.array([sim.model.actuator(n).id for n in local.JOINTS])
    sensors=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in local.JOINTS])
    np.testing.assert_array_equal(sensors,[9,10,11])
    gains,choice,logits=selector.old.predict(MODEL,env.preparation_sensors[-1])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    initial=env.observe().copy();initial_nominal=local.NOMINAL[0].copy();hidden=np.zeros(HIDDEN)
    original=sim.step_target;records=[];frozen_history=[];extras=[];states=[];phis=[];deltas=[]
    planned=[];base_planned=[];pre_direct=[];fixed_targets=[];new_targets=[];maxima=[]
    def controlled(target):
        nonlocal hidden
        k=env.controls;observed=env.observe().copy()
        np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
        frozen=local.local_feedback(observed,local.NOMINAL[k],sensors,gains) if k<529 else np.zeros(3)
        fixed=local.merge_target(target,env.targets[k],frozen,ids,sim.lower,sim.upper)
        if k<529:
            extra,hidden,phi,delta=recurrent_feedback(w,observed,local.NOMINAL[k],initial,initial_nominal,hidden,A,B)
        else:extra=np.zeros(14);hidden=np.zeros(HIDDEN);phi=np.zeros(50);delta=np.zeros(50)
        adjusted=local.merge_target(fixed,env.targets[k],extra,np.arange(14),sim.lower,sim.upper)
        assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
        if k==0:
            for v in (extra,hidden,phi,delta):np.testing.assert_array_equal(v,np.zeros_like(v))
            assert adjusted is fixed
        wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
        unperturbed=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
        frozen_history.append(frozen.copy());extras.append(extra.copy());states.append(hidden.copy());phis.append(phi.copy());deltas.append(delta.copy())
        planned.append(wanted.copy());base_planned.append(unperturbed.copy());pre_direct.append(adjusted-fixed);fixed_targets.append(fixed.copy());new_targets.append(adjusted.copy())
        maxima.append(float(np.abs(adjusted-env.targets[k]).max()))
        applied=original(adjusted)
        np.testing.assert_array_equal(applied,wanted);np.testing.assert_array_equal(sim.prev,wanted)
        return applied
    sim.step_target=controlled
    try:
        while True:
            obs=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    finally:sim.step_target=original
    row=step[5]
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],local_hip_extra_rad=frozen_history,preparation_sensors=env.preparation_sensors,
        recurrent_extra_rad=extras,recurrent_hidden=states,recurrent_phi=phis,causal_sensor_error_change=deltas,planned_before_integration_rad=planned,same_state_unperturbed_planned_rad=base_planned,
        same_state_pre_slew_direct_new_rad=pre_direct,original_double_pre_slew_target_rad=fixed_targets,adjusted_double_pre_slew_target_rad=new_targets)
    reference=selector.OUTPUT/'candidate'/f'case_{case}';old=json.loads((reference/'result.json').read_text());assert row['initial_hash']==old['initial_hash']
    with np.load(reference/'trajectory.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.asarray(arrays['applied'])[0],z['applied'][0])
        if parity or case is None:
            equal={k:bool(np.array_equal(np.asarray(arrays[k]),z[k])) for k in z.files}
            assert all(equal.values()),equal;assert row['peaks']==old['peaks'];row['complete_R157_parity']=equal
    if case is None:
        for v in (extras,states,phis,deltas):np.testing.assert_array_equal(v,np.zeros_like(v))
    row.update(parameters=w.tolist(),base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),
        full_path=True,maximum_combined_14_joint_correction_rad=max(maxima),maximum_recurrent_extra_rad=float(np.abs(extras).max()),maximum_hidden=float(np.abs(states).max()),
        online_planned_target_equals_actual_applied_bitwise=True,physics_sha256=fingerprint,physics_unchanged=fingerprint==physics_hash(sim),root_edits_during_recovery=0,
        actuator_names=[sim.model.actuator(i).name for i in range(14)],no_root_truth_or_case_metadata_in_controller=True)
    assert row['physics_unchanged']
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row


def group(pool,w,cases,path,parity=False):
    report=local.aggregate(list(pool.map(evaluate,[(w.tolist(),case,str(path/f'case_{case}'),parity) for case in cases])))
    local.write_json(path/'results.json',report);return report


def compare_smoke(output):
    for name,cases in [('zero_parity',[None,769000,773004]),('nonzero_smoke',[None,769002,773004])]:
        for case in cases:
            fresh=output/name/f'case_{case}';old=SMOKE/name/f'case_{case}'
            with np.load(fresh/'trajectory.npz',allow_pickle=False) as a,np.load(old/'trajectory.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            x=json.loads((fresh/'result.json').read_text());y=json.loads((old/'result.json').read_text())
            assert x['initial_hash']==y['initial_hash'] and x['peaks']==y['peaks'] and x['success']==y['success']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>12*1024**3
    audit=json.loads((ROOT/'outputs/getup_intervention_dynamics_audit_r174_20261009/results.json').read_text())
    assert audit['read_only']
    output.mkdir();src=output/'executed_sources';src.mkdir()
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if name.startswith('diagnostics.') and f and Path(f).suffix=='.py':shutil.copy2(f,src/Path(f).name)
    for name in [Path(__file__).name,'test_getup_recurrent_readout_r175.py','launch_getup_recurrent_readout_r175.py']:shutil.copy2(Path(__file__).with_name(name),src/name)
    frozen=output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:
        shutil.copy2(source,frozen/name);assert digest(source)==digest(frozen/name)
    a,b,probe,feature_rng=fixed_features()
    np.savez_compressed(frozen/'fixed_recurrence.npz',input_matrix=a,recurrent_matrix=b,scales=SCALE,probe=probe)
    local.write_json(frozen/'feature_rng.json',feature_rng)
    local.write_json(output/'contract.json',dict(seed=275,smoke=args.smoke,simulation_only=True,generations=2,population=8,workers=6,parameters=224,readout_shape=list(SHAPE),parameter_bound=BOUND,
        hypothesis='Cross-case rescue/regression persists for instantaneous gains, mixtures and scalar tracking memory; test nonlinear cross-sensor recurrent state with a coordinated14-actuator readout using complete-path model-free search. No claimed unique cause or improvement.',
        differences='Fixed16-dimensional nonlinear recurrence over full causal native50 error-change; trained14x16 readout. Not old feedforward PPO/BC, teacher aggregation, two tracking amplitudes, gain/mix gate or forward-model optimization.',
        formula='phi=tanh(((current_native50-nominal_native50)-(initial_native50-initial_nominal_native50))/fixed_scales); h=tanh(A*phi+B*h_previous); extra=.18*tanh(C*h)',
        fixed_feature_seed=1750275,recurrence_row_abs_sum_bound=.75,feature_matrices_not_searched=True,initial_std=.0005,std_bounds=[.0001,.002],elite=2,cem_update=[.6,.4],
        inputs='Only current/causal initial actual native50 and past internal state; same-phase nominal sequence frozen; R157 selector, R122 program and originalIMU/right-hip feedback frozen.',
        no_teacher_labels_or_root_truth_or_case_or_seed_or_directory_or_future_inputs=True,no_context_lookup=True,first_target_exact_original=True,nominal_scalar_exact_zero=True,zero_object_identity_preserved=True,
        home_original=True,hidden_reset_at_home_not_root_reset=True,extra_joint_count=14,combined_original_and_new_14_joint_cap_rad=.18,physics_reward_acceptance_original=True,
        controls=2279,control_hz=50,physics_hz=500,entry_deadline_s=12,strict_tail_s=30,training_cases=CASES,all_proposals_original_full_path_acceptance=True,
        automatic_expanded_or_qualification=False,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,
        hashes={str(f):digest(f) for f in [*src.iterdir(),*frozen.iterdir(),local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE]}))
    zero=np.zeros(SHAPE);selected=zero.copy();mean=zero.copy();std=np.full(SHAPE,.0005);rng=np.random.default_rng(275);history=[];cache={};best=None
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
        parity=group(pool,zero,[None,769000,773004],output/'zero_parity',True)
        nonzero=group(pool,probe,[None,769002,773004],output/'nonzero_smoke')
        assert all(row['maximum_recurrent_extra_rad']>0 for row in nonzero['rows'] if row['case_seed'] is not None)
        if not args.smoke:compare_smoke(output)
        local.write_json(output/'startup_closed.json',dict(parity=parity,nonzero=nonzero,independent_smoke_bitwise_equal=not args.smoke,terminal_result_saved=False,training_generations_saved=0))
        print('R175_ZERO_NOMINAL_AND_STARTUP_PARITY_PASS',flush=True)
        if args.smoke:
            local.write_json(output/'results.json',dict(smoke=True,parity=parity,nonzero=nonzero,terminal_result_saved=True,full_task_completed=False,hardware_readiness=False));return
        for generation in range(1,3):
            assert shutil.disk_usage(ROOT).free>10*1024**3,'Insufficient space; preserve closed evidence without cleanup'
            proposals=[zero.copy(),selected.copy(),*np.clip(rng.normal(mean,std,(6,*SHAPE)),-BOUND,BOUND)];reports=[];locations=[]
            for i,w in enumerate(proposals):
                key=w.tobytes().hex();path=output/'training'/f'generation_{generation:04d}'/f'candidate_{i:02d}'
                if key not in cache:cache[key]=(group(pool,w,CASES,path,not np.any(w)),str(path))
                report,location=cache[key];reports.append(report);locations.append(location)
                print('R175_FULL_CANDIDATE',generation,i,report['successes'],report['physical_failures'],flush=True)
            order=sorted(range(8),key=lambda i:local.rank(reports[i]),reverse=True);winner=order[0]
            if best is None or local.rank(reports[winner])>local.rank(best):selected=proposals[winner].copy();best=reports[winner]
            elite=np.stack([proposals[i] for i in order[:2]]);mean=.6*mean+.4*elite.mean(0);std=np.clip(.6*std+.4*elite.std(0),.0001,.002)
            history.append(dict(generation=generation,proposals=[w.tolist() for w in proposals],reports=reports,closed_trial_directories=locations,selected_parameters=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
            checkpoint=output/'checkpoints'/f'generation_{generation:04d}';checkpoint.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(checkpoint/'state.npz',parameters=selected,mean=mean,std=std,input_matrix=a,recurrent_matrix=b)
            local.write_json(checkpoint/'rng.json',rng.bit_generator.state);local.write_json(checkpoint/'history.json',history)
            local.write_json(output/'progress.json',dict(closed_generation=generation,generations=2,best=best,parameters=selected.tolist(),terminal_result_saved=False))
        local.write_json(output/'training_closed.json',dict(history=history,rng=rng.bit_generator.state,best=best,parameters=selected.tolist(),all_training_full_path=True))
        candidate=group(pool,selected,CASES,output/'development_candidate');baseline=group(pool,zero,CASES,output/'development_baseline',True)
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
        local.write_json(output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,parameters=selected.tolist(),original_development_gate=gate,terminal_result_saved=True,
            expanded_development_run=False,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
        print('R175_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)


if __name__=='__main__':main()
