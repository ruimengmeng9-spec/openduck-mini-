"""Bounded direct odd residual from causal joint-error changes on frozen R134.

No multiplying the residual by the current error, gain adaptation, case lookup,
root truth or future sensor input. Original physical and acceptance limits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_dynamic_gain_r147 as prior
from diagnostics.getup_independent_native import digest

ROOT=prior.ROOT
OUTPUT=ROOT/'outputs/getup_incremental_residual_r155_left_20261006'
SMOKE=ROOT/'outputs/getup_incremental_residual_r155_smoke_20261006'
PROBE=np.array([.01,.01,.01,.005,.005,.005])

def incremental_feedback(parameters,current,nominal,initial,nominal_initial,ids):
    parameters=np.asarray(parameters,dtype=float)
    if parameters.shape!=(6,) or not np.isfinite(parameters).all() or np.abs(parameters).max()>1:
        raise ValueError('Six finite coefficients bounded by one required')
    delta=prior.state_features(current,nominal,ids)[6:]-prior.state_features(initial,nominal_initial,ids)[6:]
    activation=np.tanh(delta)
    if not np.any(parameters) or not np.any(delta):return np.zeros(3),activation
    return -.18*np.tanh(parameters[:3]*activation[:3]+parameters[3:]*activation[3:]),activation

def evaluate(job):
    parameters,case,full,directory,parity=job
    assert case in {None,*prior.local.prior.program.TRAIN}
    local=prior.local
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,255,full)
    env.reset(case);sim=env.sim
    base,choice,logits=prior.selector.predict(prior.MODEL,env.preparation_sensors[-1])
    initial=env.observe().copy()
    actuator_ids=np.array([sim.model.actuator(n).id for n in local.JOINTS])
    sensor_ids=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in local.JOINTS])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    original=sim.step_target;extra=[];incremental_extra=[];activation_history=[];combined=[]
    def controlled(target):
        k=env.controls
        if k<529:
            feedback=local.local_feedback(env.observe(),local.NOMINAL[k],sensor_ids,base)
            incremental,activation=incremental_feedback(parameters,env.observe(),local.NOMINAL[k],initial,local.NOMINAL[0],sensor_ids)
        else:activation=np.zeros(6);feedback=np.zeros(3);incremental=np.zeros(3)
        frozen_target=local.merge_target(target,env.targets[k],feedback,actuator_ids,sim.lower,sim.upper)
        adjusted=local.merge_target(frozen_target,env.targets[k],incremental,actuator_ids,sim.lower,sim.upper)
        combined.append(float(np.abs(adjusted[actuator_ids]-env.targets[k][actuator_ids]).max()))
        assert combined[-1]<=.18+1e-12
        if k==0:assert not np.any(activation) and not np.any(incremental);np.testing.assert_array_equal(adjusted,frozen_target)
        extra.append(feedback.copy());incremental_extra.append(incremental.copy());activation_history.append(activation.copy())
        return original(adjusted)
    sim.step_target=controlled;records=[]
    try:
        while True:
            obs=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    finally:sim.step_target=original
    row=step[5];row.update(parameters=list(parameters),full_path=full,
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8) if full else bool(row['training_success']),
        base_choice=choice,base_gains=base.tolist(),initial_sensor_logits=logits.tolist(),
        maximum_combined_right_hip_correction_rad=max(combined),root_edits_during_recovery=0,no_case_metadata_in_controller=True)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],
        qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],
        local_hip_extra_rad=extra,preparation_sensors=env.preparation_sensors,
        incremental_extra_rad=incremental_extra,incremental_activation=activation_history)
    reference=prior.selector.OUTPUT/'candidate'/f'case_{case}'
    saved_row=json.loads((reference/'result.json').read_text());assert row['initial_hash']==saved_row['initial_hash']
    with np.load(reference/'trajectory.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.array(arrays['applied'])[0],z['applied'][0])
        if parity or case is None:
            equal={k:bool(np.array_equal(v,z[k][:len(records)])) for k,v in arrays.items() if k not in ('incremental_extra_rad','incremental_activation')}
            assert all(equal.values()),equal
            assert row['peaks']==saved_row['peaks'] if full else True
            row['frozen_R134_trace_bitwise_equal']=equal
    if case is None:
        np.testing.assert_array_equal(activation_history,np.zeros((len(records),6)))
        np.testing.assert_array_equal(extra,np.zeros((len(records),3)));np.testing.assert_array_equal(incremental_extra,np.zeros((len(records),3)))
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row

def group(pool,parameters,cases,full,path,parity=False):
    rows=list(pool.map(evaluate,[(parameters.tolist(),c,full,str(path/f'case_{c}'),parity) for c in cases]))
    report=prior.local.aggregate(rows);prior.local.write_json(path/'results.json',report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve()
    audit=json.loads((ROOT/'outputs/getup_error_products_r154_20261006/results.json').read_text())
    assert len(audit['rows'])==25 and all(r['scalar_activation_gain_feedback_exact'] for r in audit['rows'])
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for name in [Path(__file__).name,'test_getup_incremental_residual_r155.py','launch_getup_incremental_residual_r155.py',
        'audit_getup_error_products_r154.py','train_getup_dynamic_gain_r147.py','train_getup_success_selector_r134.py',
        'train_getup_local_hip_r130.py','train_getup_history_program_r122.py','probe_getup_sensor_history_r121.py',
        'train_getup_program_r113.py','getup_reference_env_r100.py','getup_independent_native.py',
        'train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py']:
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    shutil.copytree(prior.OUTPUT/'frozen',args.output/'frozen')
    frozen=args.output/'frozen';model_file=frozen/'initial_selector.npz'
    prior.local.write_json(args.output/'contract.json',dict(seed=255,smoke=args.smoke,workers=6,generations=6,population=12,
        method='CEM of six direct odd causal joint-error-change residual coefficients on frozen R134; not gain adaptation',
        hypothesis='R154 reconstructs gain adaptation multiplied by current error. Direct odd residuals from causal error changes avoid that multiplication; this is an unverified structural hypothesis, not a proven cause or improvement.',
        inputs='Actual current and causal initial right-hip position/velocity errors minus same-phase nominal sensor errors; frozen R134 initial native50 selector',
        inference_no_gyro_up_shared_activation=True,direct_residual_parameters=6,coefficient_limit=1.,frozen_base_gain_limit=2.,
        first_control_exact_R134=True,standard_scalar_activation_exact_zero=True,home_original=True,
        no_lookup_no_case_no_root_truth_no_future_sensor_input=True,no_mid_episode_root_edits=True,
        combined_correction_rad=.18,full_controls=2279,training_controls=629,short_tail_1s_not_acceptance=True,
        control_hz=50,physics_hz=500,entry_deadline_s=12,strict_standing_tail_s=30,
        physical_limits_rewards_and_acceptance_unchanged=True,independent_qualification_never_loaded=True,
        training_cases=[None,*prior.local.prior.program.TRAIN],automatic_expanded_or_qualification=False,
        hardware_readiness=False,full_task_completed=False,
        hashes={str(f):digest(f) for f in [prior.local.prior.program.SCENE,prior.local.prior.program.STAND,prior.local.prior.program.REFERENCE,*sources.iterdir(),*frozen.iterdir()]}))
    zero=np.zeros(6);selected=zero.copy();mean=zero.copy();std=np.full(6,.01)
    rng=np.random.default_rng(255);history=[];cache={};best=None;cases=[None,*prior.local.prior.program.TRAIN]
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=prior.init_worker,initargs=(model_file,)) as pool:
        parity=group(pool,zero,[None,769000,773004],True,args.output/'zero_parity',True)
        nonzero=group(pool,PROBE,[None,769002,773004],True,args.output/'nonzero_smoke')
        prior.local.write_json(args.output/'startup_closed.json',dict(parity=parity,nonzero=nonzero,terminal_result_saved=False))
        print('R155_ZERO_AND_NOMINAL_FULL_PARITY_PASS',flush=True)
        if args.smoke:
            prior.local.write_json(args.output/'results.json',dict(smoke=True,parity=parity,nonzero=nonzero,full_task_completed=False));return
        for generation in range(1,7):
            proposals=[zero.copy(),selected.copy(),*np.clip(rng.normal(mean,std,(10,6)),-1.,1.)]
            reports=[];locations=[]
            for i,parameters in enumerate(proposals):
                key=parameters.tobytes().hex();folder=args.output/'training'/f'generation_{generation:04d}'/f'candidate_{i:02d}'
                if key not in cache:cache[key]=(group(pool,parameters,cases,False,folder,not np.any(parameters)),str(folder))
                report,location=cache[key];reports.append(report);locations.append(location)
                print('R155_CANDIDATE',generation,i,report['successes'],report['physical_failures'],flush=True)
            ordered=sorted(range(len(proposals)),key=lambda i:prior.local.rank(reports[i]),reverse=True);winner=ordered[0]
            if best is None or prior.local.rank(reports[winner])>prior.local.rank(best):selected=proposals[winner].copy();best=reports[winner]
            elite=np.stack([proposals[i] for i in ordered[:3]])
            mean=.6*mean+.4*elite.mean(0);std=np.clip(.6*std+.4*elite.std(0),.002,.05)
            history.append(dict(generation=generation,proposals=[v.tolist() for v in proposals],reports=reports,closed_trial_directories=locations,
                selected_parameters=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
            checkpoint=args.output/'checkpoints'/f'generation_{generation:04d}';checkpoint.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(checkpoint/'state.npz',parameters=selected,mean=mean,std=std)
            prior.local.write_json(checkpoint/'rng.json',rng.bit_generator.state);prior.local.write_json(checkpoint/'history.json',history)
            prior.local.write_json(args.output/'progress.json',dict(closed_generation=generation,generations=6,best=best,parameters=selected.tolist()))
        prior.local.write_json(args.output/'training_closed.json',dict(history=history,rng=rng.bit_generator.state,best=best,parameters=selected.tolist(),short_label_only=True))
        candidate=group(pool,selected,cases,True,args.output/'development_candidate')
        baseline=group(pool,zero,cases,True,args.output/'development_baseline',True)
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and candidate['physical_failures']==0)
        prior.local.write_json(args.output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,parameters=selected.tolist(),
            original_development_gate=gate,expanded_development_run=False,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
        print('R155_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)

if __name__=='__main__':main()
