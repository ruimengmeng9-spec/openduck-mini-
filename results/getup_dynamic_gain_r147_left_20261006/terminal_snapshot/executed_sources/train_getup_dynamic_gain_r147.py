"""Bounded rank-one causal state-change gain adaptation on frozen R134.

Not static context refitting, a case selector, oracle or hardware controller.
Initial R134 action retained; all subsequent real-state feedback is causal.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_success_selector_r134 as selector
from diagnostics import train_getup_local_hip_r130 as local
from diagnostics.getup_independent_native import digest

ROOT=local.ROOT
OUTPUT=ROOT/'outputs/getup_dynamic_gain_r147_left_20261006'
MODEL=None


def state_features(current,nominal,ids):
    current=np.asarray(current,dtype=np.float32);nominal=np.asarray(nominal,dtype=np.float32)
    ids=np.asarray(ids,dtype=int)
    if current.shape!=(55,) or nominal.shape!=(55,) or ids.shape!=(3,) or len(set(ids))!=3 or np.any(ids<0) or np.any(ids>=14):
        raise ValueError('Actual sensor55, reference sensor55 and three joint indices required')
    if not np.isfinite(current).all() or not np.isfinite(nominal).all():raise ValueError('Finite sensors required')
    difference=(current-nominal).astype(float)
    # gyro rad/s, up, three hip position offsets, native velocity*.05.
    return np.concatenate((difference[:3],difference[3:6]/.05,
        difference[6+ids]/.05,difference[20+ids]/.05))


def dynamic_gains(base,parameters,current,nominal,initial,nominal_initial,ids):
    base=np.asarray(base,dtype=float);parameters=np.asarray(parameters,dtype=float)
    if base.shape!=(6,) or parameters.shape!=(18,) or not np.isfinite(base).all() or not np.isfinite(parameters).all():
        raise ValueError('Six base gains and eighteen finite parameters required')
    if np.abs(base).max()>2 or np.abs(parameters[:6]).max()>1 or np.abs(parameters[6:]).max()>2:
        raise ValueError('Declared adaptation limits exceeded')
    delta=state_features(current,nominal,ids)-state_features(initial,nominal_initial,ids)
    # No state adjustment at initial control. Scalar nominal exactness and
    # zero-parameter execution preserve the original base vector exactly.
    if not np.any(parameters[:6]) or not np.any(parameters[6:]) or not np.any(delta):
        return base.copy(),0.
    activation=float(np.tanh(parameters[6:]@np.tanh(delta)))
    return np.clip(base+parameters[:6]*activation,-2.,2.),activation


def init_worker(model_file):
    global MODEL
    local.init_worker()
    with np.load(model_file,allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}


def evaluate(job):
    parameters,case,full,directory,parity=job
    assert case in {None,*local.prior.program.TRAIN}
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,247,full)
    env.reset(case);sim=env.sim
    base,choice,logits=selector.predict(MODEL,env.preparation_sensors[-1])
    initial=env.observe().copy()
    actuator_ids=np.array([sim.model.actuator(n).id for n in local.JOINTS])
    sensor_ids=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in local.JOINTS])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    original=sim.step_target;extra=[];gain_history=[];activation_history=[];combined=[]
    def controlled(target):
        k=env.controls
        if k<529:
            gains,activation=dynamic_gains(base,parameters,env.observe(),local.NOMINAL[k],initial,local.NOMINAL[0],sensor_ids)
            feedback=local.local_feedback(env.observe(),local.NOMINAL[k],sensor_ids,gains)
        else:gains=base.copy();activation=0.;feedback=np.zeros(3)
        adjusted=local.merge_target(target,env.targets[k],feedback,actuator_ids,sim.lower,sim.upper)
        combined.append(float(np.abs(adjusted[actuator_ids]-env.targets[k][actuator_ids]).max()))
        assert combined[-1]<=.18+1e-12
        if k==0:np.testing.assert_array_equal(gains,base);assert activation==0.
        extra.append(feedback.copy());gain_history.append(gains.copy());activation_history.append(activation)
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
        maximum_combined_right_hip_correction_rad=max(combined),root_edits_during_recovery=0,
        no_case_metadata_in_controller=True)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],
        qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],
        local_hip_extra_rad=extra,preparation_sensors=env.preparation_sensors,
        dynamic_gains=gain_history,dynamic_activation=activation_history)
    old=selector.OUTPUT/'candidate'/f'case_{case}'
    assert row['initial_hash']==json.loads((old/'result.json').read_text())['initial_hash']
    with np.load(old/'trajectory.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.array(arrays['applied'])[0],z['applied'][0])
        if parity or case is None:
            equal={k:bool(np.array_equal(v,z[k][:len(records)])) for k,v in arrays.items() if k not in ('dynamic_gains','dynamic_activation','preparation_sensors')}
            np.testing.assert_array_equal(env.preparation_sensors,z['preparation_sensors'])
            assert all(equal.values()),equal
            row['frozen_R134_trace_bitwise_equal']=equal
    if case is None:
        np.testing.assert_array_equal(activation_history,np.zeros(len(records)))
        np.testing.assert_array_equal(extra,np.zeros((len(records),3)))
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row


def group(pool,parameters,cases,full,path,parity=False):
    rows=list(pool.map(evaluate,[(parameters.tolist(),c,full,str(path/f'case_{c}'),parity) for c in cases]))
    report=local.aggregate(rows);local.write_json(path/'results.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true')
    p.add_argument('--workers',type=int,default=6);p.add_argument('--generations',type=int,default=8);p.add_argument('--population',type=int,default=12)
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve()
    assert 1<=args.workers<=6 and 1<=args.generations<=8 and 6<=args.population<=12
    assert json.loads((ROOT/'outputs/getup_dynamic_feedback_audit_r146_20261006/results.json').read_text())['failed_cases']==[769002,769004,773001,773004,773005,773007,773015]
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    names=[Path(__file__).name,'test_getup_dynamic_gain_r147.py','launch_getup_dynamic_gain_r147.py','audit_getup_dynamic_feedback_r146.py',
        'train_getup_success_selector_r134.py','train_getup_local_hip_r130.py','train_getup_history_program_r122.py',
        'probe_getup_sensor_history_r121.py','train_getup_program_r113.py','getup_reference_env_r100.py',
        'getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py']
    for name in names:shutil.copy2(Path(__file__).with_name(name),sources/name)
    frozen=args.output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),
                        (local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),
                        (local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_trajectory.npz')]:
        shutil.copy2(source,frozen/name);assert digest(source)==digest(frozen/name)
    model_file=frozen/'initial_selector.npz'
    local.write_json(args.output/'contract.json',dict(seed=247,smoke=args.smoke,workers=args.workers,generations=args.generations,population=args.population,
        method='Direct CEM of rank-one current-state-change gain field on frozen R134, not end-to-end neural retraining',
        hypothesis='R146 verifies wrong initial choices and early divergent real sensor dynamics; preserving the original first action then adapting subsequent feedback may rescue trajectories, but this has not been proven',
        inputs='Current actual gyro/up/three hip positions and velocities, their initial causal errors, same-phase nominal sensors; fixed frozen R134 initial native50 selector',
        direction_limit=1.,slope_limit=2.,gains_limit=2.,rank_one_parameters=18,
        first_control_exact_R134=True,standard_scalar_activation_exact_zero=True,
        no_lookup_no_case_no_root_truth_no_future_sensor_input=True,
        no_delayed_common_prefix=True,no_geometric_risk_thresholds=True,
        no_mid_episode_root_edits=True,combined_correction_rad=.18,home_original=True,
        full_controls=2279,training_controls=629,short_tail_1s_not_acceptance=True,
        control_hz=50,physics_hz=500,entry_deadline_s=12,strict_standing_tail_s=30,
        physical_limits_rewards_and_acceptance_unchanged=True,independent_qualification_never_loaded=True,
        training_cases=[None,*local.prior.program.TRAIN],automatic_expanded_or_qualification=False,
        hardware_readiness=False,full_task_completed=False,
        hashes={str(p):digest(p) for p in [local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE,*sources.iterdir(),*frozen.iterdir()]}))
    zero=np.zeros(18);selected=zero.copy();mean=zero.copy();std=np.r_[np.full(6,.12),np.full(12,.4)]
    lower=np.r_[np.full(6,-1.),np.full(12,-2.)];upper=-lower;rng=np.random.default_rng(247)
    history=[];cache={};best=None;cases=[None,*local.prior.program.TRAIN]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(model_file,)) as pool:
        parity=group(pool,zero,[None,769000,773004],True,args.output/'zero_parity',True)
        probe=np.r_[np.array([.1,-.1,.1,.05,-.05,.05]),np.full(12,.2)]
        nonzero=group(pool,probe,[None,769002,773004],True,args.output/'nonzero_smoke')
        print('R147_ZERO_AND_NOMINAL_PARITY_PASS',flush=True)
        if args.smoke:
            local.write_json(args.output/'results.json',dict(smoke=True,parity=parity,nonzero=nonzero,full_task_completed=False));return
        for generation in range(1,args.generations+1):
            proposals=[zero.copy(),selected.copy(),*np.clip(rng.normal(mean,std,(args.population-2,18)),lower,upper)]
            reports=[];locations=[]
            for index,parameters in enumerate(proposals):
                key=parameters.tobytes().hex();folder=args.output/'training'/f'generation_{generation:04d}'/f'candidate_{index:02d}'
                if key not in cache:cache[key]=(group(pool,parameters,cases,False,folder,not np.any(parameters)),str(folder))
                report,location=cache[key];reports.append(report);locations.append(location)
                print('R147_CANDIDATE',generation,index,report['successes'],report['physical_failures'],flush=True)
            ordered=sorted(range(len(proposals)),key=lambda i:local.rank(reports[i]),reverse=True);winner=ordered[0]
            if best is None or local.rank(reports[winner])>local.rank(best):selected=proposals[winner].copy();best=reports[winner]
            elite=np.stack([proposals[i] for i in ordered[:3]])
            mean=.6*mean+.4*elite.mean(0)
            std=np.clip(.6*std+.4*elite.std(0),np.r_[np.full(6,.015),np.full(12,.04)],np.r_[np.full(6,.25),np.full(12,.6)])
            history.append(dict(generation=generation,proposals=[v.tolist() for v in proposals],reports=reports,closed_trial_directories=locations,
                selected_parameters=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
            checkpoint=args.output/'checkpoints'/f'generation_{generation:04d}';checkpoint.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(checkpoint/'state.npz',parameters=selected,mean=mean,std=std)
            local.write_json(checkpoint/'rng.json',rng.bit_generator.state);local.write_json(checkpoint/'history.json',history)
            local.write_json(args.output/'progress.json',dict(closed_generation=generation,generations=args.generations,best=best,parameters=selected.tolist()))
        local.write_json(args.output/'training_closed.json',dict(history=history,rng=rng.bit_generator.state,best=best,parameters=selected.tolist(),short_label_only=True))
        candidate=group(pool,selected,cases,True,args.output/'development_candidate')
        baseline=group(pool,zero,cases,True,args.output/'development_baseline',True)
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and candidate['physical_failures']==0)
        local.write_json(args.output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,parameters=selected.tolist(),
            original_development_gate=gate,expanded_development_run=False,independent_qualification_run=False,
            full_task_completed=False,hardware_readiness=False))
        print('R147_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)


if __name__=='__main__':main()
