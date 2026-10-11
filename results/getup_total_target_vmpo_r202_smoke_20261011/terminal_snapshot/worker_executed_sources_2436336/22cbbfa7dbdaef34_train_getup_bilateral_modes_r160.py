"""Bounded bilateral common/differential causal joint-error feedback on R157.

Modes are actuator-coordinate combinations, not claims about world momentum.
No root truth, case/seed lookup, future sensors or altered physical limits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_set_decision_r157 as prior
from diagnostics.getup_independent_native import digest

ROOT=prior.ROOT
OUTPUT=ROOT/'outputs/getup_bilateral_modes_r160_left_20261006'
SMOKE=ROOT/'outputs/getup_bilateral_modes_r160_smoke_20261006'
JOINTS=('left_hip_yaw','left_hip_roll','left_hip_pitch','right_hip_yaw','right_hip_roll','right_hip_pitch')
PROBE=np.array([.005,.004,.003,.002,.003,.004,.003,.004,.005,.004,.003,.002])
MODEL=None

def bilateral_features(current,nominal,ids):
    current=np.asarray(current,dtype=np.float32);nominal=np.asarray(nominal,dtype=np.float32);ids=np.asarray(ids,dtype=int)
    if current.shape!=(55,) or nominal.shape!=(55,) or ids.shape!=(6,):raise ValueError('Native55 and six left/right hip indices required')
    if not np.isfinite(current).all() or not np.isfinite(nominal).all() or len(set(ids))!=6 or np.any(ids<0) or np.any(ids>=14):raise ValueError('Finite sensors and distinct native indices required')
    position=(current[6+ids]-nominal[6+ids]).astype(float)/.05
    velocity=(current[20+ids]-nominal[20+ids]).astype(float)/.05
    return np.concatenate([(position[:3]+position[3:])/2,(velocity[:3]+velocity[3:])/2,
        (position[3:]-position[:3])/2,(velocity[3:]-velocity[:3])/2])

def bilateral_feedback(parameters,current,nominal,initial,nominal_initial,ids):
    parameters=np.asarray(parameters,dtype=float)
    if parameters.shape!=(12,) or not np.isfinite(parameters).all() or np.abs(parameters).max()>1.:raise ValueError('Twelve finite coefficients bounded by one required')
    delta=bilateral_features(current,nominal,ids)-bilateral_features(initial,nominal_initial,ids);activation=np.tanh(delta)
    if not np.any(parameters) or not np.any(delta):return np.zeros(6),activation
    common=-.18*np.tanh(parameters[:3]*activation[:3]+parameters[3:6]*activation[3:6])
    differential=-.18*np.tanh(parameters[6:9]*activation[6:9]+parameters[9:]*activation[9:])
    return np.clip(np.concatenate([common-differential,common+differential]),-.18,.18),activation

def init_worker(file):
    global MODEL
    prior.old.local.init_worker()
    with np.load(file,allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}

def evaluate(job):
    parameters,case,full,directory,parity=job;local=prior.old.local
    assert case in {None,*local.prior.program.TRAIN}
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,260,full)
    env.reset(case);sim=env.sim
    base,choice,logits=prior.old.predict(MODEL,env.preparation_sensors[-1]);initial=env.observe().copy()
    actuator_ids=np.array([sim.model.actuator(n).id for n in JOINTS])
    sensor_ids=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in JOINTS]);np.testing.assert_array_equal(sensor_ids,[0,1,2,9,10,11])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    original=sim.step_target;frozen_history=[];extras=[];activations=[];combined=[]
    def controlled(target):
        k=env.controls
        if k<529:
            observed=env.observe()
            frozen=local.local_feedback(observed,local.NOMINAL[k],sensor_ids[3:],base)
            extra,activation=bilateral_feedback(parameters,observed,local.NOMINAL[k],initial,local.NOMINAL[0],sensor_ids)
        else:frozen=np.zeros(3);extra=np.zeros(6);activation=np.zeros(12)
        frozen_target=local.merge_target(target,env.targets[k],frozen,actuator_ids[3:],sim.lower,sim.upper)
        adjusted=local.merge_target(frozen_target,env.targets[k],extra,actuator_ids,sim.lower,sim.upper)
        combined.append(float(np.abs(adjusted[actuator_ids]-env.targets[k][actuator_ids]).max()));assert combined[-1]<=.18+1e-12
        if k==0:assert not np.any(extra) and not np.any(activation);np.testing.assert_array_equal(adjusted,frozen_target)
        frozen_history.append(frozen.copy());extras.append(extra.copy());activations.append(activation.copy())
        return original(adjusted)
    sim.step_target=controlled;records=[]
    try:
        while True:
            observed=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,observed,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((observed.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    finally:sim.step_target=original
    row=step[5];row.update(parameters=list(parameters),full_path=full,base_choice=choice,base_gains=base.tolist(),initial_sensor_logits=logits.tolist(),
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8) if full else bool(row['training_success']),
        maximum_combined_six_hip_correction_rad=max(combined),root_edits_during_recovery=0,no_case_metadata_in_controller=True)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],
        qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],
        local_hip_extra_rad=frozen_history,preparation_sensors=env.preparation_sensors,bilateral_extra_rad=extras,bilateral_activation=activations)
    source=prior.OUTPUT/'candidate'/f'case_{case}';saved=json.loads((source/'result.json').read_text());assert row['initial_hash']==saved['initial_hash']
    with np.load(source/'trajectory.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.array(arrays['applied'])[0],z['applied'][0])
        if parity or case is None:
            equal={k:bool(np.array_equal(np.asarray(v),z[k][:len(records)])) for k,v in arrays.items() if k not in ('bilateral_extra_rad','bilateral_activation')}
            assert all(equal.values()),equal
            if full:assert row['peaks']==saved['peaks']
            row['frozen_R157_trace_bitwise_equal']=equal
    if case is None:
        np.testing.assert_array_equal(activations,np.zeros((len(records),12)));np.testing.assert_array_equal(extras,np.zeros((len(records),6)))
        np.testing.assert_array_equal(frozen_history,np.zeros((len(records),3)))
    elif np.any(parameters):
        row['actual_left_extra_nonzero']=bool(np.any(np.array(extras)[:,:3]));row['actual_right_extra_nonzero']=bool(np.any(np.array(extras)[:,3:]))
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row

def group(pool,parameters,cases,full,path,parity=False):
    report=prior.old.local.aggregate(list(pool.map(evaluate,[(parameters.tolist(),c,full,str(path/f'case_{c}'),parity) for c in cases])))
    prior.old.local.write_json(path/'results.json',report);return report

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    assert args.output.parent.resolve()==(ROOT/'outputs').resolve()
    audit=json.loads((ROOT/'outputs/getup_bilateral_coupling_r159_20261006/results.json').read_text());assert audit['read_only'] and len(audit['rows'])==6
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for name in [Path(__file__).name,'test_getup_bilateral_modes_r160.py','launch_getup_bilateral_modes_r160.py','audit_getup_bilateral_coupling_r159.py',
        'train_getup_set_decision_r157.py','train_getup_success_selector_r134.py','train_getup_local_hip_r130.py','train_getup_history_program_r122.py',
        'probe_getup_sensor_history_r121.py','train_getup_program_r113.py','getup_reference_env_r100.py','getup_independent_native.py',
        'train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py']:
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    frozen=args.output/'frozen';frozen.mkdir()
    for source,name in [(prior.OUTPUT/'training/model.npz','initial_selector.npz'),(prior.old.local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),
        (prior.old.local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:
        shutil.copy2(source,frozen/name);assert digest(source)==digest(frozen/name)
    file=frozen/'initial_selector.npz';cases=[None,*prior.old.local.prior.program.TRAIN]
    prior.old.local.write_json(args.output/'contract.json',dict(seed=260,smoke=args.smoke,generations=6,population=12,workers=6,coefficients=12,
        method='CEM of bilateral actuator-coordinate common/differential joint-error-change residuals on frozen R157',
        hypothesis='R159 eight complete valid alternatives directly change right-only extra targets but subsequent original left targets also diverge. Test coupled bilateral response, not a proved unique cause or improvement.',
        joints=JOINTS,position_scale_rad=.05,native_velocity_decode_divisor=.05,coefficient_limit=1.,
        no_world_momentum_conservation_claim=True,inputs='Actual current and causal initial six hip position/velocity sensors minus same-phase nominal sensors; frozen actual initial native50 selector',
        no_root_truth_no_case_lookup_no_future_sensor_input=True,first_target_exact_R157=True,standard_actual_scalar_extra_zero=True,home_original=True,
        combined_original_plus_new_per_joint_rad=.18,training_controls=629,full_controls=2279,control_hz=50,physics_hz=500,
        entry_deadline_s=12,strict_standing_tail_s=30,short_tail_1s_not_acceptance=True,physics_rewards_acceptance_unchanged=True,
        training_cases=cases,automatic_expanded_or_qualification=False,reserved_qualification_never_loaded=True,no_mid_episode_root_edits=True,
        full_task_completed=False,hardware_readiness=False,hashes={str(q):digest(q) for q in [prior.old.local.prior.program.SCENE,prior.old.local.prior.program.STAND,
            prior.old.local.prior.program.REFERENCE,*sources.iterdir(),*frozen.iterdir()]}))
    zero=np.zeros(12);selected=zero.copy();mean=zero.copy();std=np.full(12,.01);rng=np.random.default_rng(260);history=[];cache={};best=None
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(file,)) as pool:
        parity=group(pool,zero,[None,769000,773004],True,args.output/'zero_parity',True)
        nonzero=group(pool,PROBE,[None,769002,773004],True,args.output/'nonzero_smoke')
        assert all(r['actual_left_extra_nonzero'] and r['actual_right_extra_nonzero'] for r in nonzero['rows'] if r['case_seed'] is not None)
        prior.old.local.write_json(args.output/'startup_closed.json',dict(parity=parity,nonzero=nonzero,terminal_result_saved=False))
        print('R160_ZERO_AND_NOMINAL_FULL_PARITY_PASS',flush=True)
        if args.smoke:
            prior.old.local.write_json(args.output/'results.json',dict(smoke=True,parity=parity,nonzero=nonzero,full_task_completed=False));return
        for generation in range(1,7):
            proposals=[zero.copy(),selected.copy(),*np.clip(rng.normal(mean,std,(10,12)),-1.,1.)];reports=[];locations=[]
            for i,parameters in enumerate(proposals):
                key=parameters.tobytes().hex();path=args.output/'training'/f'generation_{generation:04d}'/f'candidate_{i:02d}'
                if key not in cache:cache[key]=(group(pool,parameters,cases,False,path,not np.any(parameters)),str(path))
                report,location=cache[key];reports.append(report);locations.append(location)
                print('R160_CANDIDATE',generation,i,report['successes'],report['physical_failures'],flush=True)
            order=sorted(range(len(proposals)),key=lambda i:prior.old.local.rank(reports[i]),reverse=True);winner=order[0]
            if best is None or prior.old.local.rank(reports[winner])>prior.old.local.rank(best):selected=proposals[winner].copy();best=reports[winner]
            elite=np.stack([proposals[i] for i in order[:3]]);mean=.6*mean+.4*elite.mean(0);std=np.clip(.6*std+.4*elite.std(0),.002,.05)
            history.append(dict(generation=generation,proposals=[v.tolist() for v in proposals],reports=reports,closed_trial_directories=locations,
                selected_parameters=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
            checkpoint=args.output/'checkpoints'/f'generation_{generation:04d}';checkpoint.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(checkpoint/'state.npz',parameters=selected,mean=mean,std=std);prior.old.local.write_json(checkpoint/'rng.json',rng.bit_generator.state)
            prior.old.local.write_json(checkpoint/'history.json',history);prior.old.local.write_json(args.output/'progress.json',dict(closed_generation=generation,generations=6,best=best,parameters=selected.tolist()))
        prior.old.local.write_json(args.output/'training_closed.json',dict(history=history,rng=rng.bit_generator.state,best=best,parameters=selected.tolist(),short_label_only=True))
        candidate=group(pool,selected,cases,True,args.output/'development_candidate');baseline=group(pool,zero,cases,True,args.output/'development_baseline',True)
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
        prior.old.local.write_json(args.output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,parameters=selected.tolist(),original_development_gate=gate,
            expanded_development_run=False,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
        print('R160_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)

if __name__=='__main__':main()
