"""One fixed causal local phase coordinate rule, simulation only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import traceback
import numpy as np
from diagnostics import train_getup_full_episode_pg_r181 as old

prior=old.prior;local=old.local;selector=old.selector;planner=old.planner;ROOT=old.ROOT
OUTPUT=ROOT/'outputs/getup_phase_coordinate_r185_left_20261010'
SMOKE=ROOT/'outputs/getup_phase_coordinate_r185_smoke_20261010'
CASES=old.CASES;SEED=285;MODEL=TANGENT=REFERENCE=None
RIDGE_RAD=.02;MAX_SHIFT_CONTROLS=1.


def local_phase(current,nominal,initial,initial_nominal,tangent,reference,control,enabled):
    values=[np.asarray(v,dtype=np.float32) for v in (current,nominal,initial,initial_nominal)]
    v=np.asarray(tangent,dtype=float);r=np.asarray(reference,dtype=float)
    if any(x.shape!=(14,) or not np.isfinite(x).all() for x in values):raise ValueError('Finite causal joint14 required')
    if v.shape!=(14,) or not np.isfinite(v).all() or r.ndim!=2 or r.shape[1]!=14 or not np.isfinite(r).all():raise ValueError('Finite fixed tangent and reference required')
    if not isinstance(control,(int,np.integer)) or control<0 or control>=len(r):raise ValueError('Original integer phase required')
    q,n,i,z=[x.astype(float) for x in values]
    delta=(q-n)-(i-z)
    offset=float(np.clip(np.dot(v,delta)/(np.dot(v,v)+RIDGE_RAD**2),-MAX_SHIFT_CONTROLS,MAX_SHIFT_CONTROLS)) if enabled and control<529 else 0.
    if not enabled or control>=529:delta=np.zeros(14)
    if offset==0.:return np.zeros(14),offset,delta
    phase=float(np.clip(control+offset,0.,min(528.,len(r)-1)))
    lo=int(np.floor(phase));hi=min(lo+1,len(r)-1);fraction=phase-lo
    shifted=(1.-fraction)*r[lo]+fraction*r[hi]
    return shifted-r[control],offset,delta


def init_worker(frozen):
    global MODEL,TANGENT,REFERENCE
    path=Path(frozen)
    with np.load(path/'initial_selector.npz',allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}
    with np.load(path/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:local.NOMINAL=z['observations'].copy()
    with np.load(local.prior.program.REFERENCE,allow_pickle=False) as z:REFERENCE=z['targets'].copy()
    TANGENT=np.gradient(local.NOMINAL[:529,6:20].astype(float),axis=0)


def evaluate(job):
    enabled,case,directory,parity=job
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,SEED,True)
    env.reset(case);sim=env.sim;fingerprint=prior.physics_hash(sim)
    ids=np.array([sim.model.actuator(n).id for n in local.JOINTS]);sensors=np.array([9,10,11])
    gains,choice,logits=selector.old.predict(MODEL,env.preparation_sensors[-1])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    original=sim.step_target;records=[];initial=env.observe().copy();last_inputs={};maxima=[]
    keys=('phase_reference_shift_rad','phase_offset_controls','phase_joint_error_change_rad','phase_nominal_tangent_rad_per_control','local_hip_extra_rad','planned_before_integration_rad','same_state_unperturbed_planned_rad','same_state_pre_slew_direct_new_rad','original_double_pre_slew_target_rad','adjusted_double_pre_slew_target_rad')
    signals={k:[] for k in keys}
    def controlled(target):
        k=env.controls;observed=env.observe().copy()
        last_inputs.update(control=np.asarray(k),current_native55=observed.copy(),initial_native55=initial.copy(),original_requested_target=target.copy(),previous_applied=sim.prev.copy(),original_reference=env.targets[k].copy())
        np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
        frozen=local.local_feedback(observed,local.NOMINAL[k],sensors,gains) if k<529 else np.zeros(3)
        fixed=local.merge_target(target,env.targets[k],frozen,ids,sim.lower,sim.upper)
        tangent=TANGENT[k] if k<529 else np.zeros(14)
        extra,offset,delta=local_phase(observed[6:20],local.NOMINAL[k,6:20],initial[6:20],local.NOMINAL[0,6:20],tangent,REFERENCE,k,enabled)
        adjusted=local.merge_target(fixed,env.targets[k],extra,np.arange(14),sim.lower,sim.upper)
        base=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
        wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
        assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
        if k==0:np.testing.assert_array_equal(extra,np.zeros(14));assert offset==0. and adjusted is fixed
        for key,value in zip(keys,(extra,offset,delta,tangent,frozen,wanted,base,adjusted-fixed,fixed,adjusted)):signals[key].append(np.asarray(value).copy())
        maxima.append(float(np.abs(adjusted-env.targets[k]).max()))
        applied=original(adjusted);np.testing.assert_array_equal(applied,wanted);np.testing.assert_array_equal(sim.prev,wanted)
        return applied
    sim.step_target=controlled
    try:
        while True:
            obs=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    except Exception:
        dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
        partial=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],initial_sensor=initial,preparation_sensors=env.preparation_sensors,**signals)
        np.savez_compressed(dest/'partial_trajectory.npz',**partial);np.savez_compressed(dest/'failed_causal_inputs.npz',**last_inputs)
        local.write_json(dest/'failure.json',dict(execution_error=True,controls_completed=len(records),terminal_result_saved=False,traceback=traceback.format_exc(),source_sha256=prior.digest(Path(__file__))))
        raise
    finally:sim.step_target=original
    row=step[5]
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],preparation_sensors=env.preparation_sensors,**signals)
    reference=selector.OUTPUT/'candidate'/f'case_{case}';previous=json.loads((reference/'result.json').read_text());assert row['initial_hash']==previous['initial_hash']
    with np.load(reference/'trajectory.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.asarray(arrays['applied'])[0],z['applied'][0])
        if parity or case is None:
            equal={key:bool(np.array_equal(np.asarray(arrays[key]),z[key])) for key in z.files};assert all(equal.values()),equal
            assert row['peaks']==previous['peaks'];row['complete_R157_parity']=equal
    if case is None:
        for key in ('phase_reference_shift_rad','phase_offset_controls','phase_joint_error_change_rad'):np.testing.assert_array_equal(signals[key],np.zeros_like(signals[key]))
    row.update(enabled=enabled,base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),full_path=True,maximum_combined_14_joint_correction_rad=max(maxima),maximum_phase_reference_shift_rad=float(np.abs(signals['phase_reference_shift_rad']).max()),maximum_phase_offset_controls=float(np.abs(signals['phase_offset_controls']).max()),physics_sha256=fingerprint,physics_unchanged=fingerprint==prior.physics_hash(sim),no_root_truth_or_case_metadata_in_controller=True,root_edits_during_recovery=0)
    assert row['physics_unchanged']
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row


def group(pool,enabled,cases,path,parity=False):
    report=local.aggregate(list(pool.map(evaluate,[(enabled,case,str(path/f'case_{case}'),parity) for case in cases])))
    local.write_json(path/'results.json',report);return report


def compare_smoke(output):
    for name,cases in [('zero_parity',[None,769000,773004]),('phase_smoke',[None,769002,773004])]:
        for case in cases:
            fresh=output/name/f'case_{case}';previous=SMOKE/name/f'case_{case}'
            with np.load(fresh/'trajectory.npz',allow_pickle=False) as a,np.load(previous/'trajectory.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            x=json.loads((fresh/'result.json').read_text());y=json.loads((previous/'result.json').read_text());assert x['initial_hash']==y['initial_hash'] and x['peaks']==y['peaks'] and x['success']==y['success']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>12*1024**3
    output.mkdir();src=output/'executed_sources';src.mkdir()
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if name.startswith('diagnostics.') and f and Path(f).suffix=='.py':shutil.copy2(f,src/Path(f).name)
    for name in (Path(__file__).name,'test_getup_phase_coordinate_r185.py','launch_getup_phase_coordinate_r185.py'):shutil.copy2(Path(__file__).with_name(name),src/name)
    frozen=output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:shutil.copy2(source,frozen/name);assert prior.digest(source)==prior.digest(frozen/name)
    local.write_json(output/'contract.json',dict(seed=SEED,simulation_only=True,smoke=args.smoke,controller='One fixed joint-trajectory local phase coordinate, no lookup, free direction/gain fitting or waiting.',ridge_rad=RIDGE_RAD,maximum_phase_shift_controls=MAX_SHIFT_CONTROLS,workers=6,formal_dynamic_attempt_budget=56,independent_smoke_attempts=6,storage_budget_gb=2,reserve_gb=10,learned_parameters=0,search_trials=0,no_learning_or_new_reward=True,nominal_first_home_exact_original=True,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,hashes={str(f):prior.digest(f) for f in [*src.iterdir(),*frozen.iterdir(),local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE]}))
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
        parity=group(pool,False,[None,769000,773004],output/'zero_parity',True)
        phase=group(pool,True,[None,769002,773004],output/'phase_smoke')
        if not args.smoke:compare_smoke(output)
        local.write_json(output/'startup_closed.json',dict(parity=parity,phase=phase,independent_smoke_bitwise_equal=not args.smoke,terminal_result_saved=False,training_generations_saved=0))
        if args.smoke:
            local.write_json(output/'results.json',dict(smoke=True,parity=parity,phase=phase,terminal_result_saved=True,full_task_completed=False,hardware_readiness=False));return
        assert shutil.disk_usage(ROOT).free>10*1024**3
        baseline=group(pool,False,CASES,output/'development_baseline',True)
        candidate=group(pool,True,CASES,output/'development_candidate')
        assert [r['initial_hash'] for r in baseline['rows']]==[r['initial_hash'] for r in candidate['rows']]
        local.write_json(output/'training_closed.json',dict(fixed_controller=True,learned_parameters=0,search_trials=0,formal_dynamic_attempts=56,rng=np.random.default_rng(SEED).bit_generator.state))
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
        local.write_json(output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,original_development_gate=gate,terminal_result_saved=True,formal_dynamic_attempts=56,full_task_completed=False,hardware_readiness=False,expanded_development_run=False,independent_qualification_run=False))
        print('R185_PHASE_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)


if __name__=='__main__':main()
