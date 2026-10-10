"""Nonnegative full-14-joint velocity-error-opposing target corrections.

Distinct from signed hip error-change gains/residuals: current same-phase
velocity error, no initial-error subtraction, all joints, monotone sign,
and every proposal ranked on complete 45.58 s original acceptance.
This sign property is not proof of total mechanical energy dissipation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_bilateral_modes_r160 as previous
from diagnostics.getup_independent_native import digest, DT, SLEW

ROOT=previous.ROOT
local=previous.prior.old.local
OUTPUT=ROOT/'outputs/getup_velocity_damping_r165_left_20261006'
SMOKE=ROOT/'outputs/getup_velocity_damping_r165_smoke_20261006'
PROBE=np.array([.002,.002,.002])
MODEL=None


def damping_feedback(parameters,current,nominal,groups):
    p=np.asarray(parameters,dtype=float);x=np.asarray(current,dtype=np.float32);n=np.asarray(nominal,dtype=np.float32);g=np.asarray(groups,dtype=int)
    if p.shape!=(3,) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>.05):raise ValueError('Three nonnegative bounded coefficients required')
    if x.shape!=(55,) or n.shape!=(55,) or not np.isfinite(x).all() or not np.isfinite(n).all():raise ValueError('Finite current and same-phase nominal sensor55 required')
    if g.shape!=(14,) or np.any(g<0) or np.any(g>2):raise ValueError('Fourteen fixed group IDs required')
    error=(x[20:34]-n[20:34]).astype(float)/.05
    if not np.any(p):return np.zeros(14),error
    extra=-.18*np.tanh(p[g]*error)
    assert np.all(extra*error<=0.)
    return extra,error


def apply_limits(target,previous_target,lower,upper):
    return np.clip(np.clip(target,lower,upper),previous_target-SLEW*DT,previous_target+SLEW*DT)


def merge_damping(target,reference,extra,error,lower,upper):
    # Original frozen R157 target already respects each reference-relative cap.
    assert np.abs(target-reference).max()<=.18+1e-12
    adjusted=local.merge_target(target,reference,extra,np.arange(14),lower,upper)
    assert np.abs(adjusted-reference).max()<=.18+1e-12
    assert np.all((adjusted-target)*error<=1e-13)
    return adjusted


def init_worker(file):
    global MODEL
    local.init_worker()
    with np.load(file,allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}


def evaluate(job):
    parameters,case,directory,parity=job
    assert case in {None,*local.prior.program.TRAIN}
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,265,True)
    env.reset(case);sim=env.sim
    names=[sim.model.actuator(i).name for i in range(sim.model.nu)];assert len(names)==14
    groups=np.array([0 if n.startswith('left_') else 1 if n.startswith('right_') else 2 for n in names])
    assert [int(np.count_nonzero(groups==i)) for i in range(3)]==[5,5,4]
    assert all(n.startswith(('head_','neck_')) for n,g in zip(names,groups) if g==2)
    right_ids=np.array([sim.model.actuator(n).id for n in local.JOINTS])
    sensor_ids=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in local.JOINTS])
    np.testing.assert_array_equal(sensor_ids,[9,10,11])
    base,choice,logits=previous.prior.old.predict(MODEL,env.preparation_sensors[-1])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    original=sim.step_target;frozen_history=[];extras=[];errors=[];direct=[];direct_slew=[];maxima=[]
    def controlled(target):
        k=env.controls
        if k<529:
            observed=env.observe();frozen=local.local_feedback(observed,local.NOMINAL[k],sensor_ids,base)
            extra,error=damping_feedback(parameters,observed,local.NOMINAL[k],groups)
        else:frozen=np.zeros(3);extra=np.zeros(14);error=np.zeros(14)
        fixed=local.merge_target(target,env.targets[k],frozen,right_ids,sim.lower,sim.upper)
        adjusted=merge_damping(fixed,env.targets[k],extra,error,sim.lower,sim.upper)
        before=apply_limits(fixed,sim.prev,sim.lower,sim.upper);after=apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
        assert np.all((after-before)*error<=1e-13)
        frozen_history.append(frozen.copy());extras.append(extra.copy());errors.append(error.copy());direct.append(adjusted-fixed);direct_slew.append(after-before)
        maxima.append(float(np.abs(adjusted-env.targets[k]).max()))
        return original(adjusted)
    sim.step_target=controlled;records=[]
    try:
        while True:
            obs=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    finally:sim.step_target=original
    row=step[5];row.update(parameters=list(parameters),full_path=True,base_choice=choice,base_gains=base.tolist(),initial_sensor_logits=logits.tolist(),
        success=bool(step[5]['valid'] and step[5]['controls']==2279 and step[5]['entry_time_s'] is not None and step[5]['entry_time_s']<=12 and step[5]['strict_tail_s']>=30-1e-8),
        maximum_combined_14_joint_correction_rad=max(maxima),group_names=['left_five_leg_joints','right_five_leg_joints','four_head_neck_joints'],
        groups=groups.tolist(),actuator_names=names,direct_correction_opposes_current_velocity_error=True,
        no_total_energy_dissipation_or_recovery_guarantee=True,no_case_metadata_in_controller=True,root_edits_during_recovery=0)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],
        qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],local_hip_extra_rad=frozen_history,
        preparation_sensors=env.preparation_sensors,damping_extra_rad=extras,actual_velocity_error_rad_s=errors,
        same_state_pre_slew_direct_new_rad=direct,same_state_post_slew_direct_new_rad=direct_slew)
    reference=previous.prior.OUTPUT/'candidate'/f'case_{case}';priorrow=json.loads((reference/'result.json').read_text());assert row['initial_hash']==priorrow['initial_hash']
    if parity or case is None:
        with np.load(reference/'trajectory.npz',allow_pickle=False) as z:
            equal={k:bool(np.array_equal(np.asarray(arrays[k]),z[k])) for k in z.files}
        assert all(equal.values()),equal;assert row['peaks']==priorrow['peaks'];row['complete_R157_parity']=equal
    if case is None:
        np.testing.assert_array_equal(extras,np.zeros((len(records),14)));np.testing.assert_array_equal(errors,np.zeros((len(records),14)))
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row


def group(pool,parameters,cases,path,parity=False):
    report=local.aggregate(list(pool.map(evaluate,[(parameters.tolist(),c,str(path/f'case_{c}'),parity) for c in cases])))
    local.write_json(path/'results.json',report);return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and not args.output.exists()
    audit=json.loads((ROOT/'outputs/getup_response_stages_r164_20261006/results.json').read_text());assert audit['read_only'] and audit['pairs']==58
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for name in [Path(__file__).name,'test_getup_velocity_damping_r165.py','launch_getup_velocity_damping_r165.py',
        'audit_getup_response_stages_r164.py','train_getup_bilateral_modes_r160.py','train_getup_set_decision_r157.py','train_getup_success_selector_r134.py',
        'train_getup_local_hip_r130.py','train_getup_history_program_r122.py','probe_getup_sensor_history_r121.py','train_getup_program_r113.py',
        'getup_reference_env_r100.py','getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py']:
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    frozen=args.output/'frozen';frozen.mkdir()
    for src,name in [(previous.prior.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),
        (local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:shutil.copy2(src,frozen/name);assert digest(src)==digest(frozen/name)
    local.write_json(args.output/'contract.json',dict(seed=265,smoke=args.smoke,generations=4,population=8,workers=6,parameters=3,coefficient_range=[0,.05],
        hypothesis='R164 early subsequent state/history response far exceeds direct bilateral additions and mostly changes existing IMU corrections. Test nonnegative velocity-error-opposing additions across all joints, not proof of causal instability or damping benefit.',
        differences_from_old='All14 joints; no initial-error subtraction; no position term; nonnegative sign invariant before and after slew; every proposal complete2279 controls, no short-tail ranking',
        formula='extra_j=-.18*tanh(nonnegative_group_coefficient*(actual_native_velocity_j-nominal_native_velocity_j)/.05)',
        inputs='Only current actual14 joint velocities and frozen same-phase nominal velocities; R157 initial actual native50 selector unchanged',
        controller_never_reads_case_seed_label_directory_root_truth_or_future_sensor=True,no_total_energy_dissipation_claim=True,
        first_perturbed_target_may_change_causally=True,nominal_scalar_extra_exact_zero=True,zero_object_identity_preserved=True,home_original=True,
        physics_reward_acceptance_original=True,combined_original_and_new_14_joint_cap_rad=.18,control_hz=50,physics_hz=500,controls=2279,
        entry_deadline_s=12,strict_tail_s=30,training_cases=[None,*local.prior.program.TRAIN],full_path_training_acceptance_not_short_label=True,
        automatic_expanded_or_qualification=False,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,
        hashes={str(f):digest(f) for f in [*sources.iterdir(),*frozen.iterdir(),local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE]}))
    zero=np.zeros(3);selected=zero.copy();mean=zero.copy();std=np.full(3,.0025);rng=np.random.default_rng(265);history=[];cache={};best=None
    cases=[None,*local.prior.program.TRAIN]
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen/'initial_selector.npz',)) as pool:
        parity=group(pool,zero,[None,769000,773004],args.output/'zero_parity',True)
        probe=group(pool,PROBE,[None,769002,773004],args.output/'nonzero_smoke')
        local.write_json(args.output/'startup_closed.json',dict(parity=parity,nonzero=probe,terminal_result_saved=False))
        print('R165_ZERO_AND_NOMINAL_PARITY_PASS',flush=True)
        if args.smoke:local.write_json(args.output/'results.json',dict(smoke=True,parity=parity,nonzero=probe,full_task_completed=False));return
        for generation in range(1,5):
            proposals=[zero.copy(),selected.copy(),*np.clip(rng.normal(mean,std,(6,3)),0.,.05)];reports=[];locations=[]
            for i,parameters in enumerate(proposals):
                key=parameters.tobytes().hex();path=args.output/'training'/f'generation_{generation:04d}'/f'candidate_{i:02d}'
                if key not in cache:cache[key]=(group(pool,parameters,cases,path,not np.any(parameters)),str(path))
                report,location=cache[key];reports.append(report);locations.append(location)
                print('R165_FULL_CANDIDATE',generation,i,report['successes'],report['physical_failures'],flush=True)
            order=sorted(range(8),key=lambda i:local.rank(reports[i]),reverse=True);winner=order[0]
            if best is None or local.rank(reports[winner])>local.rank(best):selected=proposals[winner].copy();best=reports[winner]
            elite=np.stack([proposals[i] for i in order[:2]]);mean=.6*mean+.4*elite.mean(0);std=np.clip(.6*std+.4*elite.std(0),.00025,.01)
            history.append(dict(generation=generation,proposals=[p.tolist() for p in proposals],reports=reports,closed_trial_directories=locations,
                selected_parameters=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
            checkpoint=args.output/'checkpoints'/f'generation_{generation:04d}';checkpoint.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(checkpoint/'state.npz',parameters=selected,mean=mean,std=std);local.write_json(checkpoint/'rng.json',rng.bit_generator.state)
            local.write_json(checkpoint/'history.json',history);local.write_json(args.output/'progress.json',dict(closed_generation=generation,generations=4,best=best,parameters=selected.tolist()))
        local.write_json(args.output/'training_closed.json',dict(history=history,rng=rng.bit_generator.state,best=best,parameters=selected.tolist(),all_training_full_path=True))
        candidate=group(pool,selected,cases,args.output/'development_candidate');baseline=group(pool,zero,cases,args.output/'development_baseline',True)
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
        local.write_json(args.output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,parameters=selected.tolist(),original_development_gate=gate,
            expanded_development_run=False,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
        print('R165_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)

if __name__=='__main__':main()
