"""Bounded causal executed-target tracking memory with conditional anti-windup.

Not an instantaneous position/velocity gain, phase adjustment or expert mix.
Only controller memory changes; physics, reference, home and acceptance do not.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_velocity_damping_r165 as previous
from diagnostics.getup_independent_native import digest,DT

ROOT=previous.ROOT
local=previous.local
selector=previous.previous.prior
OUTPUT=ROOT/'outputs/getup_tracking_memory_r170_left_20261008'
SMOKE=ROOT/'outputs/getup_tracking_memory_r170_smoke_20261008'
PROBE=np.array([.003,.003])
LEAK=float(np.exp(-DT/1.))
MEMORY_BOUND=1.
MODEL=None

def tracking_memory(parameters,current,nominal,initial,initial_nominal,state,groups):
    p=np.asarray(parameters,dtype=float);z=np.asarray(state,dtype=float);g=np.asarray(groups,dtype=int)
    sensors=[np.asarray(a,dtype=np.float32) for a in (current,nominal,initial,initial_nominal)]
    if p.shape!=(2,) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>.05):raise ValueError('Two bounded nonnegative leg coefficients required')
    if any(a.shape!=(55,) or not np.isfinite(a).all() for a in sensors):raise ValueError('Finite causal sensor55 snapshots required')
    if z.shape!=(14,) or not np.isfinite(z).all() or np.any(np.abs(z)>MEMORY_BOUND):raise ValueError('Bounded14 controller memory required')
    if g.shape!=(14,) or np.any(g<0) or np.any(g>2):raise ValueError('Fixed left/right/head actuator groups required')
    x,n,x0,n0=sensors
    def tracking(a):return a[34:48].astype(float)-a[6:20].astype(float)
    # Both fields are in rad relative to home. The target is actual prior applied
    # sim.prev, not unslewed history[0] and not the target currently being planned.
    delta=(tracking(x)-tracking(n))-(tracking(x0)-tracking(n0))
    drive=np.tanh(delta/.05);drive[g==2]=0.
    leaked=LEAK*z;leaked[g==2]=0.
    proposed=np.clip(leaked+DT*drive,-MEMORY_BOUND,MEMORY_BOUND)
    extra=memory_output(p,proposed,g)
    return extra,proposed,delta,drive,leaked

def memory_output(parameters,state,groups):
    extra=np.zeros(14);active=groups!=2
    if np.any(parameters):extra[active]=.18*np.tanh(parameters[groups[active]]*state[active])
    return extra

def controlled_memory(parameters,current,nominal,initial,initial_nominal,state,groups,target,reference,previous_target,lower,upper):
    extra,proposed,delta,drive,leaked=tracking_memory(parameters,current,nominal,initial,initial_nominal,state,groups)
    active=np.flatnonzero(groups!=2)
    trial=local.merge_target(target,reference,extra[active],active,lower,upper)
    base_applied=previous.apply_limits(target,previous_target,lower,upper)
    trial_applied=previous.apply_limits(trial,previous_target,lower,upper)
    # Reject only the increment pushing in the direction of a blocked correction.
    # Opposite drive can unwind memory. All planning uses current prior target.
    blocked=((extra-(trial_applied-base_applied))*drive>1e-13)&(groups!=2)
    accepted=proposed.copy();accepted[blocked]=leaked[blocked]
    extra=memory_output(np.asarray(parameters,dtype=float),accepted,groups)
    adjusted=local.merge_target(target,reference,extra[active],active,lower,upper)
    assert np.abs(adjusted-reference).max()<=.18+1e-12
    return adjusted,extra,accepted,delta,drive,blocked,proposed

def init_worker(file):
    global MODEL
    local.init_worker()
    with np.load(file,allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}

def evaluate(job):
    parameters,case,directory,parity=job
    assert case in {None,*local.prior.program.TRAIN}
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,270,True)
    env.reset(case);sim=env.sim
    names=[sim.model.actuator(i).name for i in range(sim.model.nu)]
    groups=np.array([0 if n.startswith('left_') else 1 if n.startswith('right_') else 2 for n in names])
    assert len(names)==14 and [int(np.count_nonzero(groups==i)) for i in range(3)]==[5,5,4]
    assert all(n.startswith(('head_','neck_')) for n,g in zip(names,groups) if g==2)
    right_ids=np.array([sim.model.actuator(n).id for n in local.JOINTS])
    sensor_ids=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in local.JOINTS])
    np.testing.assert_array_equal(sensor_ids,[9,10,11])
    base,choice,logits=selector.old.predict(MODEL,env.preparation_sensors[-1])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    initial=env.observe().copy();initial_nominal=local.NOMINAL[0].copy();state=np.zeros(14)
    original=sim.step_target;frozen_history=[];extras=[];states=[];deltas=[];drives=[];blocks=[];proposals=[];direct=[];maxima=[]
    def controlled(target):
        nonlocal state
        k=env.controls
        observed=env.observe()
        # Verify the causal native target field before original step_target updates.
        np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
        if k<529:
            frozen=local.local_feedback(observed,local.NOMINAL[k],sensor_ids,base)
        else:frozen=np.zeros(3)
        fixed=local.merge_target(target,env.targets[k],frozen,right_ids,sim.lower,sim.upper)
        if k<529:
            adjusted,extra,state,delta,drive,blocked,proposed=controlled_memory(parameters,observed,local.NOMINAL[k],initial,initial_nominal,state,groups,fixed,env.targets[k],sim.prev,sim.lower,sim.upper)
        else:
            adjusted=fixed;extra=np.zeros(14);state=np.zeros(14);delta=np.zeros(14);drive=np.zeros(14);blocked=np.zeros(14,dtype=bool);proposed=np.zeros(14)
        if k==0:
            np.testing.assert_array_equal(extra,np.zeros(14));np.testing.assert_array_equal(state,np.zeros(14))
            assert adjusted is fixed
        np.testing.assert_array_equal(extra[groups==2],np.zeros(4))
        frozen_history.append(frozen.copy());extras.append(extra.copy());states.append(state.copy());deltas.append(delta.copy());drives.append(drive.copy());blocks.append(blocked.copy());proposals.append(proposed.copy());direct.append(adjusted-fixed)
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
        maximum_combined_14_joint_correction_rad=max(maxima),maximum_tracking_memory=float(np.abs(states).max()),maximum_tracking_extra_rad=float(np.abs(extras).max()),
        antiwindup_rejected_channel_steps=int(np.count_nonzero(blocks)),groups=groups.tolist(),actuator_names=names,
        no_case_metadata_or_root_truth_in_controller=True,root_edits_during_recovery=0)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],local_hip_extra_rad=frozen_history,preparation_sensors=env.preparation_sensors,
        tracking_memory_extra_rad=extras,tracking_memory_state=states,tracking_error_change_rad=deltas,tracking_memory_drive=drives,antiwindup_rejected=blocks,tracking_memory_proposal=proposals,same_state_pre_slew_direct_new_rad=direct)
    reference=selector.OUTPUT/'candidate'/f'case_{case}';priorrow=json.loads((reference/'result.json').read_text());assert row['initial_hash']==priorrow['initial_hash']
    if parity or case is None:
        with np.load(reference/'trajectory.npz',allow_pickle=False) as z:equal={k:bool(np.array_equal(np.asarray(arrays[k]),z[k])) for k in z.files}
        assert all(equal.values()),equal;assert row['peaks']==priorrow['peaks'];row['complete_R157_parity']=equal
    if case is None:
        for a in (extras,states,deltas,drives,proposals):np.testing.assert_array_equal(a,np.zeros((len(records),14)))
        assert not np.any(blocks)
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row

def group(pool,parameters,cases,path,parity=False):
    report=local.aggregate(list(pool.map(evaluate,[(parameters.tolist(),c,str(path/f'case_{c}'),parity) for c in cases])))
    local.write_json(path/'results.json',report);return report

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and not args.output.exists()
    ar=json.loads((ROOT/'outputs/getup_expert_mixture_terminal_audit_r169_20261008/results.json').read_text())
    assert ar['read_only'] and ar['scalar_trajectories_audited']==625 and ar['source_hashes_unchanged']
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for name in [Path(__file__).name,'test_getup_tracking_memory_r170.py','launch_getup_tracking_memory_r170.py','train_getup_velocity_damping_r165.py','train_getup_bilateral_modes_r160.py','train_getup_set_decision_r157.py','train_getup_success_selector_r134.py','train_getup_local_hip_r130.py','train_getup_history_program_r122.py','probe_getup_sensor_history_r121.py','train_getup_program_r113.py','getup_reference_env_r100.py','getup_fullfallen_env_r32.py','getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py']:
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    frozen=args.output/'frozen';frozen.mkdir()
    for src,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:shutil.copy2(src,frozen/name);assert digest(src)==digest(frozen/name)
    np.savez_compressed(frozen/'memory_contract.npz',leak=LEAK,memory_bound=MEMORY_BOUND,dt=DT,probe=PROBE)
    local.write_json(args.output/'contract.json',dict(seed=270,smoke=args.smoke,generations=3,population=8,workers=6,parameters=2,coefficient_range=[0,.05],
        hypothesis='Instantaneous additions and expert mixing rescued few but regressed many. Test causal executed-target tracking memory, fixed leak and conditional anti-windup, not a demonstrated unique cause or safety proof.',
        differences_from_old='Actual previous executed target minus measured position, initial tracking subtraction, recurrent bounded14 memory; two leg amplitudes; no velocity/IMU gate, expert mixing, phase adjustment, fitting labels or BC; all proposals full original acceptance',
        formula='delta=((actual_prev-home)-(actual_q-home)-(nominal_prev-home)+(nominal_q-home))-causal_initial_corresponding_tracking_error; drive=tanh(delta/.05); z_proposal=clip(exp(-.02/1)*z+.02*drive,-1,1); leg_extra=+.18*tanh(nonnegative_side_coefficient*z); reject increments driving into original cap/joint/slew blockage and retain leaked state',
        memory_leak=LEAK,memory_bound=MEMORY_BOUND,leak_time_constant_s=1.,leak_or_memory_limits_not_searched=True,head_neck_no_new_action=True,
        inputs='Only current and causal initial actual14 joint positions and actual previous executed targets, same-phase nominal snapshots, past controller memory; original R157 selector unchanged',
        controller_never_reads_case_seed_label_directory_root_truth_or_future_sensor=True,no_energy_dissipation_or_safety_claim=True,first_target_exact_original=True,nominal_scalar_extra_and_memory_exact_zero=True,zero_object_identity_preserved=True,home_original=True,
        physics_reward_acceptance_original=True,combined_original_and_new_14_joint_cap_rad=.18,control_hz=50,physics_hz=500,controls=2279,entry_deadline_s=12,strict_tail_s=30,
        training_cases=[None,*local.prior.program.TRAIN],full_path_training_acceptance_not_short_label=True,automatic_expanded_or_qualification=False,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,
        hashes={str(f):digest(f) for f in [*sources.iterdir(),*frozen.iterdir(),local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE]}))
    zero=np.zeros(2);selected=zero.copy();mean=zero.copy();std=np.full(2,.003);rng=np.random.default_rng(270);history=[];cache={};best=None;cases=[None,*local.prior.program.TRAIN]
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen/'initial_selector.npz',)) as pool:
        parity=group(pool,zero,[None,769000,773004],args.output/'zero_parity',True)
        probe=group(pool,PROBE,[None,769002,773004],args.output/'nonzero_smoke')
        assert all(r['maximum_tracking_extra_rad']>0 for r in probe['rows'] if r['case_seed'] is not None),'Independent nonzero smoke must exercise feedback'
        if not args.smoke:
            for name in ('zero_parity','nonzero_smoke'):
                for row in (parity if name=='zero_parity' else probe)['rows']:
                    case=row['case_seed'];fresh=args.output/name/f'case_{case}';old=SMOKE/name/f'case_{case}'
                    with np.load(fresh/'trajectory.npz',allow_pickle=False) as a,np.load(old/'trajectory.npz',allow_pickle=False) as b:
                        assert a.files==b.files
                        for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
                    sr=json.loads((old/'result.json').read_text());assert row['initial_hash']==sr['initial_hash'] and row['peaks']==sr['peaks']
        local.write_json(args.output/'startup_closed.json',dict(parity=parity,nonzero=probe,independent_smoke_bitwise_equal=not args.smoke,terminal_result_saved=False,training_generations_saved=0))
        print('R170_ZERO_NOMINAL_AND_STARTUP_PARITY_PASS',flush=True)
        if args.smoke:local.write_json(args.output/'results.json',dict(smoke=True,parity=parity,nonzero=probe,terminal_result_saved=True,full_task_completed=False));return
        for generation in range(1,4):
            proposals=[zero.copy(),selected.copy(),*np.clip(rng.normal(mean,std,(6,2)),0.,.05)];reports=[];locations=[]
            for i,parameters in enumerate(proposals):
                key=parameters.tobytes().hex();path=args.output/'training'/f'generation_{generation:04d}'/f'candidate_{i:02d}'
                if key not in cache:cache[key]=(group(pool,parameters,cases,path,not np.any(parameters)),str(path))
                report,location=cache[key];reports.append(report);locations.append(location)
                print('R170_FULL_CANDIDATE',generation,i,report['successes'],report['physical_failures'],flush=True)
            order=sorted(range(8),key=lambda i:local.rank(reports[i]),reverse=True);winner=order[0]
            if best is None or local.rank(reports[winner])>local.rank(best):selected=proposals[winner].copy();best=reports[winner]
            elite=np.stack([proposals[i] for i in order[:2]]);mean=.6*mean+.4*elite.mean(0);std=np.clip(.6*std+.4*elite.std(0),.0003,.01)
            history.append(dict(generation=generation,proposals=[p.tolist() for p in proposals],reports=reports,closed_trial_directories=locations,selected_parameters=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
            checkpoint=args.output/'checkpoints'/f'generation_{generation:04d}';checkpoint.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(checkpoint/'state.npz',parameters=selected,mean=mean,std=std);local.write_json(checkpoint/'rng.json',rng.bit_generator.state);local.write_json(checkpoint/'history.json',history)
            local.write_json(args.output/'progress.json',dict(closed_generation=generation,generations=3,best=best,parameters=selected.tolist()))
        local.write_json(args.output/'training_closed.json',dict(history=history,rng=rng.bit_generator.state,best=best,parameters=selected.tolist(),all_training_full_path=True))
        candidate=group(pool,selected,cases,args.output/'development_candidate');baseline=group(pool,zero,cases,args.output/'development_baseline',True)
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
        local.write_json(args.output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,parameters=selected.tolist(),original_development_gate=gate,terminal_result_saved=True,expanded_development_run=False,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
        print('R170_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)

if __name__=='__main__':main()
