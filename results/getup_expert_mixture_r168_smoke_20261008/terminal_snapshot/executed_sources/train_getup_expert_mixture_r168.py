"""Bounded causal mixture of frozen expert feedback OUTPUTS on R157.

Not gain interpolation, static selector fitting, oracle lookup or hardware.
Off-program expert outputs are not assumed to be recoverable.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_velocity_damping_r165 as prior
from diagnostics.getup_independent_native import digest

ROOT=prior.ROOT
local=prior.local
OUTPUT=ROOT/'outputs/getup_expert_mixture_r168_left_20261008'
SMOKE=ROOT/'outputs/getup_expert_mixture_r168_smoke_20261008'
IDS=np.array([9,10,11])
PROBE=.02*np.stack([np.ones(12),-np.ones(12),np.tile([1.,-1.],6),np.tile([-1.,1.],6)])
MODEL=EXPERTS=None


def features(current,nominal):
    x=np.asarray(current,dtype=np.float32);n=np.asarray(nominal,dtype=np.float32)
    if x.shape!=(55,) or n.shape!=(55,) or not np.isfinite(x).all() or not np.isfinite(n).all():
        raise ValueError('Finite actual and reference sensor55 required')
    d=(x-n).astype(float)
    return np.concatenate((d[:3],d[3:6]/.05,d[6+IDS]/.05,d[20+IDS]/.05))


def mixture_feedback(parameters,current,nominal,initial,initial_nominal,base_gains,expert_gains):
    p=np.asarray(parameters,dtype=float);base=np.asarray(base_gains,dtype=float);experts=np.asarray(expert_gains,dtype=float)
    if p.shape!=(4,12) or not np.isfinite(p).all() or np.abs(p).max()>1:
        raise ValueError('Four by twelve finite bounded mixture coefficients required')
    if base.shape!=(6,) or experts.shape!=(4,6) or not np.isfinite(base).all() or not np.isfinite(experts).all() or np.abs(base).max()>2 or np.abs(experts).max()>2:
        raise ValueError('Frozen bounded base and four expert gain vectors required')
    delta=features(current,nominal)-features(initial,initial_nominal)
    phi=np.tanh(delta)
    base_feedback=local.local_feedback(current,nominal,IDS,base)
    expert_feedback=np.stack([local.local_feedback(current,nominal,IDS,g) for g in experts])
    gates=np.maximum(0.,np.tanh(p@phi))
    weights=np.r_[1.,gates]/(1.+gates.sum())
    # Direct original scalar return avoids zero-interface rounding differences.
    mixed=base_feedback if not np.any(gates) else (base_feedback+(gates[:,None]*expert_feedback).sum(0))/(1.+gates.sum())
    assert np.all(weights>=0) and abs(weights.sum()-1.)<1e-14
    values=np.vstack([base_feedback,expert_feedback])
    assert np.all(mixed>=values.min(0)-1e-14) and np.all(mixed<=values.max(0)+1e-14)
    return mixed,base_feedback,expert_feedback,gates,weights,phi


def init_worker(frozen):
    global MODEL,EXPERTS
    frozen=Path(frozen)
    with np.load(frozen/'initial_selector.npz',allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}
    with np.load(frozen/'snapshot.npz',allow_pickle=False) as z:local.WEIGHTS={k:z[k].copy() for k in z.files}
    with np.load(frozen/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:local.NOMINAL=z['observations'].copy()
    with np.load(frozen/'frozen_programs.npz',allow_pickle=False) as z:EXPERTS=z['gains'].copy()
    assert EXPERTS.shape==(4,6) and np.isfinite(EXPERTS).all() and np.abs(EXPERTS).max()<=2


def evaluate(job):
    parameters,case,directory,parity=job
    parameters=np.asarray(parameters,dtype=float)
    assert case in {None,*local.prior.program.TRAIN}
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,268,True)
    env.reset(case);sim=env.sim
    actuator_ids=np.array([sim.model.actuator(n).id for n in local.JOINTS])
    sensor_ids=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in local.JOINTS])
    np.testing.assert_array_equal(sensor_ids,IDS)
    base,choice,logits=prior.previous.prior.old.predict(MODEL,env.preparation_sensors[-1])
    initial=env.observe().copy()
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    original=sim.step_target
    mixed_history=[];base_history=[];expert_history=[];gate_history=[];weight_history=[];phi_history=[];decisions=[];maxima=[]
    def controlled(target):
        k=env.controls;observed=env.observe().copy()
        if k<529:
            mixed,frozen,experts,gates,weights,phi=mixture_feedback(parameters,observed,local.NOMINAL[k],initial,local.NOMINAL[0],base,EXPERTS)
        else:
            mixed=np.zeros(3);frozen=np.zeros(3);experts=np.zeros((4,3));gates=np.zeros(4);weights=np.r_[1.,np.zeros(4)];phi=np.zeros(12)
        if k==0:
            np.testing.assert_array_equal(mixed,frozen);np.testing.assert_array_equal(phi,np.zeros(12));np.testing.assert_array_equal(gates,np.zeros(4))
        adjusted=local.merge_target(target,env.targets[k],mixed,actuator_ids,sim.lower,sim.upper)
        maxima.append(float(np.abs(adjusted-env.targets[k]).max()));assert maxima[-1]<=.18+1e-12
        decisions.append(observed);mixed_history.append(mixed.copy());base_history.append(frozen.copy());expert_history.append(experts.copy())
        gate_history.append(gates.copy());weight_history.append(weights.copy());phi_history.append(phi.copy())
        return original(adjusted)
    sim.step_target=controlled;records=[]
    try:
        while True:
            obs=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    finally:sim.step_target=original
    np.testing.assert_array_equal(decisions,[r[0] for r in records])
    row=step[5];row.update(parameters=parameters.tolist(),full_path=True,base_choice=choice,base_gains=base.tolist(),initial_sensor_logits=logits.tolist(),
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),
        maximum_combined_14_joint_correction_rad=max(maxima),no_case_metadata_in_controller=True,root_edits_during_recovery=0,
        convex_feedback_output_mixture_not_gain_interpolation=True,no_off_program_recovery_guarantee=True)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],
        qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],local_hip_extra_rad=mixed_history,
        preparation_sensors=env.preparation_sensors,selected_base_feedback_rad=base_history,expert_feedback_rad=expert_history,
        mixture_gates=gate_history,mixture_weights=weight_history,mixture_delta_features=phi_history,
        feedback_difference_from_base_rad=np.asarray(mixed_history)-np.asarray(base_history))
    reference=prior.previous.prior.OUTPUT/'candidate'/f'case_{case}'
    priorrow=json.loads((reference/'result.json').read_text());assert row['initial_hash']==priorrow['initial_hash']
    with np.load(reference/'trajectory.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.asarray(arrays['applied'])[0],z['applied'][0])
        if parity or case is None:
            equal={k:bool(np.array_equal(np.asarray(arrays[k]),z[k])) for k in z.files}
            assert all(equal.values()),equal;assert row['peaks']==priorrow['peaks'];row['complete_R157_parity']=equal
    if case is None:
        np.testing.assert_array_equal(phi_history,np.zeros((len(records),12)))
        np.testing.assert_array_equal(gate_history,np.zeros((len(records),4)))
        np.testing.assert_array_equal(mixed_history,np.zeros((len(records),3)))
        np.testing.assert_array_equal(expert_history,np.zeros((len(records),4,3)))
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row


def group(pool,parameters,cases,path,parity=False):
    report=local.aggregate(list(pool.map(evaluate,[(parameters.tolist(),c,str(path/f'case_{c}'),parity) for c in cases])))
    local.write_json(path/'results.json',report);return report


def compare_smoke(formal):
    for groupname,cases in [('zero_parity',[None,769000,773004]),('nonzero_smoke',[None,769002,773004])]:
        for case in cases:
            a=formal/groupname/f'case_{case}';b=SMOKE/groupname/f'case_{case}'
            with np.load(a/'trajectory.npz',allow_pickle=False) as x,np.load(b/'trajectory.npz',allow_pickle=False) as y:
                assert x.files==y.files
                for k in x.files:np.testing.assert_array_equal(x[k],y[k])
            ra=json.loads((a/'result.json').read_text());rb=json.loads((b/'result.json').read_text())
            assert ra['initial_hash']==rb['initial_hash'] and ra['peaks']==rb['peaks'] and ra['success']==rb['success']


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and not args.output.exists()
    assert json.loads((ROOT/'outputs/getup_expert_directions_r167_20261008/results.json').read_text())['read_only']
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    names=[Path(__file__).name,'test_getup_expert_mixture_r168.py','launch_getup_expert_mixture_r168.py','audit_getup_expert_directions_r167.py',
        'train_getup_velocity_damping_r165.py','train_getup_bilateral_modes_r160.py','train_getup_set_decision_r157.py','train_getup_success_selector_r134.py',
        'train_getup_local_hip_r130.py','train_getup_history_program_r122.py','probe_getup_sensor_history_r121.py','train_getup_program_r113.py',
        'getup_reference_env_r100.py','getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py']
    for name in names:shutil.copy2(Path(__file__).with_name(name),sources/name)
    frozen=args.output/'frozen';frozen.mkdir()
    for src,name in [(prior.previous.prior.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),
        (local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz'),
        (ROOT/'outputs/getup_complementary_feedback_r133_left_20261006/frozen_programs.npz','frozen_programs.npz')]:
        shutil.copy2(src,frozen/name);assert digest(src)==digest(frozen/name)
    hashes={str(f):digest(f) for f in [*sources.iterdir(),*frozen.iterdir(),local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE]}
    local.write_json(args.output/'contract.json',dict(seed=268,smoke=args.smoke,generations=4,population=8,workers=6,parameters=48,coefficient_range=[-1,1],
        hypothesis='R167 same-state expert feedback directions overlap across rescue/regression; test adaptive multiway convex output mixing, not an alignment threshold or assumed off-program teacher recovery.',
        differences_from_old='Four independently state-dependent weights mix post-tanh frozen feedback outputs, not static selector refitting, rank-one gain changes, gain interpolation, additive velocity damping or BC aggregation.',
        formula='phi=tanh(error_now-error_at_control0); gate_i=max(0,tanh(W_i dot phi)); mixed=(base_feedback+sum(gate_i*expert_feedback_i))/(1+sum(gate_i))',
        inputs='Current and causal initial gyro/up/right hip position/velocity native55; same-phase nominal sensors; fixed R157 initial native50 selector',
        controller_never_reads_case_seed_label_directory_root_truth_or_future_sensor=True,first_target_R157_exact=True,nominal_scalar_exact_zero=True,
        convex_feedback_not_recovery_or_contact_safety_proof=True,home_original=True,physics_reward_acceptance_original=True,
        combined_original_and_new_14_joint_cap_rad=.18,control_hz=50,physics_hz=500,controls=2279,entry_deadline_s=12,strict_tail_s=30,
        training_cases=[None,*local.prior.program.TRAIN],full_path_training_acceptance_not_short_label=True,automatic_expanded_or_qualification=False,
        qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,hashes=hashes))
    zero=np.zeros((4,12));selected=zero.copy();mean=zero.copy();std=np.full((4,12),.02);rng=np.random.default_rng(268);history=[];cache={};best=None
    cases=[None,*local.prior.program.TRAIN]
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
        parity=group(pool,zero,[None,769000,773004],args.output/'zero_parity',True)
        probe=group(pool,PROBE,[None,769002,773004],args.output/'nonzero_smoke')
        if not args.smoke:compare_smoke(args.output)
        local.write_json(args.output/'startup_closed.json',dict(parity=parity,nonzero=probe,independent_smoke_bitwise_equal=not args.smoke,terminal_result_saved=False))
        print('R168_ZERO_NOMINAL_FIRST_TARGET_PARITY_PASS',flush=True)
        if args.smoke:
            local.write_json(args.output/'results.json',dict(smoke=True,parity=parity,nonzero=probe,full_task_completed=False));return
        for generation in range(1,5):
            proposals=[zero.copy(),selected.copy(),*np.clip(rng.normal(mean,std,(6,4,12)),-1.,1.)];reports=[];locations=[]
            for i,parameters in enumerate(proposals):
                key=parameters.tobytes().hex();path=args.output/'training'/f'generation_{generation:04d}'/f'candidate_{i:02d}'
                if key not in cache:cache[key]=(group(pool,parameters,cases,path,not np.any(parameters)),str(path))
                report,location=cache[key];reports.append(report);locations.append(location)
                print('R168_FULL_CANDIDATE',generation,i,report['successes'],report['physical_failures'],flush=True)
            order=sorted(range(8),key=lambda i:local.rank(reports[i]),reverse=True);winner=order[0]
            if best is None or local.rank(reports[winner])>local.rank(best):selected=proposals[winner].copy();best=reports[winner]
            elite=np.stack([proposals[i] for i in order[:2]]);mean=.6*mean+.4*elite.mean(0);std=np.clip(.6*std+.4*elite.std(0),.002,.05)
            history.append(dict(generation=generation,proposals=[v.tolist() for v in proposals],reports=reports,closed_trial_directories=locations,
                selected_parameters=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
            checkpoint=args.output/'checkpoints'/f'generation_{generation:04d}';checkpoint.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(checkpoint/'state.npz',parameters=selected,mean=mean,std=std);local.write_json(checkpoint/'rng.json',rng.bit_generator.state)
            local.write_json(checkpoint/'history.json',history);local.write_json(args.output/'progress.json',dict(closed_generation=generation,generations=4,best=best,parameters=selected.tolist()))
        local.write_json(args.output/'training_closed.json',dict(history=history,rng=rng.bit_generator.state,best=best,parameters=selected.tolist(),all_training_full_path=True))
        candidate=group(pool,selected,cases,args.output/'development_candidate');baseline=group(pool,zero,cases,args.output/'development_baseline',True)
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
        assert all(digest(Path(f))==h for f,h in hashes.items())
        local.write_json(args.output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,parameters=selected.tolist(),original_development_gate=gate,
            terminal_result_saved=True,expanded_development_run=False,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
        print('R168_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)

if __name__=='__main__':main()
