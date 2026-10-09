"""Finite body-relative foot task-space feedback; simulation only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import mujoco
import numpy as np
from diagnostics import train_getup_recurrent_readout_r175 as prior

local=prior.local;selector=prior.selector;planner=prior.planner;ROOT=prior.ROOT
OUTPUT=ROOT/'outputs/getup_foot_task_r177_left_20261009'
SMOKE=ROOT/'outputs/getup_foot_task_r177_smoke_20261009'
CASES=prior.CASES
SHAPE=(2,);BOUND=.05;DAMPING_M=.01;PROBE=np.array([.002,.002])
MODEL=None


class FootKinematics:
    """Private canonical configuration built only from measured joint positions.

    Never receives simulation data or actual root state. No dynamics, contacts,
    mesh distances, mj_forward or integration. The model is not mutated.
    """
    def __init__(self,model):
        self.model=model;self.data=mujoco.MjData(model)
        joints=model.actuator_trnid[:,0]
        self.qadr=model.jnt_qposadr[joints].copy();self.vadr=model.jnt_dofadr[joints].copy()
        self.home=model.keyframe('home').qpos[self.qadr].copy()
        self.canonical=model.keyframe('home').qpos.copy();self.canonical[:7]=[0,0,0,1,0,0,0]
        self.trunk=model.body('trunk_assembly').id
        self.feet=[model.body(name).id for name in ('foot_assembly','foot_assembly_2')]
        self.groups=[np.array([model.actuator(side+'_'+part).id for part in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]) for side in ('left','right')]
        np.testing.assert_array_equal(self.groups[0],[0,1,2,3,4]);np.testing.assert_array_equal(self.groups[1],[9,10,11,12,13])
        self.jp=np.zeros((3,model.nv));self.jr=np.zeros_like(self.jp);self.jt=np.zeros_like(self.jp);self.rt=np.zeros_like(self.jp)

    def measure(self,joint_relative,with_jacobian=False):
        x=np.asarray(joint_relative,dtype=np.float32)
        if x.shape!=(14,) or not np.isfinite(x).all():raise ValueError('Finite actual14 home-relative positions required')
        self.data.qpos[:]=self.canonical
        # Only leg measurements enter the private leg task geometry. Freeze
        # head joints at home so subtree-COM roundoff cannot affect leg Jacobians.
        self.data.qpos[self.qadr]=self.home
        for group in self.groups:self.data.qpos[self.qadr[group]]=self.home[group]+x[group].astype(float)
        mujoco.mj_kinematics(self.model,self.data);mujoco.mj_comPos(self.model,self.data)
        rotation=self.data.xmat[self.trunk].reshape(3,3).copy()
        positions=np.array([rotation.T@(self.data.xpos[foot]-self.data.xpos[self.trunk]) for foot in self.feet])
        if not with_jacobian:return positions
        mujoco.mj_jacBody(self.model,self.data,self.jt,self.rt,self.trunk)
        jacobians=[]
        for foot,group in zip(self.feet,self.groups):
            mujoco.mj_jacBody(self.model,self.data,self.jp,self.jr,foot)
            # Leg joints cannot rotate or translate their ancestor trunk.
            np.testing.assert_array_equal(self.rt[:,self.vadr[group]],np.zeros((3,5)))
            jacobians.append(rotation.T@(self.jp-self.jt)[:,self.vadr[group]])
        return positions,np.array(jacobians)


def foot_feedback(parameters,current,nominal,initial,initial_nominal,kinematics):
    c=np.asarray(parameters,dtype=float)
    xs=[np.asarray(x,dtype=np.float32) for x in (current,nominal,initial,initial_nominal)]
    if c.shape!=SHAPE or not np.isfinite(c).all() or np.any(c<0) or np.any(c>BOUND):raise ValueError('Two bounded nonnegative task amplitudes required')
    if any(x.shape!=(55,) or not np.isfinite(x).all() for x in xs):raise ValueError('Finite causal sensor55 required')
    x,n,x0,n0=xs
    foot,jac=kinematics.measure(x[6:20],True)
    error=(foot-kinematics.measure(n[6:20]))-(kinematics.measure(x0[6:20])-kinematics.measure(n0[6:20]))
    dq=np.array([-j.T@np.linalg.solve(j@j.T+DAMPING_M**2*np.eye(3),e) for j,e in zip(jac,error)])
    extra=np.zeros(14)
    for coefficient,group,correction in zip(c,kinematics.groups,dq):extra[group]=.18*np.tanh(coefficient*correction/.18)
    return extra,error,jac,dq


def init_worker(frozen):
    global MODEL
    path=Path(frozen)
    with np.load(path/'initial_selector.npz',allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}
    with np.load(path/'snapshot.npz',allow_pickle=False) as z:local.WEIGHTS={k:z[k].copy() for k in z.files}
    with np.load(path/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:local.NOMINAL=z['observations'].copy()


def evaluate(job):
    parameters,case,directory,parity=job;c=np.asarray(parameters,dtype=float).reshape(SHAPE)
    assert case in CASES
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,277,True)
    env.reset(case);sim=env.sim;fingerprint=prior.physics_hash(sim);fk=FootKinematics(sim.model)
    ids=np.array([sim.model.actuator(n).id for n in local.JOINTS]);sensors=np.array([9,10,11])
    gains,choice,logits=selector.old.predict(MODEL,env.preparation_sensors[-1])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    initial=env.observe().copy();initial_nominal=local.NOMINAL[0].copy()
    original=sim.step_target;records=[];frozen_history=[];extras=[];errors=[];jacobians=[];corrections=[]
    planned=[];base_planned=[];pre_direct=[];fixed_targets=[];new_targets=[];maxima=[]
    def controlled(target):
        k=env.controls;observed=env.observe().copy()
        np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
        frozen=local.local_feedback(observed,local.NOMINAL[k],sensors,gains) if k<529 else np.zeros(3)
        fixed=local.merge_target(target,env.targets[k],frozen,ids,sim.lower,sim.upper)
        if k<529:
            extra,error,jac,dq=foot_feedback(c,observed,local.NOMINAL[k],initial,initial_nominal,fk)
        else:extra=np.zeros(14);error=np.zeros((2,3));jac=np.zeros((2,3,5));dq=np.zeros((2,5))
        adjusted=local.merge_target(fixed,env.targets[k],extra,np.arange(14),sim.lower,sim.upper)
        assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
        if k==0:
            np.testing.assert_array_equal(extra,np.zeros(14));np.testing.assert_array_equal(error,np.zeros((2,3)));assert adjusted is fixed
        wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
        unperturbed=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
        frozen_history.append(frozen.copy());extras.append(extra.copy());errors.append(error.copy());jacobians.append(jac.copy());corrections.append(dq.copy())
        planned.append(wanted.copy());base_planned.append(unperturbed.copy());pre_direct.append(adjusted-fixed);fixed_targets.append(fixed.copy());new_targets.append(adjusted.copy());maxima.append(float(np.abs(adjusted-env.targets[k]).max()))
        applied=original(adjusted);np.testing.assert_array_equal(applied,wanted);np.testing.assert_array_equal(sim.prev,wanted)
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
        foot_task_extra_rad=extras,causal_foot_task_error_m=errors,foot_task_jacobian_m_per_rad=jacobians,foot_task_dls_direction_rad=corrections,planned_before_integration_rad=planned,same_state_unperturbed_planned_rad=base_planned,
        same_state_pre_slew_direct_new_rad=pre_direct,original_double_pre_slew_target_rad=fixed_targets,adjusted_double_pre_slew_target_rad=new_targets)
    reference=selector.OUTPUT/'candidate'/f'case_{case}';old=json.loads((reference/'result.json').read_text());assert row['initial_hash']==old['initial_hash']
    with np.load(reference/'trajectory.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.asarray(arrays['applied'])[0],z['applied'][0])
        if parity or case is None:
            equal={k:bool(np.array_equal(np.asarray(arrays[k]),z[k])) for k in z.files}
            assert all(equal.values()),equal;assert row['peaks']==old['peaks'];row['complete_R157_parity']=equal
    if case is None:
        for v in (extras,errors,corrections):np.testing.assert_array_equal(v,np.zeros_like(v))
    row.update(parameters=c.tolist(),base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),full_path=True,
        maximum_combined_14_joint_correction_rad=max(maxima),maximum_foot_task_extra_rad=float(np.abs(extras).max()),maximum_foot_error_m=float(np.abs(errors).max()),
        online_planned_target_equals_actual_applied_bitwise=True,physics_sha256=fingerprint,physics_unchanged=fingerprint==prior.physics_hash(sim),root_edits_during_recovery=0,
        actuator_names=[sim.model.actuator(i).name for i in range(14)],no_root_truth_or_case_metadata_in_controller=True,private_canonical_FK_without_dynamics=True)
    assert row['physics_unchanged']
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row


def group(pool,c,cases,path,parity=False):
    report=local.aggregate(list(pool.map(evaluate,[(c.tolist(),case,str(path/f'case_{case}'),parity) for case in cases])))
    local.write_json(path/'results.json',report);return report


def compare_smoke(output):
    for name,cases in [('zero_parity',[None,769000,773004]),('nonzero_smoke',[None,769002,773004])]:
        for case in cases:
            fresh=output/name/f'case_{case}';old=SMOKE/name/f'case_{case}'
            with np.load(fresh/'trajectory.npz',allow_pickle=False) as a,np.load(old/'trajectory.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            x=json.loads((fresh/'result.json').read_text());y=json.loads((old/'result.json').read_text());assert x['initial_hash']==y['initial_hash'] and x['peaks']==y['peaks'] and x['success']==y['success']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>12*1024**3
    audit=json.loads((ROOT/'outputs/getup_recurrent_terminal_audit_r176_20261009/results.json').read_text());assert audit['terminal_result_saved']
    output.mkdir();src=output/'executed_sources';src.mkdir()
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if name.startswith('diagnostics.') and f and Path(f).suffix=='.py':shutil.copy2(f,src/Path(f).name)
    for name in [Path(__file__).name,'test_getup_foot_task_r177.py','launch_getup_foot_task_r177.py']:shutil.copy2(Path(__file__).with_name(name),src/name)
    frozen=output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:
        shutil.copy2(source,frozen/name);assert prior.digest(source)==prior.digest(frozen/name)
    local.write_json(output/'contract.json',dict(seed=277,smoke=args.smoke,simulation_only=True,generations=2,population=8,workers=6,parameters=2,parameter_bound=BOUND,damping_m=DAMPING_M,
        hypothesis='R175 coordinated recurrent readout rescues four but regresses twelve with highest nonzero10/24; test fixed nonlinear kinematic foot placement projection coordinating all five joints of each leg, not another arbitrary sensor readout.',
        formula='e=(FK(q)-FK(q_nominal))-(FK(q_initial)-FK(q_initial_nominal)); dq=-J.T*solve(J*J.T+.01^2*I,e); extra[group]=.18*tanh(c[group]*dq/.18)',
        task='Two foot body origins relative to trunk in trunk frame, meters; not floor height, contact geometry, world position or predicted future state.',
        inputs='Current and causal initial actual14 home-relative positions, frozen same-phase nominal positions, immutable model geometry; other native channels unused by new kernel.',
        private_canonical_root_only=True,no_actual_root_read_or_write=True,no_forward_or_step_or_contacts_or_distance=True,no_teacher_labels_or_context_lookup=True,
        zero_object_identity_preserved=True,initial_and_nominal_extra_exact_zero=True,home_original=True,combined_14_joint_cap_rad=.18,physics_reward_acceptance_original=True,
        initial_std=.001,std_bounds=[.0001,.005],elite=2,cem_update=[.6,.4],controls=2279,control_hz=50,physics_hz=500,entry_deadline_s=12,strict_tail_s=30,training_cases=CASES,
        automatic_expanded_or_qualification=False,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,mujoco_version=mujoco.__version__,
        hashes={str(f):prior.digest(f) for f in [*src.iterdir(),*frozen.iterdir(),local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE]}))
    zero=np.zeros(2);selected=zero.copy();mean=zero.copy();std=np.full(2,.001);rng=np.random.default_rng(277);history=[];cache={};best=None
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
        parity=group(pool,zero,[None,769000,773004],output/'zero_parity',True);nonzero=group(pool,PROBE,[None,769002,773004],output/'nonzero_smoke')
        assert all(row['maximum_foot_task_extra_rad']>0 for row in nonzero['rows'] if row['case_seed'] is not None)
        if not args.smoke:compare_smoke(output)
        local.write_json(output/'startup_closed.json',dict(parity=parity,nonzero=nonzero,independent_smoke_bitwise_equal=not args.smoke,terminal_result_saved=False,training_generations_saved=0))
        print('R177_ZERO_NOMINAL_STARTUP_PARITY_PASS',flush=True)
        if args.smoke:
            local.write_json(output/'results.json',dict(smoke=True,parity=parity,nonzero=nonzero,terminal_result_saved=True,full_task_completed=False,hardware_readiness=False));return
        for generation in range(1,3):
            assert shutil.disk_usage(ROOT).free>10*1024**3,'Preserve evidence, do not clean other data'
            proposals=[zero.copy(),selected.copy(),*np.clip(rng.normal(mean,std,(6,2)),0,BOUND)];reports=[];locations=[]
            for i,c in enumerate(proposals):
                key=c.tobytes().hex();path=output/'training'/f'generation_{generation:04d}'/f'candidate_{i:02d}'
                if key not in cache:cache[key]=(group(pool,c,CASES,path,not np.any(c)),str(path))
                report,location=cache[key];reports.append(report);locations.append(location);print('R177_FULL_CANDIDATE',generation,i,report['successes'],report['physical_failures'],flush=True)
            order=sorted(range(8),key=lambda i:local.rank(reports[i]),reverse=True);winner=order[0]
            if best is None or local.rank(reports[winner])>local.rank(best):selected=proposals[winner].copy();best=reports[winner]
            elite=np.stack([proposals[i] for i in order[:2]]);mean=.6*mean+.4*elite.mean(0);std=np.clip(.6*std+.4*elite.std(0),.0001,.005)
            history.append(dict(generation=generation,proposals=[c.tolist() for c in proposals],reports=reports,closed_trial_directories=locations,selected_parameters=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
            checkpoint=output/'checkpoints'/f'generation_{generation:04d}';checkpoint.mkdir(parents=True,exist_ok=False);np.savez_compressed(checkpoint/'state.npz',parameters=selected,mean=mean,std=std)
            local.write_json(checkpoint/'rng.json',rng.bit_generator.state);local.write_json(checkpoint/'history.json',history);local.write_json(output/'progress.json',dict(closed_generation=generation,generations=2,best=best,parameters=selected.tolist(),terminal_result_saved=False))
        local.write_json(output/'training_closed.json',dict(history=history,rng=rng.bit_generator.state,best=best,parameters=selected.tolist(),all_training_full_path=True))
        candidate=group(pool,selected,CASES,output/'development_candidate');baseline=group(pool,zero,CASES,output/'development_baseline',True)
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
        local.write_json(output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,parameters=selected.tolist(),original_development_gate=gate,terminal_result_saved=True,expanded_development_run=False,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
        print('R177_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)


if __name__=='__main__':main()
