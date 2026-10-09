"""Read-only reconstruction of R183b contact projection and original limits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import mujoco
import numpy as np
from diagnostics import probe_getup_contact_projection_r183b as run

ROOT=run.ROOT
OUTPUT=ROOT/'outputs/getup_contact_projection_terminal_audit_r184_20261010'
SMOKE=ROOT/'outputs/getup_contact_projection_terminal_audit_r184_smoke_20261010'
FIELDS=('observations','preparation_sensors','applied','projection_extra_rad',
        'relative_foot_residual_m_equivalent','projection_jacobian','projection_trace',
        'contact_projection_active','causal_planned_anchor','local_hip_extra_rad',
        'planned_before_integration_rad','same_state_unperturbed_planned_rad',
        'original_double_pre_slew_target_rad','adjusted_double_pre_slew_target_rad')
STORED_MAIN_SHA='b63812459d125ba830d0c6a2a85ab83267cc0731eb0877d6ed7de5cd58a81348'
DT=.02
SLEW=5.24
ITERATIONS=3
FD=1e-6
RCOND=1e-10
ROTATION_LENGTH=.05
STEPS=(1.,.5,.25,.125,.0625)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def limits(target,previous,lower,upper):
    return np.clip(np.clip(target,lower,upper),previous-SLEW*DT,previous+SLEW*DT)


class IndependentGeometry:
    """Only canonical planned joint kinematics; no actual root or integration."""
    def __init__(self,model):
        self.model=model
        self.data=mujoco.MjData(model)
        self.qadr=model.jnt_qposadr[model.actuator_trnid[:,0]].copy()
        self.canonical=model.keyframe('home').qpos.copy()
        self.canonical[:7]=[0,0,0,1,0,0,0]
        self.home=self.canonical[self.qadr].copy()
        self.legs=np.array([model.actuator(side+'_'+joint).id for side in ('left','right')
            for joint in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
        self.feet=[model.body(name).id for name in ('foot_assembly','foot_assembly_2')]

    def pose(self,planned):
        self.data.qpos[:]=self.canonical
        self.data.qpos[self.qadr[self.legs]]=planned[self.legs]
        mujoco.mj_kinematics(self.model,self.data)
        left,right=self.feet
        rotation=self.data.xmat[left].reshape(3,3)
        return (rotation.T@(self.data.xpos[right]-self.data.xpos[left])).copy(), (rotation.T@self.data.xmat[right].reshape(3,3)).copy()

    def anchor(self,planned,nominal):
        position,rotation=self.pose(planned)
        nominal_position,nominal_rotation=self.pose(nominal)
        offset=np.eye(3) if np.array_equal(planned,nominal) else rotation@nominal_rotation.T
        return np.concatenate([position-nominal_position,offset.reshape(9)])

    def residual(self,planned,nominal,anchor):
        position,rotation=self.pose(planned)
        nominal_position,nominal_rotation=self.pose(nominal)
        offset=anchor[3:].reshape(3,3)
        desired=nominal_rotation if np.array_equal(offset,np.eye(3)) else offset@nominal_rotation
        angle=np.zeros(3)
        if not np.array_equal(rotation,desired):
            quaternion=np.empty(4)
            mujoco.mju_mat2Quat(quaternion,(rotation@desired.T).reshape(9))
            if quaternion[0]<0:quaternion=-quaternion
            mujoco.mju_quat2Vel(angle,quaternion,1.)
        return np.concatenate([position-(nominal_position+anchor[:3]),ROTATION_LENGTH*angle])

    def jacobian(self,planned,nominal,anchor):
        jac=np.empty((6,10))
        for j,index in enumerate(self.legs):
            plus=planned.copy();minus=planned.copy()
            plus[index]+=FD;minus[index]-=FD
            jac[:,j]=(self.residual(plus,nominal,anchor)-self.residual(minus,nominal,anchor))/(2*FD)
        return jac


def reconstruct(planned,nominal,contacts,anchor,lower,upper,geometry,enabled):
    """Never accepts saved active, trace, residual, Jacobian or added action."""
    error=geometry.residual(planned,nominal,anchor)
    trace=np.zeros((ITERATIONS+1,3));trace[:,0]=np.linalg.norm(error)
    jac=np.zeros((6,10))
    active=bool(enabled and np.all(contacts>0))
    if not active or not np.any(error):return planned,error,jac,trace,active
    result=planned.copy()
    for iteration in range(ITERATIONS):
        jac=geometry.jacobian(result,nominal,anchor)
        correction=-np.linalg.pinv(jac,rcond=RCOND)@error
        accepted=0.
        for scale in STEPS:
            trial=result.copy()
            trial[geometry.legs]=np.clip(result[geometry.legs]+scale*correction,lower[geometry.legs],upper[geometry.legs])
            trial_error=geometry.residual(trial,nominal,anchor)
            if np.linalg.norm(trial_error)<np.linalg.norm(error):
                result=trial;error=trial_error;accepted=scale;break
        trace[iteration+1]=[np.linalg.norm(error),accepted,np.linalg.matrix_rank(jac,tol=RCOND)]
    if np.array_equal(result,planned):return planned,error,jac,trace,active
    return result,error,jac,trace,active


def jobs():
    all_jobs=[]
    for root in (run.SMOKE,run.OUTPUT):
        names=('zero_parity','projection_smoke') if root==run.SMOKE else ('zero_parity','projection_smoke','development_baseline','development_candidate')
        for name in names:
            report=read(root/name/'results.json')
            for row in report['rows']:all_jobs.append(root/name/('case_'+str(row['case_seed'])))
    assert len(all_jobs)==62 and len(set(all_jobs))==62
    return all_jobs


def smoke_jobs():
    return [run.OUTPUT/'zero_parity/case_None',run.OUTPUT/'projection_smoke/case_None',
        run.OUTPUT/'development_baseline/case_769000',
        *[run.OUTPUT/'development_candidate'/('case_'+str(case)) for case in (773005,769006,773009)]]


def identity(path):return path.parent.parent.name+'__'+path.parent.name+'__'+path.name


def scalar(path,geometry,nominal_sensor,nominal_planned,reference,selector,save=None):
    row=read(path/'result.json')
    with np.load(path/'trajectory.npz',allow_pickle=False) as z:
        data={key:z[key].copy() for key in FIELDS}
    obs=data['observations'];n=len(obs)
    assert n==2279 and row['controls']==n and row['valid']
    gains,choice,logits=run.selector.old.predict(selector,data['preparation_sensors'][-1])
    np.testing.assert_array_equal(gains,row['base_gains'])
    assert choice==row['base_choice']
    np.testing.assert_array_equal(logits,row['initial_sensor_logits'])
    lower,upper=geometry.model.jnt_range[geometry.model.actuator_trnid[:,0]].T.copy()
    previous=geometry.home.copy();anchor=None
    keys=('projection_extra_rad','relative_foot_residual_m_equivalent','projection_jacobian','projection_trace',
        'contact_projection_active','causal_planned_anchor','local_hip_extra_rad','planned_before_integration_rad',
        'same_state_unperturbed_planned_rad','adjusted_double_pre_slew_target_rad')
    records={key:[] for key in keys};feasible_history=[];raw_double=[];max_total=0.
    for k,x in enumerate(obs):
        np.testing.assert_array_equal(x[34:48],(previous-geometry.home).astype(np.float32))
        frozen=run.local.local_feedback(x,nominal_sensor[k],[9,10,11],gains) if k<529 else np.zeros(3)
        # The original IMU-adjusted request was not separately saved. This
        # exact recorded pre-slew target is a declared audit boundary/input,
        # not a claim that the original double IMU is reconstructed.
        fixed=data['original_double_pre_slew_target_rad'][k].copy()
        base=limits(fixed,previous,lower,upper)
        adjusted=fixed
        error=np.zeros(6);jac=np.zeros((6,10));trace=np.zeros((4,3));active=False;feasible=False
        if k==0:anchor=geometry.anchor(base,nominal_planned[k])
        elif k<529:
            raw_lower=np.maximum(reference[k]-.18,lower)
            raw_upper=np.minimum(reference[k]+.18,upper)
            lo=np.maximum(raw_lower,limits(raw_lower,previous,lower,upper))
            hi=np.minimum(raw_upper,limits(raw_upper,previous,lower,upper))
            leg=geometry.legs
            feasible=bool(np.all(lo[leg]<=hi[leg]) and np.all(base[leg]>=lo[leg]) and np.all(base[leg]<=hi[leg]))
            if feasible:
                projected,error,jac,trace,active=reconstruct(base,nominal_planned[k],x[48:50],anchor,lo,hi,geometry,row['enabled'])
                if projected is not base:
                    adjusted=projected.copy();adjusted[5:9]=fixed[5:9]
        else:anchor=np.zeros(12)
        wanted=limits(adjusted,previous,lower,upper)
        extra=wanted-base
        max_total=max(max_total,float(np.abs(adjusted-reference[k]).max()))
        assert max_total<=.18+1e-12
        values=(extra,error,jac,trace,active,anchor,frozen,wanted,base,adjusted)
        for key,value in zip(keys,values):records[key].append(np.asarray(value).copy())
        feasible_history.append(feasible)
        raw_double.append(bool(k<529 and np.all(x[48:50]>0)))
        np.testing.assert_array_equal(wanted,data['applied'][k])
        previous=wanted
    for key,values in records.items():np.testing.assert_array_equal(np.asarray(values),data[key],err_msg=str(path)+' '+key)
    assert max_total==row['maximum_combined_14_joint_correction_rad']
    assert int(np.count_nonzero(records['contact_projection_active']))==row['active_contact_controls']
    np.testing.assert_array_equal(records['projection_extra_rad'][0],np.zeros(14))
    np.testing.assert_array_equal(np.asarray(records['projection_extra_rad'])[529:],np.zeros((1750,14)))
    np.testing.assert_array_equal(np.asarray(records['projection_extra_rad'])[:,5:9],np.zeros((n,4)))
    if row['case_seed'] is None or not row['enabled']:
        np.testing.assert_array_equal(records['projection_extra_rad'],np.zeros((n,14)))
    nonzero=np.flatnonzero(np.any(np.asarray(records['projection_extra_rad'])!=0,axis=1))
    summary=dict(path=str(path),identity=identity(path),case_seed=row['case_seed'],enabled=row['enabled'],
        controls=n,success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],peaks=row['peaks'],
        entry_time_s=row['entry_time_s'],strict_tail_s=row['strict_tail_s'],
        independent_anchor_bounds_active_solver_and_limits_bitwise=True,
        original_right_hip_feedback_bitwise=True,original_double_IMU_request_reconstructed=False,
        raw_double_contact_controls=int(np.count_nonzero(raw_double)),
        feasible_controls=int(np.count_nonzero(feasible_history)),
        active_controls=row['active_contact_controls'],first_nonzero=int(nonzero[0]) if len(nonzero) else None,
        maximum_projection_extra_rad=float(np.abs(records['projection_extra_rad']).max()),maximum_combined=max_total)
    if save is not None:
        save.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(save/'signals.npz',**{key:np.asarray(value) for key,value in records.items()},
            feasible=np.asarray(feasible_history),raw_double_contact=np.asarray(raw_double))
        write(save/'summary.json',summary)
    return summary


def terminal_pairing():
    rows=[]
    for case in run.CASES:
        candidate=run.OUTPUT/'development_candidate'/('case_'+str(case))
        baseline=run.OUTPUT/'development_baseline'/('case_'+str(case))
        original=run.selector.OUTPUT/'candidate'/('case_'+str(case))
        c,b,o=[read(path/'result.json') for path in (candidate,baseline,original)]
        assert c['initial_hash']==b['initial_hash']==o['initial_hash']
        # Only this separate original-field comparison decodes recorded
        # qpos/qvel; never the scalar, decision, pose or limit reconstruction.
        with np.load(candidate/'trajectory.npz',allow_pickle=False) as cz, np.load(baseline/'trajectory.npz',allow_pickle=False) as bz, np.load(original/'trajectory.npz',allow_pickle=False) as oz:
            base={key:bool(np.array_equal(bz[key],oz[key])) for key in oz.files}
            assert all(base.values()) and b['peaks']==o['peaks']
            difference={key:bool(np.array_equal(cz[key],bz[key])) for key in cz.files}
        rows.append(dict(case_seed=case,initial_hash=c['initial_hash'],baseline_original_fields_bitwise=base,
            candidate_baseline_field_equality=difference,candidate_success=c['success'],baseline_success=b['success'],
            candidate_valid=c['valid'],baseline_valid=b['valid'],candidate_peaks=c['peaks'],baseline_peaks=b['peaks']))
    return rows


def source_inventory(all_jobs):
    paths=[Path(__file__).resolve(),Path(__file__).with_name('test_getup_contact_projection_audit_r184.py'),
        Path(__file__).with_name('launch_getup_contact_projection_audit_r184.py')]
    paths.extend(Path(module.__file__).resolve() for name,module in list(sys.modules.items())
        if name.startswith('diagnostics.') and getattr(module,'__file__',None) and Path(module.__file__).suffix=='.py')
    for root in (run.OUTPUT,run.SMOKE):
        paths.extend(root/'frozen'/name for name in ('initial_selector.npz','snapshot.npz','nominal_sensor_trajectory.npz'))
        paths.extend((root/'contract.json',root/'results.json'))
    paths.extend((run.OUTPUT/'training_closed.json',run.local.prior.program.REFERENCE,run.local.prior.program.SCENE,run.local.prior.program.STAND))
    for path in all_jobs:paths.extend((path/'result.json',path/'trajectory.npz'))
    return {str(path):digest(path) for path in sorted(set(paths))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>10*1024**3
    assert digest(Path(run.__file__))==STORED_MAIN_SHA
    all_jobs=jobs();selected=smoke_jobs() if args.smoke else all_jobs
    before=source_inventory(all_jobs)
    output.mkdir();src=output/'executed_sources';src.mkdir()
    for path in before:
        f=Path(path)
        if f.suffix=='.py':shutil.copy2(f,src/f.name)
    assert (src/Path(__file__).name).exists()
    for root in (run.OUTPUT,run.SMOKE):
        contract=read(root/'contract.json')
        for path,expected in contract['hashes'].items():assert digest(path)==expected
    frozen=run.OUTPUT/'frozen'
    with np.load(frozen/'initial_selector.npz',allow_pickle=False) as z:selector={key:z[key].copy() for key in z.files}
    with np.load(frozen/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:
        nominal_sensor=z['observations'].copy();nominal_planned=z['applied'].copy()
    with np.load(run.local.prior.program.REFERENCE,allow_pickle=False) as z:reference=z['targets'].copy()
    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    geometry=IndependentGeometry(model)
    model_before={key:np.asarray(getattr(model,key)).copy() for key in ('jnt_range','key_qpos','body_pos','body_quat','actuator_trnid','mesh_vert')}
    write(output/'contract.json',dict(simulation_only=True,read_only=True,new_dynamic_trajectories=0,smoke=args.smoke,
        audited_existing_trajectory_budget=6 if args.smoke else 62,storage_budget_gb=.25,reserve_gb=10,
        private_canonical_MjData=True,only_kinematics=True,no_environment_or_forward_integration_contact_force=True,
        scalar_decision_inputs=FIELDS[:3]+('original_double_pre_slew_target_rad',),
        saved_active_solver_outputs_not_used_as_decisions=True,original_IMU_double_request_reconstructed=False,
        R183_original_partial_frames_recovered=False,R183b_original_main_startup_capture_repaired=False,
        source_hashes=before,full_task_completed=False,hardware_readiness=False,qualification=False))
    records=[]
    for path in selected:
        save=output/'signals'/identity(path) if args.smoke or path in smoke_jobs() or path.parent.name=='development_candidate' else None
        records.append(scalar(path,geometry,nominal_sensor,nominal_planned,reference,selector,save))
        print('R184_SCALAR',identity(path),'bitwise',flush=True)
    pairs=terminal_pairing()
    if not args.smoke:
        old=read(SMOKE/'results.json')
        subset=[next(row for row in records if row['identity']==prior['identity']) for prior in old['rows']]
        assert subset==old['rows'] and pairs==old['terminal_pairs']
        for prior in subset:
            ident=prior['identity']
            with np.load(output/'signals'/ident/'signals.npz',allow_pickle=False) as a, np.load(SMOKE/'signals'/ident/'signals.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    after=source_inventory(all_jobs);assert before==after
    for key,value in model_before.items():np.testing.assert_array_equal(value,getattr(model,key))
    write(output/'source_hashes.json',dict(before=before,after=after,unchanged=True))
    write(output/'results.json',dict(rows=records,terminal_pairs=pairs,terminal_result_saved=True,
        existing_trajectories_audited=len(records),new_dynamic_trajectories=0,all_scalar_bitwise=True,
        independent_smoke_matches=not args.smoke,source_evidence_unchanged=True,
        original_IMU_double_request_reconstructed=False,qualification=False,full_task_completed=False,hardware_readiness=False))
    print('R184_READ_ONLY_TERMINAL',len(records),flush=True)


if __name__=='__main__':main()

