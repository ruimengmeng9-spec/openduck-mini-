"""Independent read-only R185b scalar and target planning reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import mujoco
import numpy as np
from diagnostics import probe_getup_phase_coordinate_r185b as run

ROOT=run.ROOT
OUTPUT=ROOT/'outputs/getup_phase_coordinate_terminal_audit_r186_20261010'
SMOKE=ROOT/'outputs/getup_phase_coordinate_terminal_audit_r186_smoke_20261010'
STORED_MAIN_SHA='62109cd963e4ad406dfc69adaf9cb0565ab39c17358804b11ff4f79a1a552842'
FIELDS=('observations','preparation_sensors','applied','phase_reference_shift_rad',
        'phase_offset_controls','phase_joint_error_change_rad','phase_nominal_tangent_rad_per_control',
        'local_hip_extra_rad','planned_before_integration_rad','same_state_unperturbed_planned_rad',
        'same_state_pre_slew_direct_new_rad','original_double_pre_slew_target_rad',
        'adjusted_double_pre_slew_target_rad')
DT=.02
SLEW=5.24
RIDGE_RAD=.02
MAX_SHIFT=1.
RECOVERY=529


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


def limits(target,previous,lower,upper):
    return np.clip(np.clip(target,lower,upper),previous-SLEW*DT,previous+SLEW*DT)


def phase(current,nominal,initial,initial_nominal,tangent,reference,control,enabled):
    """No recorded phase, added request, labels or future actual sensor input."""
    xs=[np.asarray(x,dtype=np.float32) for x in (current,nominal,initial,initial_nominal)]
    v=np.asarray(tangent,dtype=float);r=np.asarray(reference,dtype=float)
    if any(x.shape!=(14,) or not np.isfinite(x).all() for x in xs):raise ValueError('Finite joint14 required')
    if v.shape!=(14,) or not np.isfinite(v).all() or r.ndim!=2 or r.shape[1]!=14 or not np.isfinite(r).all():raise ValueError('Finite reference required')
    if not isinstance(control,(int,np.integer)) or control<0 or control>=len(r):raise ValueError('Integer phase required')
    q,n,i,z=[x.astype(float) for x in xs]
    delta=(q-n)-(i-z)
    offset=float(np.clip(np.dot(v,delta)/(np.dot(v,v)+RIDGE_RAD**2),-MAX_SHIFT,MAX_SHIFT)) if enabled and control<RECOVERY else 0.
    if not enabled or control>=RECOVERY:delta=np.zeros(14)
    if offset==0.:return np.zeros(14),offset,delta
    coordinate=float(np.clip(control+offset,0.,min(RECOVERY-1.,len(r)-1)))
    lo=int(np.floor(coordinate));hi=min(lo+1,len(r)-1);fraction=coordinate-lo
    shifted=(1.-fraction)*r[lo]+fraction*r[hi]
    return shifted-r[control],offset,delta


def merge(fixed,reference,extra,lower,upper):
    if not np.any(extra):return fixed
    return np.clip(reference+np.clip(fixed-reference+extra,-.18,.18),lower,upper)


def jobs():
    paths=[]
    for root in (run.SMOKE,run.OUTPUT):
        names=('zero_parity','phase_smoke') if root==run.SMOKE else ('zero_parity','phase_smoke','development_baseline','development_candidate')
        for name in names:
            for row in read(root/name/'results.json')['rows']:paths.append(root/name/('case_'+str(row['case_seed'])))
    assert len(paths)==62 and len(set(paths))==62
    return paths


def smoke_jobs():
    return [run.OUTPUT/'zero_parity/case_None',run.OUTPUT/'phase_smoke/case_None',
        run.OUTPUT/'development_baseline/case_769000',
        *[run.OUTPUT/'development_candidate'/('case_'+str(case)) for case in (769002,773005,769006)]]


def identity(path):return path.parent.parent.name+'__'+path.parent.name+'__'+path.name


def scalar(path,home,lower,upper,nominal,tangents,reference,selector,save=None):
    row=read(path/'result.json')
    with np.load(path/'trajectory.npz',allow_pickle=False) as z:data={key:z[key].copy() for key in FIELDS}
    obs=data['observations'];n=len(obs)
    assert n==row['controls'] and 0<n<=2279
    gains,choice,logits=run.selector.old.predict(selector,data['preparation_sensors'][-1])
    np.testing.assert_array_equal(gains,row['base_gains']);assert choice==row['base_choice']
    np.testing.assert_array_equal(logits,row['initial_sensor_logits'])
    previous=home.copy();initial=obs[0].copy();max_total=0.
    keys=('phase_reference_shift_rad','phase_offset_controls','phase_joint_error_change_rad',
        'phase_nominal_tangent_rad_per_control','local_hip_extra_rad','planned_before_integration_rad',
        'same_state_unperturbed_planned_rad','same_state_pre_slew_direct_new_rad','adjusted_double_pre_slew_target_rad')
    records={key:[] for key in keys};post=[]
    for k,x in enumerate(obs):
        np.testing.assert_array_equal(x[34:48],(previous-home).astype(np.float32))
        hip=run.local.local_feedback(x,nominal[k],[9,10,11],gains) if k<RECOVERY else np.zeros(3)
        # This recorded exact post-IMU/post-original-hip double request is a
        # declared input boundary, not reconstruction of the unsaved double IMU.
        fixed=data['original_double_pre_slew_target_rad'][k].copy()
        tangent=tangents[k] if k<RECOVERY else np.zeros(14)
        extra,offset,delta=phase(x[6:20],nominal[k,6:20],initial[6:20],nominal[0,6:20],tangent,reference,k,row['enabled'])
        adjusted=merge(fixed,reference[k],extra,lower,upper)
        base=limits(fixed,previous,lower,upper);wanted=limits(adjusted,previous,lower,upper)
        values=(extra,offset,delta,tangent,hip,wanted,base,adjusted-fixed,adjusted)
        for key,value in zip(keys,values):records[key].append(np.asarray(value).copy())
        post.append(wanted-base)
        total=float(np.abs(adjusted-reference[k]).max());max_total=max(max_total,total)
        assert total<=.18+1e-12
        np.testing.assert_array_equal(wanted,data['applied'][k]);previous=wanted
    for key,value in records.items():np.testing.assert_array_equal(value,data[key],err_msg=str(path)+' '+key)
    assert max_total==row['maximum_combined_14_joint_correction_rad']
    assert float(np.abs(records['phase_reference_shift_rad']).max())==row['maximum_phase_reference_shift_rad']
    assert float(np.abs(records['phase_offset_controls']).max())==row['maximum_phase_offset_controls']
    for key in ('phase_reference_shift_rad','phase_offset_controls','phase_joint_error_change_rad'):
        np.testing.assert_array_equal(records[key][0],np.zeros_like(records[key][0]))
        if n>RECOVERY:np.testing.assert_array_equal(np.asarray(records[key])[RECOVERY:],np.zeros_like(np.asarray(records[key])[RECOVERY:]))
        if row['case_seed'] is None or not row['enabled']:np.testing.assert_array_equal(records[key],np.zeros_like(records[key]))
    first=np.flatnonzero(np.any(np.asarray(post)!=0,axis=1))
    summary=dict(path=str(path),identity=identity(path),case_seed=row['case_seed'],enabled=row['enabled'],
        controls=n,success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],peaks=row['peaks'],
        entry_time_s=row['entry_time_s'],strict_tail_s=row['strict_tail_s'],
        phase_and_original_limits_independently_bitwise=True,original_right_hip_feedback_bitwise=True,
        original_double_IMU_request_reconstructed=False,raw_request_max_rad=float(np.abs(records['phase_reference_shift_rad']).max()),
        phase_coordinate_max=float(np.abs(records['phase_offset_controls']).max()),
        pre_slew_direct_max_rad=float(np.abs(records['same_state_pre_slew_direct_new_rad']).max()),
        post_slew_direct_max_rad=float(np.abs(post).max()),first_nonzero_direct=int(first[0]) if len(first) else None,maximum_combined=max_total)
    if save is not None:
        save.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(save/'signals.npz',**{key:np.asarray(value) for key,value in records.items()},post_slew_direct_new_rad=np.asarray(post))
        write(save/'summary.json',summary)
    return summary


def terminal_pairing():
    rows=[]
    for case in run.CASES:
        cpath=run.OUTPUT/'development_candidate'/('case_'+str(case))
        bpath=run.OUTPUT/'development_baseline'/('case_'+str(case))
        opath=run.selector.OUTPUT/'candidate'/('case_'+str(case))
        c,b,o=[read(p/'result.json') for p in (cpath,bpath,opath)]
        assert c['initial_hash']==b['initial_hash']==o['initial_hash']
        # Only separate equality decodes qpos/qvel; never scalar or planning.
        with np.load(cpath/'trajectory.npz',allow_pickle=False) as cz,np.load(bpath/'trajectory.npz',allow_pickle=False) as bz,np.load(opath/'trajectory.npz',allow_pickle=False) as oz:
            equality={key:bool(np.array_equal(bz[key],oz[key])) for key in oz.files}
            assert all(equality.values()) and b['peaks']==o['peaks']
            difference={key:bool(np.array_equal(cz[key],bz[key])) for key in cz.files}
        rows.append(dict(case_seed=case,initial_hash=c['initial_hash'],baseline_original_fields_bitwise=equality,
            candidate_baseline_field_equality=difference,candidate_success=c['success'],baseline_success=b['success'],
            candidate_valid=c['valid'],baseline_valid=b['valid'],candidate_peaks=c['peaks'],baseline_peaks=b['peaks']))
    return rows


def source_inventory(all_jobs):
    paths=[Path(__file__).resolve(),Path(__file__).with_name('test_getup_phase_coordinate_audit_r186.py'),Path(__file__).with_name('launch_getup_phase_coordinate_audit_r186.py')]
    paths.extend(Path(m.__file__).resolve() for name,m in list(sys.modules.items()) if name.startswith('diagnostics.') and getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py')
    for root in (run.OUTPUT,run.SMOKE):
        paths.extend(root/'frozen'/name for name in ('initial_selector.npz','snapshot.npz','nominal_sensor_trajectory.npz'))
        paths.extend((root/'contract.json',root/'results.json'))
    paths.extend((run.OUTPUT/'training_closed.json',run.local.prior.program.REFERENCE,run.local.prior.program.SCENE,run.local.prior.program.STAND))
    for p in all_jobs:paths.extend((p/'result.json',p/'trajectory.npz'))
    failed=ROOT/'outputs/getup_phase_coordinate_r185_smoke_20261010'
    paths.extend(p for p in failed.rglob('*') if p.is_file())
    paths.append(failed.with_suffix('.log'))
    return {str(p):digest(p) for p in sorted(set(paths))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>10*1024**3
    assert digest(run.__file__)==STORED_MAIN_SHA
    all_jobs=jobs();selected=smoke_jobs() if args.smoke else all_jobs;before=source_inventory(all_jobs)
    output.mkdir();src=output/'executed_sources';src.mkdir()
    for path in before:
        p=Path(path)
        if p.suffix=='.py':shutil.copy2(p,src/p.name)
    assert (src/Path(__file__).name).exists()
    for root in (run.OUTPUT,run.SMOKE):
        for path,expected in read(root/'contract.json')['hashes'].items():assert digest(path)==expected
    frozen=run.OUTPUT/'frozen'
    with np.load(frozen/'initial_selector.npz',allow_pickle=False) as z:selector={key:z[key].copy() for key in z.files}
    with np.load(frozen/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    with np.load(run.local.prior.program.REFERENCE,allow_pickle=False) as z:reference=z['targets'].copy()
    tangents=np.gradient(nominal[:RECOVERY,6:20].astype(float),axis=0)
    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    arrays={key:np.asarray(getattr(model,key)).copy() for key in ('jnt_range','key_qpos','body_pos','body_quat','actuator_trnid','mesh_vert')}
    addresses=model.jnt_qposadr[model.actuator_trnid[:,0]]
    home=model.keyframe('home').qpos[addresses].copy();lower,upper=model.jnt_range[model.actuator_trnid[:,0]].T.copy()
    write(output/'contract.json',dict(simulation_only=True,read_only=True,new_dynamic_trajectories=0,smoke=args.smoke,
        audited_existing_trajectory_budget=6 if args.smoke else 62,storage_budget_gb=.25,reserve_gb=10,
        model_mapping_only=True,no_environment_MjData_forward_integration_contact_force=True,
        scalar_inputs=('observations','preparation_sensors','original_double_pre_slew_target_rad'),comparison_fields=FIELDS,
        saved_phase_extra_adjusted_or_planned_not_used_as_decisions=True,original_IMU_double_request_reconstructed=False,
        R185_missing_prepare_partial_recovered=False,source_hashes=before,full_task_completed=False,hardware_readiness=False,qualification=False))
    rows=[]
    for path in selected:
        save=output/'signals'/identity(path) if args.smoke or path in smoke_jobs() or path.parent.name=='development_candidate' else None
        rows.append(scalar(path,home,lower,upper,nominal,tangents,reference,selector,save));print('R186_SCALAR',identity(path),'bitwise',flush=True)
    pairs=terminal_pairing()
    if not args.smoke:
        old=read(SMOKE/'results.json');subset=[next(r for r in rows if r['identity']==prior['identity']) for prior in old['rows']]
        assert subset==old['rows'] and pairs==old['terminal_pairs']
        for row in subset:
            ident=row['identity']
            with np.load(output/'signals'/ident/'signals.npz',allow_pickle=False) as a,np.load(SMOKE/'signals'/ident/'signals.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    after=source_inventory(all_jobs);assert before==after
    for key,value in arrays.items():np.testing.assert_array_equal(value,getattr(model,key))
    write(output/'source_hashes.json',dict(before=before,after=after,unchanged=True))
    write(output/'results.json',dict(rows=rows,terminal_pairs=pairs,terminal_result_saved=True,
        existing_trajectories_audited=len(rows),controls_audited=sum(r['controls'] for r in rows),new_dynamic_trajectories=0,
        all_scalar_and_limits_bitwise=True,independent_smoke_matches=not args.smoke,source_evidence_unchanged=True,
        original_IMU_double_request_reconstructed=False,qualification=False,full_task_completed=False,hardware_readiness=False))
    print('R186_READ_ONLY_TERMINAL',len(rows),flush=True)


if __name__=='__main__':main()
