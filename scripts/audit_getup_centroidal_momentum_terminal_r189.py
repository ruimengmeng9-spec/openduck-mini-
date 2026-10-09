"""Independent offline native-sensor momentum and original target reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import mujoco
import numpy as np
from diagnostics import probe_getup_centroidal_momentum_r188b as run

ROOT=run.ROOT
OUTPUT=ROOT/'outputs/getup_centroidal_momentum_terminal_audit_r189_20261010'
SMOKE=ROOT/'outputs/getup_centroidal_momentum_terminal_audit_r189_smoke_20261010'
STORED_MAIN_SHA='4a24d4797579fd1549c54424ab80a680677ec90725e1f006bee40f63e734ba94'
COMPILED_SHA='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
SIGNALS=('momentum_request_rad','momentum_error_kg_m2_per_s','momentum_allocation_kg_m2',
 'momentum_unconstrained_residual_kg_m2_per_s','actual_momentum_kg_m2_per_s',
 'nominal_momentum_kg_m2_per_s','initial_momentum_kg_m2_per_s','nominal_initial_momentum_kg_m2_per_s',
 'local_hip_extra_rad','planned_before_integration_rad','same_state_unperturbed_planned_rad',
 'same_state_pre_slew_direct_new_rad','adjusted_double_pre_slew_target_rad')
FIELDS=('observations','preparation_sensors','applied','original_double_pre_slew_target_rad',*SIGNALS)
DT=.02
SLEW=5.24
RECOVERY=529

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')

def model_digest(model):
    buffer=np.empty(mujoco.mj_sizeModel(model),dtype=np.uint8)
    mujoco.mj_saveModel(model,buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()

def load_model():
    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    # Exact original NativeSim configuration, not a changed simulation option.
    model.opt.timestep=.002
    assert model_digest(model)==COMPILED_SHA
    return model

class IndependentGeometry:
    """Canonical planning state only; no episode/root truth or dynamics."""
    def __init__(self,model):
        self.model=model;self.data=mujoco.MjData(model)
        joints=model.actuator_trnid[:,0]
        self.qadr=model.jnt_qposadr[joints].copy();self.vadr=model.jnt_dofadr[joints].copy()
        assert len(joints)==14 and len(np.unique(self.vadr))==14
        self.home=model.keyframe('home').qpos[self.qadr].copy()
        self.canonical=model.keyframe('home').qpos.copy()
        free=np.flatnonzero(model.jnt_type==mujoco.mjtJoint.mjJNT_FREE);assert len(free)==1
        j=int(free[0]);qa=int(model.jnt_qposadr[j]);va=int(model.jnt_dofadr[j])
        self.canonical[qa:qa+7]=[0,0,0,1,0,0,0]
        self.rootlinear=np.arange(va,va+3);self.rootangular=np.arange(va+3,va+6)
        self.rootbody=int(model.jnt_bodyid[j]);self.trunk=model.body('trunk_assembly').id
        gyro=model.sensor('gyro').id;assert model.sensor_type[gyro]==mujoco.mjtSensor.mjSENS_GYRO
        self.site=int(model.sensor_objid[gyro]);self.gyrobody=int(model.site_bodyid[self.site])
        assert model.body_weldid[self.gyrobody]==model.body_weldid[self.trunk]
        self.legs=np.array([model.actuator(s+'_'+p).id for s in ('left','right') for p in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
        np.testing.assert_array_equal(self.legs,[0,1,2,3,4,9,10,11,12,13])
        self.head=np.array([j for j in range(14) if j not in self.legs])
        self.lower,self.upper=model.jnt_range[joints].T.copy()

    def measure(self,sensor34):
        x=np.asarray(sensor34,dtype=np.float32)
        if x.shape!=(34,) or not np.isfinite(x).all():raise ValueError('Finite native34 required')
        d,m=self.data,self.model
        d.qpos[:]=self.canonical;d.qpos[self.qadr]=self.home+x[6:20].astype(float);d.qvel[:]=0.
        mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        matrix=np.zeros((3,m.nv));mujoco.mj_angmomMat(m,d,matrix,self.rootbody)
        jr=np.zeros_like(matrix);mujoco.mj_jacSite(m,d,None,jr,self.site)
        velocity=np.zeros(m.nv);velocity[self.vadr]=x[20:34].astype(float)/.05
        rotation=d.site_xmat[self.site].reshape(3,3)
        velocity[self.rootangular]=np.linalg.solve(jr[:,self.rootangular],rotation@x[:3].astype(float)-jr[:,self.vadr]@velocity[self.vadr])
        bodymatrix=d.xmat[self.trunk].reshape(3,3).T@matrix
        return bodymatrix@velocity,bodymatrix[:,self.vadr[self.legs]].copy(),bodymatrix,velocity

def momentum(current,nominal,initial,initial_nominal,geometry,control,enabled):
    if not isinstance(control,(int,np.integer)) or control<0:raise ValueError('Nonnegative integer phase required')
    xs=[np.asarray(x,dtype=np.float32) for x in (current,nominal,initial,initial_nominal)]
    if any(x.shape!=(34,) or not np.isfinite(x).all() for x in xs):raise ValueError('Finite native34 required')
    if not enabled or control>=RECOVERY:
        return np.zeros(14),np.zeros(3),np.zeros((3,10)),np.zeros(3),*[np.zeros(3) for _ in range(4)]
    h,A,_,_=geometry.measure(xs[0]);hn=geometry.measure(xs[1])[0]
    hi=geometry.measure(xs[2])[0];hz=geometry.measure(xs[3])[0]
    error=(h-hn)-(hi-hz);raw=np.zeros(14)
    if np.any(error!=0.):raw[geometry.legs]=-DT*np.linalg.pinv(A,rcond=1e-10)@error
    residual=error+A@(raw[geometry.legs]/DT)
    return raw,error,A,residual,h,hn,hi,hz

def limits(target,previous,lower,upper):
    return np.clip(np.clip(target,lower,upper),previous-SLEW*DT,previous+SLEW*DT)

def merge(fixed,reference,raw,legs,lower,upper):
    if not np.any(raw[legs]!=0.):return fixed
    result=fixed.copy()
    result[legs]=np.clip(reference[legs]+np.clip(fixed[legs]-reference[legs]+raw[legs],-.18,.18),lower[legs],upper[legs])
    return result

def jobs():
    paths=[]
    for root in (run.SMOKE,run.OUTPUT):
        names=('zero_parity','momentum_smoke') if root==run.SMOKE else ('zero_parity','momentum_smoke','development_baseline','development_candidate')
        for name in names:
            for row in read(root/name/'results.json')['rows']:paths.append(root/name/('case_'+str(row['case_seed'])))
    assert len(paths)==62 and len(set(paths))==62
    return paths

def smoke_jobs():
    return [run.OUTPUT/'zero_parity/case_None',run.OUTPUT/'momentum_smoke/case_None',
     run.OUTPUT/'development_baseline/case_769000',
     *[run.OUTPUT/'development_candidate'/('case_'+str(c)) for c in (769002,773004,769001)]]

def identity(path):return path.parent.parent.name+'__'+path.parent.name+'__'+path.name

def scalar(path,geometry,nominal,reference,selector,save=None):
    row=read(path/'result.json')
    with np.load(path/'trajectory.npz',allow_pickle=False) as z:data={key:z[key].copy() for key in FIELDS}
    obs=data['observations'];n=len(obs);assert n==row['controls'] and 0<n<=2279
    assert row['compiled_model_sha256']==COMPILED_SHA and row['compiled_model_unchanged']
    gains,choice,logits=run.selector.old.predict(selector,data['preparation_sensors'][-1])
    np.testing.assert_array_equal(gains,row['base_gains']);assert choice==row['base_choice']
    np.testing.assert_array_equal(logits,row['initial_sensor_logits'])
    previous=geometry.home.copy();initial=obs[0].copy();records={key:[] for key in SIGNALS};post=[];max_total=0.
    for k,x in enumerate(obs):
        np.testing.assert_array_equal(x[34:48],(previous-geometry.home).astype(np.float32))
        hip=run.local.local_feedback(x,nominal[k],[9,10,11],gains) if k<RECOVERY else np.zeros(3)
        # Already post-IMU/old-right-hip exact double input; upstream unsaved
        # double request cannot be reconstructed from float32 observations.
        fixed=data['original_double_pre_slew_target_rad'][k].copy()
        values=momentum(x[:34],nominal[k,:34],initial[:34],nominal[0,:34],geometry,k,row['enabled'])
        raw=values[0];adjusted=merge(fixed,reference[k],raw,geometry.legs,geometry.lower,geometry.upper)
        np.testing.assert_array_equal(adjusted[geometry.head],fixed[geometry.head])
        base=limits(fixed,previous,geometry.lower,geometry.upper)
        wanted=limits(adjusted,previous,geometry.lower,geometry.upper)
        for key,value in zip(SIGNALS,(*values,hip,wanted,base,adjusted-fixed,adjusted)):records[key].append(np.asarray(value).copy())
        post.append(wanted-base);total=float(np.abs(adjusted-reference[k]).max());max_total=max(max_total,total)
        assert total<=.18+1e-12
        np.testing.assert_array_equal(wanted,data['applied'][k]);previous=wanted
        if k==0:assert adjusted is fixed
    for key,values in records.items():np.testing.assert_array_equal(values,data[key],err_msg=str(path)+' '+key)
    assert max_total==row['maximum_combined_14_joint_correction_rad']
    rawmax=float(np.abs(records['momentum_request_rad']).max());postmax=float(np.abs(post).max())
    assert rawmax==row['maximum_raw_momentum_request_rad'] and postmax==row['maximum_same_state_post_slew_direct_difference_rad']
    for key in SIGNALS[:4]:
        if n>RECOVERY:np.testing.assert_array_equal(np.asarray(records[key])[RECOVERY:],np.zeros_like(np.asarray(records[key])[RECOVERY:]))
        if not row['enabled']:np.testing.assert_array_equal(records[key],np.zeros_like(records[key]))
    for key in (SIGNALS[0],SIGNALS[1],SIGNALS[3]):
        np.testing.assert_array_equal(records[key][0],np.zeros_like(records[key][0]))
        if row['case_seed'] is None:np.testing.assert_array_equal(records[key],np.zeros_like(records[key]))
    first=np.flatnonzero(np.any(np.asarray(post)!=0,axis=1))
    summary=dict(path=str(path),identity=identity(path),case_seed=row['case_seed'],enabled=row['enabled'],
     controls=n,success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],peaks=row['peaks'],
     entry_time_s=row['entry_time_s'],strict_tail_s=row['strict_tail_s'],
     momentum_and_original_limits_independently_bitwise=True,original_right_hip_feedback_bitwise=True,
     original_double_IMU_request_reconstructed=False,raw_request_max_rad=rawmax,
     pre_slew_direct_max_rad=float(np.abs(records['same_state_pre_slew_direct_new_rad']).max()),
     post_slew_direct_max_rad=postmax,first_nonzero_direct=int(first[0]) if len(first) else None,
     maximum_combined=max_total,residual_max=float(np.abs(records[SIGNALS[3]]).max()))
    if save is not None:
        save.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(save/'signals.npz',**{key:np.asarray(value) for key,value in records.items()},post_slew_direct_new_rad=np.asarray(post))
        write(save/'summary.json',summary)
    return summary

def terminal_pairing():
    rows=[]
    for case in run.CASES:
        cpath=run.OUTPUT/'development_candidate'/('case_'+str(case));bpath=run.OUTPUT/'development_baseline'/('case_'+str(case))
        opath=run.selector.OUTPUT/'candidate'/('case_'+str(case));c,b,o=[read(p/'result.json') for p in (cpath,bpath,opath)]
        assert c['initial_hash']==b['initial_hash']==o['initial_hash']
        # Isolated full-field equality only: qpos/qvel never enter planning.
        with np.load(cpath/'trajectory.npz',allow_pickle=False) as cz,np.load(bpath/'trajectory.npz',allow_pickle=False) as bz,np.load(opath/'trajectory.npz',allow_pickle=False) as oz:
            equal={key:bool(np.array_equal(bz[key],oz[key])) for key in oz.files};assert all(equal.values()) and b['peaks']==o['peaks']
            difference={key:bool(np.array_equal(cz[key],bz[key])) for key in cz.files}
        rows.append(dict(case_seed=case,initial_hash=c['initial_hash'],baseline_original_fields_bitwise=equal,
         candidate_baseline_field_equality=difference,candidate_success=c['success'],baseline_success=b['success'],
         candidate_valid=c['valid'],baseline_valid=b['valid'],candidate_peaks=c['peaks'],baseline_peaks=b['peaks']))
    return rows

def python_sources():
    paths={Path(__file__).resolve(),Path(__file__).with_name('test_getup_centroidal_momentum_audit_r189.py'),Path(__file__).with_name('launch_getup_centroidal_momentum_audit_r189.py')}
    paths.update(Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py' and Path(m.__file__).is_file())
    return sorted(paths)

def source_inventory(all_jobs):
    paths=set(python_sources());paths.update(Path(p) for p in run.model_files(run.local.prior.program.SCENE))
    for root in (run.OUTPUT,run.SMOKE):
        paths.update(root/'frozen'/name for name in ('initial_selector.npz','snapshot.npz','nominal_sensor_trajectory.npz'))
        paths.update(root/name for name in ('contract.json','results.json','input_hashes_after.json','startup_closed.json'))
        for manifest in [root/'contract.json',*root.glob('worker_source_manifest_*.json')]:
            doc=read(manifest);sources=doc['sources'] if manifest.name=='contract.json' else doc
            folder=root/'executed_sources' if manifest.name=='contract.json' else root/manifest.stem.replace('worker_source_manifest_','worker_executed_sources_')
            for p,info in sources.items():
                assert digest(p)==info['sha256']==digest(folder/info['copy']);paths.update((Path(p),folder/info['copy']))
        paths.add(root.with_suffix('.log'))
    paths.update((run.OUTPUT/'training_closed.json',run.local.prior.program.REFERENCE,run.local.prior.program.STAND))
    for p in all_jobs:paths.update((p/'result.json',p/'trajectory.npz'))
    for case in run.CASES:
        p=run.selector.OUTPUT/'candidate'/('case_'+str(case));paths.update((p/'result.json',p/'trajectory.npz'))
    for root in (ROOT/'outputs/getup_centroidal_momentum_r188_smoke_20261010',ROOT/'outputs/getup_centroidal_momentum_launcher_r188c_20261010'):
        paths.update(p for p in root.rglob('*') if p.is_file())
    paths.add(ROOT/'outputs/getup_centroidal_momentum_r188_smoke_20261010.log')
    return {str(p):digest(p) for p in sorted(paths)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>(10+.25)*1024**3
    assert digest(run.__file__)==STORED_MAIN_SHA and digest(run.kernel.__file__)==run.KERNEL_SHA
    all_jobs=jobs();selected=smoke_jobs() if args.smoke else all_jobs
    # Preload all data used below before imported-Python source capture.
    frozen=run.OUTPUT/'frozen'
    with np.load(frozen/'initial_selector.npz',allow_pickle=False) as z:selector={key:z[key].copy() for key in z.files}
    with np.load(frozen/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    with np.load(run.local.prior.program.REFERENCE,allow_pickle=False) as z:reference=z['targets'].copy()
    model=load_model();geometry=IndependentGeometry(model);model_before=model_digest(model)
    before=source_inventory(all_jobs);output.mkdir();src=output/'executed_sources';src.mkdir();manifest={}
    for p in python_sources():
        name=hashlib.sha256(str(p).encode()).hexdigest()[:16]+'_'+p.name
        shutil.copy2(p,src/name);sha=digest(p);assert digest(src/name)==sha
        manifest[str(p)]=dict(sha256=sha,copy=name)
    assert str(Path(__file__).resolve()) in manifest
    write(output/'executed_sources_manifest.json',manifest)
    for root in (run.OUTPUT,run.SMOKE):
        for path,expected in read(root/'contract.json')['hashes'].items():assert digest(path)==expected
    write(output/'contract.json',dict(simulation_only=True,read_only=True,new_dynamic_trajectories=0,smoke=args.smoke,
     audited_existing_trajectory_budget=6 if args.smoke else 62,storage_budget_gb=.25,reserve_gb=10,
     private_canonical_kinematics_comPos_angmomMat_jacSite_only=True,no_environment_forward_integration_contact_force=True,
     scalar_inputs=('observations','preparation_sensors','original_double_pre_slew_target_rad'),comparison_fields=FIELDS,
     saved_h_A_e_raw_adjusted_or_planned_not_used_as_decisions=True,original_IMU_double_request_reconstructed=False,
     historical_source_or_partial_gaps_recovered=False,source_hashes=before,compiled_model_sha256=model_before,
     executed_python_sources=manifest,native_binaries_not_startup_copied=True,
     full_task_completed=False,hardware_readiness=False,qualification=False))
    rows=[]
    for path in selected:
        save=output/'signals'/identity(path) if args.smoke or path in smoke_jobs() or path.parent.name=='development_candidate' else None
        rows.append(scalar(path,geometry,nominal,reference,selector,save));print('R189_SCALAR',identity(path),'bitwise',flush=True)
    pairs=terminal_pairing()
    if not args.smoke:
        prior=read(SMOKE/'results.json');subset=[next(r for r in rows if r['identity']==p['identity']) for p in prior['rows']]
        assert subset==prior['rows'] and pairs==prior['terminal_pairs']
        for row in subset:
            ident=row['identity']
            with np.load(output/'signals'/ident/'signals.npz',allow_pickle=False) as a,np.load(SMOKE/'signals'/ident/'signals.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    after=source_inventory(all_jobs);assert before==after and model_before==model_digest(model)
    for p,info in manifest.items():assert digest(p)==info['sha256']==digest(src/info['copy'])
    size=sum(p.stat().st_size for p in output.rglob('*') if p.is_file());assert size<.25*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    write(output/'source_hashes.json',dict(before=before,after=after,unchanged=True,compiled_model_unchanged=True))
    write(output/'results.json',dict(rows=rows,terminal_pairs=pairs,terminal_result_saved=True,
     existing_trajectories_audited=len(rows),controls_audited=sum(r['controls'] for r in rows),new_dynamic_trajectories=0,
     all_scalar_and_limits_bitwise=True,independent_smoke_matches=not args.smoke,source_evidence_unchanged=True,
     original_IMU_double_request_reconstructed=False,qualification=False,full_task_completed=False,hardware_readiness=False))
    print('R189_READ_ONLY_TERMINAL',len(rows),flush=True)

if __name__=='__main__':main()
