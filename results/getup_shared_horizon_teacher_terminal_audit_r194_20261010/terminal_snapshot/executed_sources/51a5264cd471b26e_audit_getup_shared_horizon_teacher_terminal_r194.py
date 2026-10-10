"""Independent existing R193 causal sensor, teacher selection and target audit; no dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import mujoco
import numpy as np
from diagnostics import train_getup_shared_horizon_teacher_r193 as run

ROOT=run.ROOT
OUTPUT=ROOT/'outputs/getup_shared_horizon_teacher_terminal_audit_r194_20261010'
SMOKE=ROOT/'outputs/getup_shared_horizon_teacher_terminal_audit_r194_smoke_20261010'
STORED_MAIN_SHA='8f26bc2e103f5489380c836fb02fc1e23e26e6485df95cd6830b7cc15ba7678b'
COMPILED_SHA='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
SIGNALS=('teacher_request_rad','teacher_causal_delta','teacher_waveform_rad','teacher_activation',
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
def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.write('\n')

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

class IndependentMapping:
    """Static home, joint bounds and actuator mapping only; no MjData."""
    def __init__(self,model):
        assert model.nu==14
        joints=model.actuator_trnid[:,0]
        qadr=model.jnt_qposadr[joints]
        self.home=model.keyframe('home').qpos[qadr].copy()
        self.legs=np.array([model.actuator(s+'_'+p).id for s in ('left','right') for p in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
        np.testing.assert_array_equal(self.legs,[0,1,2,3,4,9,10,11,12,13])
        self.head=np.array([i for i in range(14) if i not in self.legs])
        self.lower,self.upper=model.jnt_range[joints].T.copy()

BASIS=np.sin(np.pi*np.arange(529)[:,None]/528*np.arange(1,5)[None,:])
BASIS[[0,-1]]=0.;BASIS.setflags(write=False)
SCALE=np.r_[np.ones(3),np.full(3,.05),np.full(14,.05),np.full(14,.05)]
SCALE.setflags(write=False)

def teacher(current,nominal,initial,initial_nominal,coefficients,mapping,control):
    c=np.asarray(coefficients,dtype=float)
    if c.shape!=(10,4) or not np.isfinite(c).all() or np.abs(c).max()>1e-4:raise ValueError('Finite bounded ten-leg coefficients')
    xs=[np.asarray(x,dtype=np.float32) for x in (current,nominal,initial,initial_nominal)]
    if any(x.shape!=(34,) or not np.isfinite(x).all() for x in xs):raise ValueError('Finite native34 only')
    if not isinstance(control,(int,np.integer)) or control<0:raise ValueError('Nonnegative integer control')
    raw=np.zeros(14);delta=np.zeros(34);wave=np.zeros(10);a=0.
    if 0<control<RECOVERY:
        delta=((xs[0].astype(float)-xs[1].astype(float))-(xs[2].astype(float)-xs[3].astype(float)))/SCALE
        a=float(np.tanh(np.sqrt(np.mean(delta*delta))))
        wave=c@BASIS[control];raw[mapping.legs]=a*wave
    return raw,delta,wave,a

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
        names=['zero_parity','teacher_smoke']
        if root==run.OUTPUT:names += ['teacher_training/program_'+str(i).zfill(2) for i in range(5)]+['development_baseline','development_candidate']
        for name in names:
            for row in read(root/name/'results.json')['rows']:paths.append(root/name/('case_'+str(row['case_seed'])))
    assert len(paths)==187 and len(set(paths))==187
    return paths

def smoke_jobs():
    p=run.OUTPUT/'teacher_training/program_01'
    return [run.OUTPUT/'zero_parity/case_None',run.OUTPUT/'teacher_smoke/case_None',
            run.OUTPUT/'development_baseline/case_769000',
            p/'case_769002',p/'case_769001',p/'case_773009']

def identity(path):return str(path.relative_to(ROOT/'outputs')).replace('/','__')

def job_coefficients(path,selected_program):
    # Offline job routing only: parameters regenerated from the original proposal RNG.
    coeff,_=proposals();group=path.parent.name
    if group.startswith('program_'):index=int(group.removeprefix('program_'))
    elif group=='teacher_smoke':index=1
    elif group=='development_candidate':index=selected_program
    else:assert group in ('zero_parity','development_baseline');index=0
    return coeff[index]

def scalar(path,geometry,nominal,reference,selector,coefficients,save=None):
    row=read(path/'result.json')
    with np.load(path/'trajectory.npz',allow_pickle=False) as z:data={key:z[key].copy() for key in FIELDS}
    obs=data['observations'];n=len(obs);assert n==row['controls'] and 0<n<=2279
    assert row['compiled_model_sha256']==COMPILED_SHA and row['compiled_model_unchanged']
    assert row['physics_sha256']=='4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae' and row['physics_unchanged']
    np.testing.assert_array_equal(coefficients,row['teacher_coefficients_rad'])
    assert row['enabled']==bool(np.any(coefficients)) and row['root_edits_during_recovery']==0
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
        values=teacher(x[:34],nominal[k,:34],initial[:34],nominal[0,:34],coefficients,geometry,k)
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
    rawmax=float(np.abs(records['teacher_request_rad']).max());postmax=float(np.abs(post).max())
    assert rawmax==row['maximum_raw_teacher_request_rad'] and postmax==row['maximum_same_state_post_slew_direct_difference_rad']
    for key in SIGNALS[:4]:
        if n>RECOVERY:np.testing.assert_array_equal(np.asarray(records[key])[RECOVERY:],np.zeros_like(np.asarray(records[key])[RECOVERY:]))
    if not row['enabled']:np.testing.assert_array_equal(records[SIGNALS[0]],np.zeros_like(records[SIGNALS[0]]))
    for key in (SIGNALS[0],SIGNALS[1]):
        np.testing.assert_array_equal(records[key][0],np.zeros_like(records[key][0]))
        if row['case_seed'] is None:np.testing.assert_array_equal(records[key],np.zeros_like(records[key]))
    first=np.flatnonzero(np.any(np.asarray(post)!=0,axis=1))
    summary=dict(path=str(path),identity=identity(path),case_seed=row['case_seed'],enabled=row['enabled'],
     controls=n,success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],peaks=row['peaks'],
     entry_time_s=row['entry_time_s'],strict_tail_s=row['strict_tail_s'],
     teacher_and_original_limits_independently_bitwise=True,original_right_hip_feedback_bitwise=True,
     original_double_IMU_request_reconstructed=False,raw_request_max_rad=rawmax,
     pre_slew_direct_max_rad=float(np.abs(records['same_state_pre_slew_direct_new_rad']).max()),
     post_slew_direct_max_rad=postmax,first_nonzero_direct=int(first[0]) if len(first) else None,
     maximum_combined=max_total,causal_delta_max=float(np.abs(records[SIGNALS[1]]).max()))
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
        assert c==b
        # Isolated full-field equality only: qpos/qvel never enter planning.
        with np.load(cpath/'trajectory.npz',allow_pickle=False) as cz,np.load(bpath/'trajectory.npz',allow_pickle=False) as bz,np.load(opath/'trajectory.npz',allow_pickle=False) as oz:
            equal={key:bool(np.array_equal(bz[key],oz[key])) for key in oz.files};assert all(equal.values()) and b['peaks']==o['peaks']
            difference={key:bool(np.array_equal(cz[key],bz[key])) for key in cz.files};assert all(difference.values())
        rows.append(dict(case_seed=case,initial_hash=c['initial_hash'],baseline_original_fields_bitwise=equal,
         candidate_baseline_field_equality=difference,candidate_success=c['success'],baseline_success=b['success'],
         candidate_valid=c['valid'],baseline_valid=b['valid'],candidate_peaks=c['peaks'],baseline_peaks=b['peaks']))
    return rows

def python_sources():
    paths={Path(__file__).resolve(),Path(__file__).with_name('test_getup_shared_horizon_teacher_audit_r194.py'),Path(__file__).with_name('launch_getup_shared_horizon_teacher_audit_r194.py')}
    paths.update(Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py' and Path(m.__file__).is_file())
    return sorted(paths)

def source_inventory(all_jobs):
    paths=set(python_sources())
    paths.update(Path(p) for p in run.model_files(run.local.prior.program.SCENE))
    proof_path=ROOT/'tmp/getup_shared_horizon_teacher_r193_terminal_verification_20261010.json'
    proof=read(proof_path);paths.add(proof_path)
    assert proof['natural_exit'] and proof['formal_attempts']==181 and proof['independent_attempts']==6
    for root in (run.OUTPUT,run.SMOKE,ROOT/'outputs/getup_shared_horizon_teacher_launcher_r193_20261010'):
        paths.update(p for p in root.rglob('*') if p.is_file())
        for name,sha in proof['source_inventory'][root.name].items():assert digest(root/name)==sha
        manifests=[]
        if root!=ROOT/'outputs/getup_shared_horizon_teacher_launcher_r193_20261010':
            manifests.append((read(root/'contract.json')['sources'],root/'executed_sources'))
            for manifest in root.glob('worker_source_manifest_*.json'):
                manifests.append((read(manifest),root/manifest.stem.replace('worker_source_manifest_','worker_executed_sources_')))
        else:
            manifests.append((read(root/'contract.json')['sources'],root/'executed_sources'))
            for label in ('initial','formal'):manifests.append((read(root/label/'test_source_manifest.json'),root/label/'executed_sources'))
        for sources,folder in manifests:
            for p,info in sources.items():
                assert digest(p)==info['sha256']==digest(folder/info['copy'])
                paths.add(Path(p))
    paths.update((run.OUTPUT.with_suffix('.log'),run.SMOKE.with_suffix('.log'),ROOT/'tmp/getup_shared_horizon_teacher_r193_launcher_20261010.log'))
    paths.update((run.local.prior.program.REFERENCE,run.local.prior.program.STAND))
    for case in run.CASES:
        p=run.selector.OUTPUT/'candidate'/('case_'+str(case));paths.update((p/'result.json',p/'trajectory.npz'))
    for name in ('shared_horizon_teacher_kernel_r193.py','train_getup_shared_horizon_teacher_r193.py','test_getup_shared_horizon_teacher_r193.py','launch_getup_shared_horizon_teacher_r193.py','archive_getup_shared_horizon_teacher_terminal_r193.py','archive_getup_shared_horizon_teacher_terminal_r193b.py'):
        paths.add(Path(run.__file__).with_name(name))
    for name in ('GETUP_SHARED_HORIZON_TEACHER_PLAN_R193_20261010.md','GETUP_SHARED_HORIZON_TEACHER_TERMINAL_R193_20261010.md','getup_shared_horizon_teacher_r193_archive_verification_failure_01.json'):
        paths.add(ROOT/'tmp'/name)
    return {str(p):digest(p) for p in sorted(paths)}

def capture_sources(destination):
    destination.mkdir(parents=True,exist_ok=False);manifest={}
    for p in python_sources():
        name=hashlib.sha256(str(p).encode()).hexdigest()[:16]+'_'+p.name
        shutil.copy2(p,destination/name);sha=digest(p);assert digest(destination/name)==sha
        manifest[str(p)]=dict(sha256=sha,copy=name)
    return manifest

def aggregate(rows):
    return dict(rows=rows,successes=sum(r['success'] for r in rows if r['case_seed'] is not None),
                nominal_success=next(r['success'] for r in rows if r['case_seed'] is None),
                physical_failures=sum(not r['valid'] for r in rows),return_sum=float(sum(r['return_sum'] for r in rows)))

def retention(report,baseline):
    if len(report['rows'])!=len(baseline['rows']):raise ValueError('Complete paired rows')
    preserved=[];rescued=[];regressed=[]
    for x,b in zip(report['rows'],baseline['rows']):
        if x['case_seed']!=b['case_seed'] or x['initial_hash']!=b['initial_hash']:raise ValueError('Paired original initial hashes')
        if x['success'] and b['success']:preserved.append(x['case_seed'])
        if x['success'] and not b['success']:rescued.append(x['case_seed'])
        if b['success'] and not x['success']:regressed.append(x['case_seed'])
    return dict(preserved=preserved,rescued=rescued,regressed=regressed,eligible=bool(report['nominal_success'] and not report['physical_failures'] and not regressed and report['successes']>baseline['successes']))

def choose(reports):
    e=[retention(r,reports[0]) for r in reports];allowed=[i for i in range(1,len(e)) if e[i]['eligible']]
    def rank(i):
        returns=[r['return_sum'] for r in reports[i]['rows']]
        return reports[i]['successes'],min(returns),sum(returns),-i
    return (max(allowed,key=rank) if allowed else 0),e

def proposals():
    rng=np.random.default_rng(293)
    d=np.clip(rng.normal(0,1e-5,size=(2,10,4)),-1e-4,1e-4)
    return [np.zeros((10,4)),d[0],-d[0],d[1],-d[1]],rng.bit_generator.state

def common_selection():
    coeff,state=proposals();reports=[]
    for root in (run.OUTPUT,run.SMOKE):
        proposal=read(root/'proposal_rng.json')
        assert proposal['seed']==293 and proposal['rng']==state
        np.testing.assert_array_equal(proposal['coefficients'],coeff)
    for i in range(5):
        folder=run.OUTPUT/'teacher_training'/('program_'+str(i).zfill(2))
        report=aggregate([read(folder/('case_'+str(case))/'result.json') for case in run.CASES])
        assert report==read(folder/'results.json');reports.append(report)
        for r in report['rows']:np.testing.assert_array_equal(r['teacher_coefficients_rad'],coeff[i])
        selected,evidence=choose(reports);closed=read(run.OUTPUT/('closed_program_'+str(i).zfill(2)+'.json'))
        assert closed==dict(program=i,coefficients=coeff[i].tolist(),reports=reports,selected=selected,evidence=evidence,rng=state,full_training_attempts=25*(i+1))
    selected,evidence=choose(reports);training=read(run.OUTPUT/'training_closed.json');final=read(run.OUTPUT/'results.json')
    assert training==dict(shared_teacher=True,nonzero_search_trials=4,selected_program=selected,reports=reports,retention=evidence,formal_dynamic_attempts=181,rng=state)
    for name in ('baseline','candidate'):
        folder=run.OUTPUT/('development_'+name)
        report=aggregate([read(folder/('case_'+str(c))/'result.json') for c in run.CASES])
        assert report==final[name]==read(folder/'results.json')
        for r in report['rows']:np.testing.assert_array_equal(r['teacher_coefficients_rad'],coeff[0 if name=='baseline' else selected])
    assert final['selected_program']==selected and final['retention']==evidence
    e=retention(final['candidate'],final['baseline']);assert final['final_retention']==e
    gate=bool(final['candidate']['nominal_success'] and final['candidate']['successes']>=22 and final['candidate']['successes']>final['baseline']['successes'] and not final['candidate']['physical_failures'] and e['eligible'])
    assert final['original_development_gate']==gate and final['terminal_result_saved']
    assert not any(final[k] for k in ('full_task_completed','hardware_readiness','expanded_development_run','independent_qualification_run'))
    with np.load(run.OUTPUT/'selected_teacher.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['coefficients'],coeff[selected]);np.testing.assert_array_equal(z['basis'],BASIS)
    return dict(selected_program=selected,retention=evidence,proposal_rng=state,closed_programs=5,common_reports_independently_equal=True,original_development_gate=gate)

def old_field_equalities(all_jobs):
    annotations={'baseline','global_program_choice','gains','no_case_metadata_in_controller','original_complete_trace_bitwise_equal','model_group','selected_fixed_R133_program_bitwise_equal','R134_baseline_exact'}
    count=0
    for path in all_jobs:
        row=read(path/'result.json')
        if row['enabled'] and row['case_seed'] is not None:continue
        old=run.selector.OUTPUT/'candidate'/path.name;original=read(old/'result.json')
        assert set(original)-set(row)<=annotations
        for key in set(original)&set(row):assert row[key]==original[key],key
        # Strictly isolated equality; root arrays are never returned to scalar.
        with np.load(path/'trajectory.npz',allow_pickle=False) as x,np.load(old/'trajectory.npz',allow_pickle=False) as y:
            for key in y.files:np.testing.assert_array_equal(x[key],y[key],err_msg=key)
        count+=1
    assert count==87
    for name in ('zero_parity','teacher_smoke'):
        for row in read(run.SMOKE/name/'results.json')['rows']:
            case='case_'+str(row['case_seed'])
            a=run.OUTPUT/name/case;b=run.SMOKE/name/case
            assert read(a/'result.json')==read(b/'result.json')
            with np.load(a/'trajectory.npz',allow_pickle=False) as x,np.load(b/'trajectory.npz',allow_pickle=False) as y:
                assert x.files==y.files
                for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    return count

def budget():
    folders=(SMOKE,OUTPUT,ROOT/'outputs/getup_shared_horizon_teacher_audit_launcher_r194_20261010')
    live=sum(p.stat().st_size for folder in folders if folder.exists() for p in folder.rglob('*') if p.is_file())
    assert 3*live<.25*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3,'Live/archive/Git budget or reserve exhausted'
    return live

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
    model=load_model();geometry=IndependentMapping(model);model_before=model_digest(model)
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
     audited_existing_trajectory_budget=6 if args.smoke else 187,storage_budget_gb=.25,reserve_gb=10,
     no_private_MjData_geometry_RNE=True,no_environment_forward_inverse_integration_contact_distance_passive_actual_force=True,
     scalar_inputs=('observations','preparation_sensors','original_double_pre_slew_target_rad'),comparison_fields=FIELDS,
     saved_delta_activation_wave_raw_adjusted_or_planned_not_used_as_decisions=True,original_IMU_double_request_reconstructed=False,
     historical_source_or_partial_gaps_recovered=False,source_hashes=before,compiled_model_sha256=model_before,
     executed_python_sources=manifest,native_binaries_not_startup_copied=True,
     full_task_completed=False,hardware_readiness=False,qualification=False))
    rows=[]
    selection=common_selection();parity_count=old_field_equalities(all_jobs)
    write(output/'common_selection.json',selection)
    for path in selected:
        budget()
        save=output/'signals'/identity(path) if args.smoke or path in smoke_jobs() or not read(path/'result.json')['valid'] else None
        coefficients=job_coefficients(path,selection['selected_program'])
        rows.append(scalar(path,geometry,nominal,reference,selector,coefficients,save));print('R194_SCALAR',identity(path),'bitwise',flush=True)
    pairs=terminal_pairing()
    if not args.smoke:
        prior=read(SMOKE/'results.json');subset=[next(r for r in rows if r['identity']==p['identity']) for p in prior['rows']]
        assert subset==prior['rows'] and pairs==prior['terminal_pairs'] and selection==prior['common_selection'] and parity_count==prior['R157_old_field_parity_count']
        for row in subset:
            ident=row['identity']
            with np.load(output/'signals'/ident/'signals.npz',allow_pickle=False) as a,np.load(SMOKE/'signals'/ident/'signals.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
    after=source_inventory(all_jobs);assert before==after and model_before==model_digest(model)
    for p,info in manifest.items():assert digest(p)==info['sha256']==digest(src/info['copy'])
    size=budget()
    write(output/'source_hashes.json',dict(before=before,after=after,unchanged=True,compiled_model_unchanged=True))
    write(output/'results.json',dict(rows=rows,terminal_pairs=pairs,common_selection=selection,R157_old_field_parity_count=parity_count,terminal_result_saved=True,
     existing_trajectories_audited=len(rows),controls_audited=sum(r['controls'] for r in rows),new_dynamic_trajectories=0,
     all_scalar_and_limits_bitwise=True,independent_smoke_matches=not args.smoke,source_evidence_unchanged=True,
     original_IMU_double_request_reconstructed=False,qualification=False,full_task_completed=False,hardware_readiness=False))
    print('R194_READ_ONLY_TERMINAL',len(rows),flush=True)

if __name__=='__main__':main()
