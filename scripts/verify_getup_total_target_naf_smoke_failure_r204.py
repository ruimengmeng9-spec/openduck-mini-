"""Independent read-only saved R204 smoke decisions; no dynamics, production decisions or learning."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import numpy as np
import mujoco
from diagnostics import train_getup_total_target_naf_r204 as run
ROOT=Path('/data/shijinsheng/open_duck')
REPO=ROOT/'github/openduck-mini-'
BASE='3b2ae6dc6d745cf02f2ba57c7552251305c1c179'
DEST=ROOT/'outputs/getup_total_target_naf_r204_smoke_failure_verification_20261011'
PROOF=ROOT/'tmp/getup_total_target_naf_r204_smoke_failure_verification_20261011.json'
DIRECTORIES=(run.NUMERIC,run.NUMERIC_SOURCE,run.PREFLIGHT,run.LAUNCH,run.SMOKE)
SPEC=[('zero_parity',[None,769000,773004]),('initial_standard',[None]),('initial_sample',[769002,773004])]
OLD_FIELDS=['pose','case_seed','initial_hash','actual_fallen_start','valid','training_success','controls','strict_tail_s','entry_time_s','return_sum','final','peaks','prefix_start','root_edits_after_initialization']
MJB='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
PHYSICS='4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def eq(a,b,label=''):np.testing.assert_array_equal(a,b,err_msg=label)
def ah(a):
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())
def inventory(p):return {str(f.relative_to(p)):digest(f) for f in sorted(p.rglob('*')) if f.is_file()}
def sources(m,folder):
    for p,v in m.items():assert digest(p)==v['sha256']==digest(folder/v['copy']),p
    return len(m)
def idle():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
        pid,args=line.strip().split(None,1)
        if int(pid)==os.getpid():continue
        first=Path(args.split()[0]).name
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in args and any(t in args for t in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_','verify_getup_'))),args
        assert 'publish_getup' not in args and 'git push' not in args,args
def resource(additional=0):
    files=set()
    for directory in (ROOT/'tmp',ROOT/'outputs',Path(run.__file__).parent):
        for item in directory.iterdir():
            if 'r204' in item.name.lower():
                if item.is_file():files.add(item)
                elif item.is_dir():files.update(f for f in item.rglob('*') if f.is_file())
    live=sum(p.stat().st_size for p in files);forecast=(640+640+128)*1024**2
    assert live+forecast+additional<=2*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
    return dict(live_bytes=live,future_archive_git_extra_bytes=forecast,additional_bound_bytes=additional,
        live_plus_forecast_and_additional_bytes=live+forecast+additional,free_bytes=shutil.disk_usage(ROOT).free)

def feature(o,initial,k):return np.r_[np.tanh(o[:50].astype(float)),np.tanh(initial[:50].astype(float)),min(k/528.,1.)]
def fresh():
    rng=np.random.default_rng(304);w={}
    for i,(a,b) in enumerate(((101,32),(32,16),(16,66))):
        w[f'q_w{i}']=rng.normal(0,1/np.sqrt(a),(a,b));w[f'q_b{i}']=np.zeros(b)
    w['q_w2'][:,1:]=0.
    r,c=np.tril_indices(10)
    w['q_b2'][11:]=np.where(r==c,np.log(np.expm1(.999)),0.)
    return w
def heads(w,x):
    h=x
    for i in range(3):
        h=h@w[f'q_w{i}']+w[f'q_b{i}']
        if i<2:h=np.tanh(h)
    raw=h[1:11];r,c=np.tril_indices(10);t=h[11:]
    values=np.where(r==c,np.logaddexp(0.,t)+.001,t)
    L=np.zeros((10,10));L[r,c]=values
    return np.asarray(h[0]),np.tanh(raw),L,raw

def report(folder,cases):
    saved=read(folder/'results.json');rows=[read(folder/f'case_{c}/result.json') for c in cases]
    assert rows==saved['rows']
    expected=dict(successes=sum(bool(r['success']) for r in rows if r['case_seed'] is not None),
        nominal_success=next((bool(r['success']) for r in rows if r['case_seed'] is None),None),
        physical_failures=sum(not r['valid'] for r in rows),return_sum=sum(r['return_sum'] for r in rows))
    for k,v in expected.items():assert saved[k]==v,k
    return saved

def verify():
    before={str(d):inventory(d) for d in DIRECTORIES};manifest_counts=[]
    ne=read(ROOT/'tmp/getup_total_target_naf_r204_numeric_exit_20261011.json')
    assert ne['natural_exit'] and ne['exit_code']==0 and ne['source_hashes_unchanged']
    nr=read(run.NUMERIC/'result.json');assert nr['passed'] and nr['tests_run']==24 and nr['skipped']==nr['errors']==nr['failures']==0
    for name in ('startup_source_manifest.json','source_manifest.json'):
        manifest_counts.append([str(run.NUMERIC/name),sources(read(run.NUMERIC/name),run.NUMERIC/'executed_sources')])
    for p,sha in ne['source_hashes'].items():assert digest(run.NUMERIC_SOURCE/p)==sha
    for folder in (run.LAUNCH,run.SMOKE):
        c=read(folder/'contract.json');manifest_counts.append([str(folder),sources(c['sources'],folder/'executed_sources')])
        if folder==run.SMOKE:
            for p,sha in c['model_files'].items():assert digest(p)==sha
            for p,sha in c['hashes'].items():assert digest(p)==sha
            assert read(folder/'input_hashes_after.json')['source_and_model_files_unchanged']
            manifest_counts.append([str(folder/'runtime_source_manifest.json'),sources(read(folder/'runtime_source_manifest.json'),folder/'executed_sources')])
        for f in folder.glob('worker_source_manifest_*.json'):
            dst=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''))
            manifest_counts.append([str(f),sources(read(f),dst)])
    for folder in (run.PREFLIGHT,run.LAUNCH/'initial'):
        c=read(folder/'interface_result.json')
        assert c['tests_run']==16 and c['skipped']==c['errors']==c['failures']==0 and c['source_hashes_unchanged']
        text=(folder/'regression.log').read_text();assert 'Ran 16 tests' in text and '\nOK\n' in text and 'skipped' not in text
        for name in ('source_manifest.json','runtime_source_manifest.json'):
            manifest_counts.append([str(folder/name),sources(read(folder/name),folder/'executed_sources')])
        for f in (folder/'worker_fixture').glob('worker_source_manifest_*.json'):
            dst=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''))
            manifest_counts.append([str(f),sources(read(f),dst)])
    for p in (ROOT/'tmp/getup_total_target_naf_r204_preflight_exit_20261011.json',ROOT/'tmp/getup_total_target_naf_r204_launch_request_exit_20261011.json',run.LAUNCH/'initial/regression_natural_exit.json'):
        c=read(p);assert c['natural_exit'] and c['exit_code']==0 and c['source_hashes_unchanged']
    se=read(run.LAUNCH/'smoke_natural_exit.json');assert se['natural_exit'] and se['exit_code']==0 and se['actual_subprocess_return_observed']
    le=read(ROOT/'tmp/getup_total_target_naf_r204_launcher_exit_20261011.json')
    assert le['natural_exit'] and le['exit_code']==0 and le['actual_launcher_subprocess_return_observed'] and le['source_hashes_unchanged']
    for p,sha in le['source_hashes'].items():assert digest(p)==sha
    lr=read(run.LAUNCH/'result.json')
    assert lr['formal_not_run'] and not lr['smoke_gate_passed'] and lr['formal_exit_code'] is None and lr['smoke_exit_code']==0
    assert not run.OUTPUT.exists() and not (run.LAUNCH/'formal').exists() and not (run.LAUNCH/'formal_natural_exit.json').exists()
    assert not list(run.SMOKE.glob('checkpoint_*'))
    weights=fresh();saved=load(run.SMOKE/'initial.npz');assert weights.keys()==saved.keys()
    for key,val in weights.items():eq(val,saved[key],key)
    assert sum(v.size for v in weights.values())==4914

    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    size=mujoco.mj_sizeModel(model);buffer=np.empty(size,dtype=np.uint8);mujoco.mj_saveModel(model,buffer=buffer)
    assert hashlib.sha256(buffer.tobytes()).hexdigest()==MJB
    legs=np.array([model.actuator(s+'_'+part).id for s in ('left','right') for part in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
    head=np.array([i for i in range(14) if i not in legs]);joints=model.actuator_trnid[:,0]
    lower,upper=model.jnt_range[joints].T.copy();home=model.keyframe('home').qpos[model.jnt_qposadr[joints]]
    with np.load(run.local.prior.program.REFERENCE,allow_pickle=False) as z:reference=z['targets'].copy()
    selector=load(run.SMOKE/'frozen/initial_selector.npz')
    with np.load(run.SMOKE/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    count=controls=baseline_count=0;rows=[]
    for folder in (run.SMOKE,):
        for path in sorted(folder.glob('*/case_*')):
            row=read(path/'result.json');mode=row['policy_mode'];n=row['controls'];count+=1;controls+=n
            group=path.parent.name
            weightfile=folder/'initial.npz'
            assert digest(weightfile)==row['policy_model_sha256']
            assert row['physics_unchanged'] and row['compiled_model_unchanged'] and row['physics_sha256']==PHYSICS and row['compiled_model_sha256']==MJB
            rng=np.random.default_rng(row['sampling_seed'])
            with np.load(path/'trajectory.npz',allow_pickle=False) as archive:
                z={k:archive[k] for k in archive.files if k not in ('qpos','qvel')};obs=z['observations'];assert len(obs)==n
                assert z['construction_preparation_sensors'].shape==z['preparation_sensors'].shape==(40,50)
                assert z['construction_preparation_times'].shape==(40,)
                eq(obs[1:,34:48],(z['applied'][:-1]-home).astype(np.float32))
                sensor=z['preparation_sensors'][-1].astype(float)
                logits=(sensor-selector['mean'])/selector['std']@selector['weight']+selector['bias']
                choice=int(np.argmax(logits));gains=selector['global_gains'][choice].copy()
                assert logits.tolist()==row['initial_sensor_logits'] and choice==row['base_choice'] and gains.tolist()==row['base_gains']
                for k in range(n):
                    ids=np.array([9,10,11]);current=obs[k].astype(np.float32);nom=nominal[k].astype(np.float32)
                    position=(current[6+ids]-nom[6+ids]).astype(float)/.05;velocity=(current[20+ids]-nom[20+ids]).astype(float)/.05
                    right=-.18*np.tanh(gains[:3]*position+gains[3:]*velocity) if k<529 else np.zeros(3)
                    eq(right,z['local_hip_extra_rad'][k],'right_feedback')
                    active=mode!='baseline' and 0<k<529;assert bool(z['policy_active'][k])==active
                    x=np.zeros(101);mean=np.zeros(10);noise=np.zeros(10);proposal=np.zeros(10);requested=np.zeros(10)
                    value=np.asarray(0.);raw=np.zeros(10);L=np.zeros((10,10))
                    if active:
                        x=feature(obs[k],obs[0],k);value,mean,L,raw=heads(weights,x)
                        if mode=='sample':noise=rng.normal(size=10);proposal=mean+1e-4*noise;requested=np.clip(proposal,-1.,1.)
                        else:proposal=mean.copy();requested=mean.copy()
                    for key,val in [('policy_features',x),('policy_mean',mean),('policy_noise',noise),('policy_unclipped_proposal',proposal),('policy_requested_action',requested),('policy_value',value),('policy_raw_mean',raw),('policy_lower',L)]:eq(val,z[key][k],key)
                    fixed=z['original_double_pre_slew_target_rad'][k];adjusted=fixed.copy()
                    if active:adjusted[legs]=np.clip(reference[k,legs]+.18*requested,lower[legs],upper[legs])
                    eq(adjusted,z['adjusted_double_pre_slew_target_rad'][k]);eq(adjusted[head],fixed[head])
                    if k:
                        prev=z['applied'][k-1]
                        planned=np.clip(np.clip(adjusted,lower,upper),prev-5.24*.02,prev+5.24*.02)
                        base=np.clip(np.clip(fixed,lower,upper),prev-5.24*.02,prev+5.24*.02)
                        eq(planned,z['planned_before_integration_rad'][k]);eq(base,z['same_state_unperturbed_planned_rad'][k])
                eq(z['planned_before_integration_rad'],z['applied'])
                assert float(np.abs(z['adjusted_double_pre_slew_target_rad']-reference[:n]).max())==row['maximum_combined_14_joint_correction_rad']
                assert abs(sum(z['original_step_reward'])-row['return_sum'])<1e-10
                assert row['success']==bool(row['valid'] and n==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8)
                oldpath=run.selector.OUTPUT/'candidate'/f"case_{row['case_seed']}";oldrow=read(oldpath/'result.json')
                assert row['initial_hash']==oldrow['initial_hash']
                with np.load(oldpath/'trajectory.npz',allow_pickle=False) as old:
                    eq(z['applied'][0],old['applied'][0])
                    if mode=='baseline':
                        baseline_count+=1
                        for key in old.files:eq(archive[key] if key in ('qpos','qvel') else z[key],old[key],key)
                        for key in OLD_FIELDS:assert row[key]==oldrow[key],key
                        for key in ('success','initial_sensor_logits'):assert row[key]==oldrow[key],key
            assert rng.bit_generator.state==read(path/'rng.json')
            rows.append(dict(group=folder.name+'/'+group,case=row['case_seed'],mode=mode,controls=n,success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],original_peaks=row['peaks'],combined_max=row['maximum_combined_14_joint_correction_rad'],all_original_arrays={key:ah(v) for key,v in z.items()},npz_sha256=digest(path/'trajectory.npz'),rng_sha256=digest(path/'rng.json'),raw_saved_rng_state=read(path/'rng.json')))
            print('R204_SAVED_PATH_VERIFIED',count,group,row['case_seed'],n,flush=True)

    assert count==6 and controls==11441 and baseline_count==3
    for path in run.SMOKE.glob('*/case_*'):
        m=read(path/'runtime_source_manifest.json')
        candidates=[p for p in run.SMOKE.glob('worker_executed_sources_*') if all((p/v['copy']).exists() for v in m.values())]
        assert candidates
        manifest_counts.append([str(path/'runtime_source_manifest.json'),sources(m,candidates[0])])
    startup=read(run.SMOKE/'startup_closed.json');result=read(run.SMOKE/'results.json')
    assert startup['reports']==result['reports'] and startup['attempts']==result['actual_dynamic_attempts']==6
    assert not startup['smoke_gate_passed'] and not result['smoke_gate_passed'] and not result['training_run']
    for name,cases in SPEC:assert report(run.SMOKE/name,cases)==startup['reports'][name]
    original_invalid=[r for r in rows if not r['valid']]
    assert len(original_invalid)==1 and original_invalid[0]['case']==769002 and original_invalid[0]['controls']==46
    assert not original_invalid[0]['success']
    assert before=={str(d):inventory(d) for d in DIRECTORIES}
    idle();resources=resource(8*1024**2)
    return dict(audit_complete=True,new_dynamic_attempts=0,verifier_source_sha256=digest(__file__),
        closed_saved_paths=count,controls=controls,baseline_paths_all_14_original_fields_and_arrays_exact=baseline_count,
        fresh_initial_Q_weights_all_blocks_exact={k:ah(v) for k,v in weights.items()},production_initial_npz_sha256=digest(run.SMOKE/'initial.npz'),
        all_causal101_V_raw_mean_mean_lower_iid_noise_unclipped_proposal_boxrequest_sampling_RNG_exact=True,det_no_RNG_consumption=True,
        selector_right_feedback_independently_exact=True,post_IMU_right_double_boundary_ten_leg_replacement_head_exact=True,
        planning_after_first_recursive_joint_slew_previous_exact=True,first_applied_R157_boundary_exact=True,all_planned_equals_applied=True,
        constructor_and_reset_each40x50=True,actual_smoke_and_launcher_natural_exit_zero=True,smoke_physical_gate_passed=False,
        original_invalid_labels_peaks_retained=original_invalid,formal_run=False,production_joint_Q_updates=0,
        production_initial_target_Adam_raw_checkpoint_not_saved=True,reference_initializer_not_fabricated_original_checkpoint=True,
        no_formal_checkpoints_replay_updates_deterministic_checks_final_or_fallback=True,
        source_manifest_counts=manifest_counts,source_inventory=before,source_evidence_hashes_unchanged=True,
        full_model_MJB_sha256=MJB,original_physics_sha256=PHYSICS,
        original_before_request_full_IMU_double_reconstructed=False,first_joint_slew_planning_independently_reconstructed=False,
        environment_MjData_forward_integrate_contact_force_relabel_invoked=False,production_kernel_decision_or_update_functions_invoked=False,
        same_numpy_not_cross_implementation_bitwise_or_physical_safety_guarantee=True,
        all_existing_original_arrays_present_not_rewritten=True,independent_full_arrays_compared_hashes_saved_no_duplicate_large_NPZ=True,
        dynamic_budget162_closed=True,unexecuted_formal156_closed=True,additional_dynamic_budget_allowed=False,rows=rows,resources=resources,
        archive_publisher_not_yet_implemented=True,best_remains_R157_18_of_24=True,full_task_completed=False,hardware_readiness=False,qualification_passed=False)

def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6]);idle()
    assert not DEST.exists() and not PROOF.exists()
    resource(32*1024**2)
    DEST.mkdir();copydir=DEST/'executed_sources';copydir.mkdir();manifest={}
    paths={Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).is_file():paths.add(Path(p).resolve())
    for p in sorted(paths):
        name=hashlib.sha256(str(p).encode()).hexdigest()[:16]+'_'+p.name
        shutil.copy2(p,copydir/name);manifest[str(p)]=dict(sha256=digest(p),copy=name)
    with (DEST/'source_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    try:
        result=verify();sources(manifest,copydir)
        runtime=dict(manifest)
        for module in list(sys.modules.values()):
            p=getattr(module,'__file__',None)
            if p and Path(p).suffix=='.py' and Path(p).is_file():
                path=Path(p).resolve()
                if str(path) not in runtime:
                    name=hashlib.sha256(str(path).encode()).hexdigest()[:16]+'_'+path.name
                    assert not (copydir/name).exists()
                    shutil.copy2(path,copydir/name);runtime[str(path)]=dict(sha256=digest(path),copy=name)
        sources(runtime,copydir)
        with (DEST/'runtime_source_manifest.json').open('x') as f:json.dump(runtime,f,indent=2)
        result['new_verifier_startup_source_count']=len(manifest)
        result['new_verifier_complete_runtime_source_count']=len(runtime)
        with PROOF.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        resource()
        print('R204_SMOKE_FAILURE_VERIFICATION_COMPLETE',result['closed_saved_paths'],result['controls'],digest(PROOF),flush=True)
    except Exception:
        with (DEST/'failure.json').open('x') as f:json.dump(dict(traceback=traceback.format_exc(),new_dynamic_attempts=0,proof_saved=PROOF.exists()),f,indent=2)
        raise
if __name__=='__main__':main()
