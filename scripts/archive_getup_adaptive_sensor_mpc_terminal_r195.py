"""Append-only R195 evidence, independent online recursion, no new simulation."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
import numpy as np
from diagnostics import train_getup_adaptive_sensor_mpc_r195 as run
ROOT=run.ROOT;REPO=ROOT/'github/openduck-mini-'
L=ROOT/'outputs/getup_adaptive_sensor_mpc_launcher_r195_20261010'
PROOF=ROOT/'tmp/getup_adaptive_sensor_mpc_r195_terminal_verification_20261010.json'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def inventory(root):return {str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
def idle(base):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines():
        first=Path(line.split()[0]).name if line.split() else ''
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in line and any(w in line for w in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_')))
        assert not ('publish_getup' in line or 'git push' in line)
def source_check(manifest,folder):
    for name,value in manifest.items():assert digest(name)==value['sha256']==digest(folder/value['copy'])

def verify():
    launch=read(L/'result.json');result=read(run.SMOKE/'results.json')
    assert launch['natural_exit'] and launch['smoke_exit_code']==0 and launch['formal_exit_code'] is None and launch['formal_not_run'] and launch['source_hashes_unchanged']
    assert not run.OUTPUT.exists() and result['terminal_result_saved'] and result['actual_dynamic_attempts']==8 and not result['prediction_controller_run']
    assert not result['causal_effect']['passed'] and result['causal_effect']['all_complete_valid'] and result['causal_effect']['pairs']==2
    contract=read(run.SMOKE/'contract.json');source_check(contract['sources'],run.SMOKE/'executed_sources')
    assert read(run.SMOKE/'input_hashes_after.json')['source_and_model_files_unchanged']
    assert contract['model_files']==run.model_files(run.local.prior.program.SCENE)
    for path,sha in contract['hashes'].items():assert digest(path)==sha
    for f in run.SMOKE.glob('worker_source_manifest_*.json'):
        source_check(read(f),f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json','')))
    source_check(read(L/'contract.json')['sources'],L/'executed_sources')
    source_check(read(L/'initial/test_source_manifest.json'),L/'initial/executed_sources')
    assert 'Ran 22 tests' in (L/'initial/regression.log').read_text() and '\nOK\n' in (L/'initial/regression.log').read_text()
    run.init_worker(run.SMOKE/'frozen')
    mapping=run.kernel.Mapping(__import__('mujoco').MjModel.from_xml_path(str(run.local.prior.program.SCENE)))
    rng=np.random.default_rng(295);noise=rng.normal(0,1e-5,(529,10));probe=rng.normal(size=10);probe/=np.linalg.norm(probe);noise[0]=0
    counts=controls=0;rows=[];probe_weights={};before=inventory(run.SMOKE)
    for source,name in [(run.selector.OUTPUT/'training/model.npz','initial_selector.npz'),(run.local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(run.local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:assert digest(source)==digest(run.SMOKE/'frozen'/name)
    for name,mode,cases,parity in run.SMOKE_GROUPS:
        for case in cases:
            path=run.SMOKE/name/f'case_{case}';row=read(path/'result.json')
            assert row['valid'] and row['controls']==2279 and not row['adaptive_fallback_controls']
            assert row['physics_sha256']=='4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae'
            assert row['compiled_model_sha256']=='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
            controls+=row['controls']
            with np.load(path/'trajectory.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['planned_before_integration_rad'],z['applied'])
                initial=z['observations'][0,:34].astype(float)-run.local.NOMINAL[0,:34].astype(float)
                theta=np.zeros((34,78));p=100*np.eye(78);previous=None;pending=None;change=np.zeros(34);updates=0
                gains,choice,logits=run.selector.old.predict(run.MODEL,z['preparation_sensors'][-1])
                np.testing.assert_array_equal(gains,row['base_gains']);assert choice==row['base_choice']
                for k in range(2279):
                    expected_feedback=run.local.local_feedback(z['observations'][k],run.local.NOMINAL[k],np.array([9,10,11]),gains) if k<529 else np.zeros(3)
                    np.testing.assert_array_equal(expected_feedback,z['local_hip_extra_rad'][k])
                    x=np.zeros(34);raw=np.zeros(14);error=np.zeros(34);features=np.zeros(78);prediction=np.zeros(34)
                    if mode!='zero' and k<529:
                        x=((z['observations'][k,:34].astype(float)-run.local.NOMINAL[k,:34].astype(float))-initial)/run.kernel.SCALE
                        if k==0:x[:]=0
                        change=np.zeros(34) if previous is None else x-previous
                        if pending is not None:
                            pf=p@pending;d=.995+pending@pf;error=change-theta@pending
                            theta+=np.outer(error,pf/d);p=(p-np.outer(pf,pf)/d)/.995;p=(p+p.T)*.5;updates+=1
                        if k==64:probe_weights[(mode,case)]=theta[:,68:].copy()
                        if k:
                            activation=float(np.tanh(np.sqrt(np.mean(x*x))));extra=activation*noise[k]
                            if k==64:extra+=activation*1e-5*probe*(1 if mode=='positive' else -1)
                            raw[mapping.legs]=np.clip(extra,-1e-4,1e-4)
                        features=np.r_[x,change,(z['applied'][k]-z['same_state_unperturbed_planned_rad'][k])[mapping.legs]/1e-4]
                        prediction=x+theta@features;pending=features.copy();previous=x.copy()
                    np.testing.assert_array_equal(raw,z['adaptive_request_rad'][k])
                    np.testing.assert_array_equal(x,z['adaptive_causal_delta'][k])
                    np.testing.assert_array_equal(error,z['adaptive_update_error'][k])
                    np.testing.assert_array_equal(features,z['adaptive_transition_input'][k])
                    np.testing.assert_array_equal(prediction,z['adaptive_one_step_prediction'][k])
                    np.testing.assert_array_equal(change if mode!='zero' and k<529 else np.zeros(34),z['adaptive_previous_change'][k])
                    assert np.abs(raw).max()<=1e-4
                with np.load(path/'online_terminal.npz',allow_pickle=False) as terminal:
                    np.testing.assert_array_equal(theta,terminal['theta']);np.testing.assert_array_equal(p,terminal['p'])
                assert row['adaptive_updates']==updates;counts+=updates
                assert read(path/'online_rng.json')==rng.bit_generator.state
                if parity or case is None:
                    previous_path=run.selector.OUTPUT/'candidate'/f'case_{case}'
                    with np.load(previous_path/'trajectory.npz',allow_pickle=False) as old:
                        for key in old.files:np.testing.assert_array_equal(z[key],old[key])
                    old_result=read(previous_path/'result.json')
                    assert row['peaks']==old_result['peaks'] and row['initial_hash']==old_result['initial_hash']
            rows.append(dict(mode=mode,case=case,controls=row['controls'],valid=row['valid'],success=row['success'],updates=updates,original_peaks=row['peaks'],initial_hash=row['initial_hash'],raw_max=row['maximum_raw_adaptive_request_rad'],combined_max=row['maximum_combined_14_joint_correction_rad']))
    # Pair metrics re-derived from true current inputs; outputs compared, never decisions.
    truths=[];estimates=[]
    for case in (769002,773004):
        pa=run.SMOKE/'positive_smoke'/f'case_{case}';pb=run.SMOKE/'negative_smoke'/f'case_{case}'
        with np.load(pa/'trajectory.npz',allow_pickle=False) as a,np.load(pb/'trajectory.npz',allow_pickle=False) as b:
            for key in ('observations','applied','normalized_residual','time','strict'):
                np.testing.assert_array_equal(a[key][:65 if key=='observations' else 64],b[key][:65 if key=='observations' else 64])
            for key in ('qpos','qvel'):np.testing.assert_array_equal(a[key][:64],b[key][:64])
            np.testing.assert_array_equal(probe_weights[('positive',case)],probe_weights[('negative',case)])
            difference=(a['applied'][64]-b['applied'][64])[mapping.legs];assert np.any(difference)
            truth=(a['observations'][65,:34].astype(float)-b['observations'][65,:34].astype(float))/run.kernel.SCALE
            estimate=probe_weights[('positive',case)]@(difference/1e-4)
            with np.load(pa/'held_probe_effect.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(truth,z['actual_scaled']);np.testing.assert_array_equal(estimate,z['predicted_scaled']);np.testing.assert_array_equal(difference,z['executed_difference_rad'])
            truths.append(truth);estimates.append(estimate)
    truth=np.asarray(truths);estimate=np.asarray(estimates);metrics={}
    for name,sl in [('gyro',slice(0,3)),('framezaxis',slice(3,6)),('q',slice(6,20)),('native_velocity',slice(20,34))]:
        zero=float(np.mean(truth[:,sl]**2));error=float(np.mean((estimate[:,sl]-truth[:,sl])**2))
        metrics[name]=dict(zero_effect_mse=zero,model_effect_mse=error,passes=bool(zero>0 and error<.8*zero))
    assert metrics==result['causal_effect']['metrics']
    pair=result['causal_effect']
    assert before==inventory(run.SMOKE),'Verify must not modify existing evidence'
    live=sum(f.stat().st_size for d in (run.SMOKE,L) for f in d.rglob('*') if f.is_file())
    assert controls==18232 and counts==2640 and 3*live<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    return dict(terminal_result_saved=True,natural_smoke_exit=True,new_closed_paths=8,controls=controls,online_realized_updates=counts,formal_run=False,prediction_controller_run=False,full_online_recursion_independently_bitwise=True,planned_equals_saved_applied=True,original_right_feedback_bitwise=True,original_IMU_double_request_reconstructed=False,whole_original_planning_independently_recomputed=False,causal_effect=pair,rows=rows,source_inventory={d.name:inventory(d) for d in (run.SMOKE,L)},live_bytes=live,full_task_completed=False,hardware_readiness=False,qualification=False,launcher_standard_gate_limitation='Perturbation-only groups have nominal_success null and launcher treats it as false; causal gate independently fails anyway. No source repair or dynamic rerun.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--verify',action='store_true');a=parser.parse_args()
    idle(a.base);proof=verify()
    if a.verify:
        assert not PROOF.exists();run.local.write_json(PROOF,proof)
        print(json.dumps({k:v for k,v in proof.items() if k not in ('source_inventory','rows')},indent=2));return
    assert read(PROOF)==proof
    before_git=sum(p.stat().st_size for p in (REPO/'.git').rglob('*') if p.is_file())
    for root in (run.SMOKE,L):
        target=REPO/'results'/root.name/'terminal_snapshot';assert not target.exists()
        shutil.copytree(root,target);assert inventory(root)==inventory(target)==proof['source_inventory'][root.name]
        log=ROOT/'tmp/getup_adaptive_sensor_mpc_r195_launcher_20261010.log' if root==L else root.with_suffix('.log')
        shutil.copy2(log,target/'process.log')
        run.local.write_json(target/'snapshot_metadata.json',dict(terminal_result_saved=True,natural_exit=True,new_closed_paths=8,formal_run=False,prediction_controller_run=False,source_gaps_recovered=False,full_task_completed=False,hardware_readiness=False,qualification=False))
        run.local.write_json(target/'artifact_manifest.json',inventory(target))
        subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    target=REPO/'results'/run.SMOKE.name/'terminal_verification.json';assert not target.exists();shutil.copy2(PROOF,target)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    for name in ('adaptive_sensor_mpc_kernel_r195.py','train_getup_adaptive_sensor_mpc_r195.py','test_getup_adaptive_sensor_mpc_r195.py','launch_getup_adaptive_sensor_mpc_r195.py',Path(__file__).name):
        target=REPO/'scripts'/name;assert not target.exists();shutil.copy2(Path(__file__).with_name(name),target);assert digest(target)==digest(Path(__file__).with_name(name))
        subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
    names=('GETUP_ADAPTIVE_SENSOR_MPC_PLAN_R195_20261010.md','GETUP_ADAPTIVE_SENSOR_MPC_TERMINAL_R195_20261010.md','publish_getup_adaptive_sensor_mpc_terminal_r195_20261010.py')
    for name in names:
        target=REPO/('scripts/'+name if name.endswith('.py') else name);assert not target.exists();shutil.copy2(ROOT/'tmp'/name,target)
        subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R195 online identification smoke and failed causal action-effect gate'],cwd=REPO,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    after_git=sum(p.stat().st_size for p in (REPO/'.git').rglob('*') if p.is_file())
    archive=sum(p.stat().st_size for d in (run.SMOKE,L) for p in (REPO/'results'/d.name).rglob('*') if p.is_file())
    assert proof['live_bytes']+archive+max(0,after_git-before_git)<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    print(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),live_bytes=proof['live_bytes'],archive_bytes=archive,git_increment_bytes=after_git-before_git,free_bytes=shutil.disk_usage(ROOT).free)),flush=True)
if __name__=='__main__':main()
