"""Append-only R198 readonly scalar, parameter update, selection verification and archival."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import mujoco
from diagnostics import train_getup_parameter_pair_r198 as run

ROOT=run.ROOT;REPO=ROOT/'github/openduck-mini-'
DIRECTORIES=(run.PREFLIGHT,run.LAUNCH,run.SMOKE,run.OUTPUT)
PROOF=ROOT/'tmp/getup_parameter_pair_r198_terminal_verification_20261011.json'
BLOCKS=('actor_w0','actor_b0','actor_w1','actor_b1')
SHAPES=((101,16),(16,),(16,10),(10,))
SOURCES=('parameter_pair_kernel_r198.py','train_getup_parameter_pair_r198.py','test_getup_parameter_pair_r198.py','launch_getup_parameter_pair_r198.py')
MJB='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
PHYSICS='4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae'

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def inventory(path):return {str(p.relative_to(path)):digest(p) for p in sorted(path.rglob('*')) if p.is_file()}
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sources(manifest,folder):
    for p,v in manifest.items():assert digest(p)==v['sha256']==digest(folder/v['copy'])
def idle(base):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines():
        first=Path(line.split()[0]).name if line.split() else ''
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in line and any(s in line for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_')))
        assert 'publish_getup' not in line and 'git push' not in line

def features(obs,initial,k):
    return np.r_[np.tanh(obs[:50].astype(float)),np.tanh(initial[:50].astype(float)),min(k/528.,1.)]
def net(w,x):
    return np.tanh(np.tanh(x@w['actor_w0']+w['actor_b0'])@w['actor_w1']+w['actor_b1'])
def flatten(w):return np.concatenate([w[k].reshape(-1) for k in BLOCKS])
def unpack(t):
    w={};offset=0
    for key,shape in zip(BLOCKS,SHAPES):
        n=int(np.prod(shape));w[key]=t[offset:offset+n].reshape(shape).copy();offset+=n
    assert offset==1802;return w
def exact_weights(expected,path):
    actual=load(path);assert set(actual)==set(BLOCKS)
    for k in BLOCKS:np.testing.assert_array_equal(expected[k],actual[k])

def aggregate(path,cases):
    rows=[read(path/f'case_{case}'/'result.json') for case in cases]
    assert [r['case_seed'] for r in rows]==cases
    nominal=[r for r in rows if r['case_seed'] is None]
    report=dict(rows=rows,successes=sum(r['success'] for r in rows if r['case_seed'] is not None),
                nominal_success=nominal[0]['success'] if nominal else None,
                physical_failures=sum(not r['valid'] for r in rows),
                return_sum=float(sum(r['return_sum'] for r in rows)))
    assert report==read(path/'results.json')
    return report
def good(report,baseline):
    old={r['case_seed']:r for r in baseline['rows']}
    return bool(report['nominal_success'] and not report['physical_failures'] and report['successes']>baseline['successes'] and all(not old[r['case_seed']]['success'] or r['success'] for r in report['rows']))
def pair(a,b,result_equal=True):
    with np.load(a/'trajectory.npz',allow_pickle=False) as za,np.load(b/'trajectory.npz',allow_pickle=False) as zb:
        assert za.files==zb.files
        # Root fields are decoded only for this isolated equality comparison.
        for key in za.files:np.testing.assert_array_equal(za[key],zb[key],err_msg=key)
    if result_equal:
        assert read(a/'result.json')==read(b/'result.json')
        assert read(a/'rng.json')==read(b/'rng.json')

def learner(result,baseline):
    rng=np.random.default_rng(298)
    initial=dict(actor_w0=rng.normal(size=(101,16))/np.sqrt(101.),actor_b0=np.zeros(16),actor_w1=np.zeros((16,10)),actor_b1=np.zeros(10))
    for folder in (run.SMOKE,run.OUTPUT):exact_weights(initial,folder/'initial.npz')
    exact_weights(initial,run.OUTPUT/'checkpoint_0000.npz')
    theta=flatten(initial);rng=np.random.default_rng(29801);before=rng.bit_generator.state
    eps=rng.normal(size=(2,1802));after=rng.bit_generator.state
    np.testing.assert_array_equal(eps,load(run.OUTPUT/'directions.npz')['directions'])
    for folder in (run.SMOKE,run.OUTPUT):
        assert read(folder/'proposal_rng_before.json')==before and read(folder/'proposal_rng_after.json')==after
        exact_weights(unpack(theta+1e-5*eps[0]),folder/'smoke_probe.npz')
    history=[];candidates=[];scores=np.empty((2,2))
    for i in range(4):
        name=f'proposal_{i:04d}';sign=1 if i%2==0 else -1
        exact_weights(unpack(theta+sign*1e-5*eps[i//2]),run.OUTPUT/(name+'.npz'))
        report=aggregate(run.OUTPUT/name,run.CASES)
        score=float(np.mean(np.asarray([r['return_sum'] for r in report['rows']],dtype=np.float64)))
        scores[i//2,i%2]=score;eligible=good(report,baseline)
        history.append(dict(model=name,report=report,score=score,eligible=eligible,direction=i//2,sign=sign))
        if eligible:candidates.append((report['successes'],min(r['return_sum'] for r in report['rows']),sum(r['return_sum'] for r in report['rows']),-i,name))
    assert history==result['history']==read(run.OUTPUT/'training_closed_history.json')
    coefficients=(scores[:,0]-scores[:,1])/(2.*1e-5)
    gradient=np.mean(coefficients[:,None]*eps,axis=0)
    norm=float(np.linalg.norm(gradient));bounded=gradient/max(1.,norm)
    updated=unpack(theta+1e-4*bounded);exact_weights(updated,run.OUTPUT/'checkpoint_0001.npz')
    expected=dict(scores=scores,coefficients=coefficients,gradient=gradient,gradient_norm=norm,bounded_gradient=bounded)
    saved=load(run.OUTPUT/'parameter_update.npz');assert expected.keys()==saved.keys()
    for key,val in expected.items():np.testing.assert_array_equal(val,saved[key])
    changes={k:float(np.linalg.norm(updated[k]-initial[k])) for k in BLOCKS}
    assert all(v>0 for v in changes.values())
    assert read(run.OUTPUT/'parameter_update.json')==dict(changes=changes,gradient_norm=norm,updates=1,all_100_original_returns_including_failures_used=True,old_data_used=False,optimizer=False)
    assert result['parameter_changes']==changes and result['parameter_updates']==1
    check=aggregate(run.OUTPUT/'check_0001',run.CASES);eligible=good(check,baseline)
    assert check==result['updated_check'] and eligible==result['updated_eligible']
    assert read(run.OUTPUT/'check_closed.json')==dict(check=check,eligible=eligible,updates=1)
    if eligible:candidates.append((check['successes'],min(r['return_sum'] for r in check['rows']),sum(r['return_sum'] for r in check['rows']),-4,'checkpoint_0001'))
    selected=max(candidates)[4] if candidates else None
    assert result['selected_model']==selected and result['selected_R157_fallback']==(selected is None)
    return dict(initial_seed_and_all_four_blocks_exact=True,two_direction_rng_exact=True,all_four_proposals_exact=True,all_100_original_returns_including_failures_used=True,scores=scores.tolist(),coefficients=coefficients.tolist(),gradient_norm=norm,gradient_norm_bounded_update_all_four_blocks_exact=True,changes=changes,common_selection_exact=True,selected_model=selected)

def verify():
    before={d.name:inventory(d) for d in DIRECTORIES}
    launch=read(run.LAUNCH/'result.json');result=read(run.OUTPUT/'results.json')
    assert launch['natural_exit'] and launch['smoke_exit_code']==launch['formal_exit_code']==0 and not launch['formal_not_run'] and launch['source_hashes_unchanged']
    assert result['terminal_result_saved'] and result['actual_dynamic_attempts']==181
    for folder in (run.LAUNCH,run.SMOKE,run.OUTPUT):
        c=read(folder/'contract.json');sources(c['sources'],folder/'executed_sources')
        if folder!=run.LAUNCH:
            assert c['model_files']==run.utility.model_files(run.local.prior.program.SCENE)
            for p,sha in c['hashes'].items():assert digest(p)==sha
            a=read(folder/'input_hashes_after.json')
            assert a==dict(source_and_model_files_unchanged=True,hashes=c['hashes'],model_files=c['model_files'])
            for f in folder.glob('worker_source_manifest_*.json'):
                sources(read(f),f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json','')))
    receipt=read(ROOT/'tmp/getup_parameter_pair_r198_preflight_exit_20261011.json')
    assert receipt['observed_natural_exit_code']==0 and receipt['tests']==24 and receipt['seconds']==.603 and receipt['new_dynamic']==0
    for folder in (run.PREFLIGHT,run.LAUNCH/'initial',run.LAUNCH/'formal'):
        log=(folder/'regression.log').read_text();assert 'Ran 24 tests' in log and '\nOK\n' in log
        sources(read(folder/'source_manifest.json'),folder/'executed_sources')
        for name in SOURCES:
            paths=list((folder/'executed_sources').glob('*_'+name))
            assert len(paths)==1 and digest(paths[0])==digest(Path(run.__file__).with_name(name))
    # Static target/model arrays only. Never instantiate MjData or an environment.
    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    assert run.utility.model_digest(model)==MJB
    legs=np.array([model.actuator(s+'_'+part).id for s in ('left','right') for part in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
    head=np.array([i for i in range(14) if i not in legs])
    lower,upper=model.jnt_range[model.actuator_trnid[:,0]].T.copy()
    with np.load(run.local.prior.program.REFERENCE,allow_pickle=False) as z:reference=z['targets'].copy()
    selector_model=load(run.SMOKE/'frozen/initial_selector.npz')
    with np.load(run.SMOKE/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    baseline=aggregate(run.OUTPUT/'development_baseline',run.CASES)
    assert baseline==result['baseline'] and baseline['successes']==18 and baseline['nominal_success'] and not baseline['physical_failures']
    learner_proof=learner(result,baseline)
    counts=controls=baseline_paths=0;rows=[];cache={}
    for folder in (run.SMOKE,run.OUTPUT):
        reports={}
        for name,mode,cases,parity in run.SMOKE_SPEC:
            reports[name]=aggregate(folder/name,cases)
            for case in cases:pair(run.OUTPUT/name/f'case_{case}',run.SMOKE/name/f'case_{case}')
        assert all(not v['physical_failures'] and all(r['controls']==2279 for r in v['rows']) for v in reports.values())
        assert reports['zero_parity']['nominal_success'] and reports['initial_standard']['nominal_success']
        assert read(folder/'startup_closed.json')==dict(reports=reports,smoke_gate_passed=True,attempts=6,terminal_result_saved=True)
        if folder==run.SMOKE:
            assert read(folder/'results.json')==dict(smoke=True,reports=reports,smoke_gate_passed=True,actual_dynamic_attempts=6,terminal_result_saved=True,training_run=False,qualification_passed=False,full_task_completed=False,hardware_readiness=False)
        for path in sorted(folder.glob('*/case_*')):
            row=read(path/'result.json');mode=row['policy_mode'];counts+=1;controls+=row['controls']
            assert mode in ('baseline','deterministic')
            assert row['physics_unchanged'] and row['compiled_model_unchanged'] and row['physics_sha256']==PHYSICS and row['compiled_model_sha256']==MJB
            name=path.parent.name
            if name=='initial_probe':weightfile=folder/'smoke_probe.npz'
            elif name.startswith('proposal_'):weightfile=folder/(name+'.npz')
            elif name=='check_0001':weightfile=folder/'checkpoint_0001.npz'
            elif name=='candidate' and result['selected_model'] is not None:weightfile=folder/(result['selected_model']+'.npz')
            else:weightfile=folder/'initial.npz'
            assert digest(weightfile)==row['policy_model_sha256']
            if str(weightfile) not in cache:cache[str(weightfile)]=load(weightfile)
            w=cache[str(weightfile)];rng=np.random.default_rng(row['sampling_seed'])
            with np.load(path/'trajectory.npz',allow_pickle=False) as archive:
                z={key:archive[key] for key in archive.files if key not in ('qpos','qvel')}
                obs=z['observations'];assert len(obs)==row['controls']
                initial=obs[0];gains,choice,logits=run.selector.old.predict(selector_model,z['preparation_sensors'][-1])
                assert gains.tolist()==row['base_gains'] and choice==row['base_choice'] and logits.tolist()==row['initial_sensor_logits']
                assert z['construction_preparation_sensors'].shape==(40,55) and z['preparation_sensors'].shape==(40,55)
                maxima=[]
                for k in range(row['controls']):
                    right=run.local.local_feedback(obs[k],nominal[k],np.array([9,10,11]),gains) if k<529 else np.zeros(3)
                    np.testing.assert_array_equal(right,z['local_hip_extra_rad'][k])
                    active=mode!='baseline' and 0<k<529
                    assert bool(z['policy_active'][k])==active
                    mean=net(w,features(obs[k],initial,k)) if active else np.zeros(10)
                    for key,val in [('policy_mean',mean),('policy_noise',np.zeros(10)),('policy_requested_action',mean)]:np.testing.assert_array_equal(val,z[key][k])
                    fixed=z['original_double_pre_slew_target_rad'][k];adjusted=fixed.copy()
                    if active:adjusted[legs]=np.clip(reference[k,legs]+.18*mean,lower[legs],upper[legs])
                    np.testing.assert_array_equal(adjusted,z['adjusted_double_pre_slew_target_rad'][k])
                    np.testing.assert_array_equal(adjusted[head],fixed[head])
                    maxima.append(float(np.abs(adjusted-reference[k]).max()))
                    assert maxima[-1]<=.18+1e-12
                    if k:
                        previous=z['applied'][k-1]
                        planned=np.clip(np.clip(adjusted,lower,upper),previous-5.24*.02,previous+5.24*.02)
                        base=np.clip(np.clip(fixed,lower,upper),previous-5.24*.02,previous+5.24*.02)
                        np.testing.assert_array_equal(planned,z['planned_before_integration_rad'][k]);np.testing.assert_array_equal(base,z['same_state_unperturbed_planned_rad'][k])
                assert max(maxima)==row['maximum_combined_14_joint_correction_rad']
                np.testing.assert_array_equal(z['planned_before_integration_rad'],z['applied'])
                assert abs(sum(z['original_step_reward'])-row['return_sum'])<1e-10
                assert row['success']==bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8)
                oldpath=run.selector.OUTPUT/'candidate'/f"case_{row['case_seed']}";oldrow=read(oldpath/'result.json')
                assert row['initial_hash']==oldrow['initial_hash']
                with np.load(oldpath/'trajectory.npz',allow_pickle=False) as old:np.testing.assert_array_equal(z['applied'][0],old['applied'][0])
                if mode=='baseline':
                    baseline_paths+=1
                    with np.load(oldpath/'trajectory.npz',allow_pickle=False) as old:
                        for key in old.files:np.testing.assert_array_equal(archive[key] if key in ('qpos','qvel') else z[key],old[key])
                    for key in oldrow:
                        if key in row:assert row[key]==oldrow[key],key
            assert read(path/'rng.json')==rng.bit_generator.state
            rows.append(dict(group=folder.name+'/'+name,case=row['case_seed'],mode=mode,controls=row['controls'],success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],original_peaks=row['peaks'],combined_max=row['maximum_combined_14_joint_correction_rad']))
            if counts%25==0:print('R198_READONLY_PATHS',counts,flush=True)
    assert counts==187 and controls==422222 and baseline_paths==56
    candidate=aggregate(run.OUTPUT/'candidate',run.CASES);assert candidate==result['candidate']
    assert result['selected_model'] is None
    for case in run.CASES:pair(run.OUTPUT/'candidate'/f'case_{case}',run.OUTPUT/'development_baseline'/f'case_{case}',False)
    assert result['original_development_gate']==bool(candidate['nominal_success'] and not candidate['physical_failures'] and candidate['successes']>=22 and candidate['successes']>18)
    assert not any(result[k] for k in ('expanded_development_run','independent_qualification_run','qualification_passed','full_task_completed','hardware_readiness'))
    assert before=={d.name:inventory(d) for d in DIRECTORIES}
    live=sum(f.stat().st_size for d in DIRECTORIES for f in d.rglob('*') if f.is_file())
    assert 3*live<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    return dict(terminal_result_saved=True,natural_exit=True,new_dynamic=0,closed_paths=counts,controls=controls,baseline_paths_all_old_fields_bitwise=baseline_paths,policy_mean_action_zero_noise_rng_independently_exact=True,post_double_boundary_replacement_independently_exact=True,planning_controls_after_first_independently_exact=True,first_control_original_R157_applied_parity=True,all_planned_equals_applied=True,learner=learner_proof,original_before_request_IMU_double_reconstructed=False,whole_old_feedback_independently_recomputed=False,cross_implementation_bitwise_or_safety_claim=False,test_startup_mapping_manifest_verified=True,explicit_four_test_source_copies_verified=True,rows=rows,source_inventory=before,live_bytes=live,full_task_completed=False,hardware_readiness=False,qualification=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6]);idle(a.base);proof=verify()
    if a.verify:
        assert not PROOF.exists();run.local.write_json(PROOF,proof)
        print(json.dumps({k:v for k,v in proof.items() if k not in ('source_inventory','rows')},indent=2));return
    assert proof==read(PROOF)
    before_git=sum(f.stat().st_size for f in (REPO/'.git').rglob('*') if f.is_file())
    for d in DIRECTORIES:
        target=REPO/'results'/d.name/'terminal_snapshot';assert not target.exists();shutil.copytree(d,target)
        assert inventory(target)==proof['source_inventory'][d.name]
        log=d.with_suffix('.log') if d in (run.SMOKE,run.OUTPUT) else ROOT/'tmp/getup_parameter_pair_r198_launcher_20261011.log' if d==run.LAUNCH else None
        if log and log.exists():shutil.copy2(log,target/'process.log')
        run.local.write_json(target/'snapshot_metadata.json',dict(terminal_result_saved=True,natural_exit=True,new_dynamic=0,closed_paths=187,full_task_completed=False,hardware_readiness=False,qualification=False))
        run.local.write_json(target/'artifact_manifest.json',inventory(target))
        subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    for name in (*SOURCES,Path(__file__).name):
        target=REPO/'scripts'/name;assert not target.exists();shutil.copy2(Path(__file__).with_name(name),target)
        subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
    for name in ('GETUP_PARAMETER_PAIR_PLAN_R198_20261011.json','GETUP_PARAMETER_PAIR_TERMINAL_R198_20261011.json','publish_getup_parameter_pair_terminal_r198_20261011.py'):
        target=REPO/('scripts/'+name if name.endswith('.py') else name);assert not target.exists();shutil.copy2(ROOT/'tmp'/name,target)
        subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
    target=REPO/'results'/run.PREFLIGHT.name/'preflight_exit_receipt.json'
    assert not target.exists();shutil.copy2(ROOT/'tmp/getup_parameter_pair_r198_preflight_exit_20261011.json',target)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    target=REPO/'results'/run.OUTPUT.name/'terminal_verification.json';assert not target.exists();shutil.copy2(PROOF,target)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve finite R198 whole-policy parameter pairs and failed update'],cwd=REPO,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    after_git=sum(f.stat().st_size for f in (REPO/'.git').rglob('*') if f.is_file())
    archive=sum(f.stat().st_size for d in DIRECTORIES for f in (REPO/'results'/d.name).rglob('*') if f.is_file())
    assert proof['live_bytes']+archive+max(0,after_git-before_git)<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    print(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),live_bytes=proof['live_bytes'],archive_bytes=archive,git_increment_bytes=after_git-before_git,free_bytes=shutil.disk_usage(ROOT).free)),flush=True)
if __name__=='__main__':main()

