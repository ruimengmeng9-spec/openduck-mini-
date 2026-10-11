"""Independent saved-decision failure audit; never constructs an environment or MjData."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import mujoco
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jp
import optax
from jax.flatten_util import ravel_pytree
from flax import serialization
from diagnostics import train_getup_total_target_vmpo_r202 as run

ROOT=Path('/data/shijinsheng/open_duck')
REPO=ROOT/'github/openduck-mini-'
BASE='9b4edd92c86f0c689f15f904c0298cae52f17b82'
DEST=ROOT/'outputs/getup_total_target_vmpo_r202_interruption_verification_20261011'
PROOF=ROOT/'tmp/getup_total_target_vmpo_r202_interruption_verification_20261011.json'
REPAIR=ROOT/'outputs/getup_total_target_vmpo_launcher_r202b_20261011'
DIRECTORIES=(run.NUMERIC,run.NUMERIC_B,run.PREFLIGHT,run.LAUNCH,run.SMOKE,run.OUTPUT,run.NUMERIC_SOURCE,REPAIR)
CASES=[None,*range(769000,769008),*range(773000,773016)]
SPEC=[('zero_parity',[None,769000,773004]),('initial_standard',[None]),('initial_sample',[769002,773004])]
MJB='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
PHYSICS='4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def eq(a,b,label=''):np.testing.assert_array_equal(a,b,err_msg=label)
def inventory(p):return {str(f.relative_to(p)):digest(f) for f in sorted(p.rglob('*')) if f.is_file()}
def sources(m,folder):
    for p,v in m.items():assert digest(p)==v['sha256']==digest(folder/v['copy']),p
    return len(m)
def idle():
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines():
        first=Path(line.split()[0]).name if line.split() else ''
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in line and any(t in line for t in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_'))),line
        assert 'publish_getup' not in line and 'git push' not in line,line
def net(w,x,name,xp=np):
    for i in range(3):
        x=x@w[f'{name}_w{i}']+w[f'{name}_b{i}']
        if i<2:x=xp.tanh(x)
    return x
def feature(o,initial,k):return np.r_[np.tanh(o[:50].astype(float)),np.tanh(initial[:50].astype(float)),min(k/528.,1.)]
def density(z,mu,xp=np):return -.5*xp.sum(((z-mu)/1e-4)**2+2*np.log(1e-4)+np.log(2*np.pi),axis=-1)
def returns(rewards):
    out=np.empty_like(rewards,dtype=np.float64);total=0.
    for k in range(len(rewards)-1,-1,-1):total=rewards[k]+(.99**.2)*total;out[k]=total
    return out
def fresh():
    rng=np.random.default_rng(302);w={}
    for name,last in [('actor',10),('value',1)]:
        for i,(n,m) in enumerate([(101,32),(32,16),(16,last)]):
            w[f'{name}_w{i}']=np.zeros((n,m)) if name=='actor' and i==2 else rng.normal(0,1/np.sqrt(n),(n,m))
            w[f'{name}_b{i}']=np.zeros(m)
    return w
def report(folder,cases):
    saved=read(folder/'results.json');rows=[read(folder/f'case_{c}/result.json') for c in cases]
    assert saved['rows']==rows
    assert saved['successes']==sum(bool(r['success']) for r in rows if r['case_seed'] is not None)
    assert saved['nominal_success']==next((bool(r['success']) for r in rows if r['case_seed'] is None),None)
    assert saved['physical_failures']==sum(not r['valid'] for r in rows)
    assert saved['return_sum']==sum(r['return_sum'] for r in rows)
    return saved

def mc_batch(iteration,w):
    path=run.OUTPUT/f'training_{iteration:04d}';episodes=[]
    for case in CASES:
        with np.load(path/f'case_{case}/trajectory.npz',allow_pickle=False) as z:
            obs=z['observations'][:,:50].astype(float);n=len(obs)
            x=np.stack([feature(o,obs[0],k) for k,o in enumerate(obs)])
            reward=z['original_step_reward'];ret=returns(reward);v=net(w,x,'value')[:,0]
            active=(np.arange(n)>0)&(np.arange(n)<529);eq(active,z['policy_active'])
            episodes.append(dict(features=x,returns=ret,original_reward=reward,preupdate_value=v,raw_advantage=ret-v,active=active,latents=z['policy_latent'],control=np.arange(n,dtype=np.int64)))
    combined={k:np.concatenate([e[k] for e in episodes]) for k in episodes[0]}
    raw=combined['raw_advantage'];active=combined['active'];mean=float(raw[active].mean());std=float(raw[active].std());scale=std+1e-8
    adv=np.zeros_like(raw);adv[active]=(raw[active]-mean)/scale
    combined['normalized_advantage']=adv;combined['episode_offsets']=np.r_[0,np.cumsum([len(e['returns']) for e in episodes])]
    assert not (path/'complete_original_MC_batch.npz').exists()
    np.savez_compressed(DEST/'independent_saved_reward_MC_reference.npz',**combined)
    return combined

def initial_state_and_reference_MC():
    w=fresh()
    for path in (run.SMOKE/'initial.npz',run.OUTPUT/'initial.npz',run.OUTPUT/'checkpoint_0000.npz'):
        saved=load(path);assert saved.keys()==w.keys()
        for k,v in w.items():eq(v,saved[k],k)
    actor={k:jp.asarray(v) for k,v in w.items() if k.startswith('actor_')}
    value={k:jp.asarray(v) for k,v in w.items() if k.startswith('value_')}
    # Initialization only; no update, gradient, E-step, fitting or kernel Learner invocation.
    actor_state=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-7)).init(actor)
    value_state=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-4)).init(value)
    initial_bytes=serialization.to_bytes(dict(actor=actor,value=value,
        actor_state=actor_state,value_state=value_state,alpha=1.,
        actor_fit_attempts=0,actor_batches=0,value_updates=0))
    assert initial_bytes==(run.OUTPUT/'checkpoint_0000.learner.msgpack').read_bytes()
    rng=np.random.default_rng(302)
    assert rng.bit_generator.state==read(run.OUTPUT/'checkpoint_0000_rng.json')
    batch=mc_batch(1,w)
    assert len(batch['returns'])==55000 and int(batch['active'].sum())==12975
    missing=['learner_rng_before.json','complete_original_MC_batch.npz','advantage_normalization.json',
             'actor_update.json','actor_full_decisions.npz','value_update.json','value_full_updates.npz']
    assert all(not (run.OUTPUT/'training_0001'/p).exists() for p in missing)
    return dict(fresh302_all12_initial_blocks_exact=True,
        initial_actor_value_two_Adam_alpha_counts_full_msgpack_exact=True,
        initial_RNG_exact=True,original_checkpoint1_exists=False,
        completed_persisted_actor_updates=0,completed_persisted_value_updates=0,
        unobserved_in_memory_progress_not_claimed=True,
        independent_reference_MC_samples=55000,independent_reference_MC_active_samples=12975,
        independent_reference_MC_prevalue_advantage_original_home_invalid_terminal=True,
        independent_reference_MC_sha256=digest(DEST/'independent_saved_reward_MC_reference.npz'),
        original_MC_and_learning_decisions_saved=False,
        reference_MC_is_not_original_unsaved_training_trace=True,
        missing_original_evidence_not_filled=missing,
        Learner_update_gradient_Estep_fitting_invoked=False)


def verify():
    before={str(d):inventory(d) for d in DIRECTORIES};manifest_counts=[]
    manifest_counts.append([str(run.NUMERIC_B),sources(read(run.NUMERIC_B/'source_manifest.json'),run.NUMERIC_B/'executed_sources')])
    stopped=read(ROOT/'tmp/getup_total_target_vmpo_r202_user_authorized_task_stop_20261011.json')
    assert stopped['user_authorized_task_stop'] and stopped['formal_budget_consumed_not_restarted']
    assert not stopped['remaining_proc_entries'] and not stopped['natural_exit_zero_claim']
    assert not (run.LAUNCH/'result.json').exists() and not (run.OUTPUT/'results.json').exists()
    assert not (run.LAUNCH/'failure.json').exists() and not (run.OUTPUT/'failure.json').exists()
    for name in ('checkpoint_0001.npz','checkpoint_0001.learner.msgpack','checkpoint_0002.npz',
                 'check_0001','check_0002','training_0002','candidate','training_closed_history.json',
                 'input_hashes_after.json'):assert not (run.OUTPUT/name).exists()
    for folder in (run.LAUNCH,run.SMOKE,run.OUTPUT):
        c=read(folder/'contract.json');manifest_counts.append([str(folder),sources(c['sources'],folder/'executed_sources')])
        if folder!=run.LAUNCH:
            for p,sha in c['model_files'].items():assert digest(p)==sha
            for p,sha in c['hashes'].items():assert digest(p)==sha
            if folder==run.SMOKE:assert read(folder/'input_hashes_after.json')['source_and_model_files_unchanged']
            else:assert not (folder/'input_hashes_after.json').exists()
            for f in folder.glob('worker_source_manifest_*.json'):
                dest=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''));manifest_counts.append([str(f),sources(read(f),dest)])
    preflight=read(ROOT/'tmp/getup_total_target_vmpo_r202_preflight_exit_20261011.json');assert preflight['natural_exit'] and preflight['exit_code']==0 and preflight['new_dynamic_attempts']==0 and preflight['source_hashes_unchanged']
    for p,sha in preflight['source_hashes'].items():
        where=Path(p) if Path(p).is_absolute() else (ROOT/'tmp'/p if p.endswith('.json') else Path(run.__file__).with_name(p))
        assert digest(where)==sha,str(where)
    for folder in (run.PREFLIGHT,REPAIR/'complete_gate'):
        log=(folder/'regression.log').read_text();assert 'Ran 16 tests' in log and '\nOK\n' in log and 'skipped' not in log
        manifest_counts.append([str(folder),sources(read(folder/'source_manifest.json'),folder/'executed_sources')])
        for f in (folder/'worker_fixture').glob('worker_source_manifest_*.json'):
            dest=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''));manifest_counts.append([str(f),sources(read(f),dest)])
    for folder in (run.LAUNCH/'initial',run.LAUNCH/'formal'):
        log=(folder/'regression.log').read_text();assert 'Ran 16 tests' in log and 'OK (skipped=2)' in log
        assert not (folder/'source_manifest.json').exists()
    gate=read(REPAIR/'result.json');assert gate['passed'] and gate['tests_run']==16 and gate['skipped']==0
    gate_exit=read(ROOT/'tmp/getup_total_target_vmpo_r202b_gate_repair_exit_20261011.json')
    assert gate_exit['natural_exit'] and gate_exit['exit_code']==0 and gate_exit['new_dynamic_attempts']==0
    numeric_exit=read(ROOT/'tmp/getup_total_target_vmpo_r202b_numeric_exit_20261011.json')
    assert numeric_exit['natural_exit'] and numeric_exit['exit_code']==0 and numeric_exit['source_hashes_unchanged']
    numeric_result=read(run.NUMERIC_B/'result.json')
    assert numeric_result['passed'] and numeric_result['tests_run']==24
    for p,sha in numeric_exit['source_hashes'].items():assert digest(run.NUMERIC_SOURCE/p)==sha
    # Static model only: actuator mapping, home/ranges, full MJB bytes; no MjData.
    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    size=mujoco.mj_sizeModel(model);buffer=np.empty(size,dtype=np.uint8);mujoco.mj_saveModel(model,buffer=buffer);assert hashlib.sha256(buffer.tobytes()).hexdigest()==MJB
    legs=np.array([model.actuator(s+'_'+part).id for s in ('left','right') for part in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')]);head=np.array([i for i in range(14) if i not in legs])
    joints=model.actuator_trnid[:,0];lower,upper=model.jnt_range[joints].T.copy();home=model.keyframe('home').qpos[model.jnt_qposadr[joints]]
    with np.load(run.local.prior.program.REFERENCE,allow_pickle=False) as z:reference=z['targets'].copy()
    selector_model=load(run.SMOKE/'frozen/initial_selector.npz')
    with np.load(run.SMOKE/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    count=controls=baseline_count=0;rows=[];weight_cache={}
    for folder in (run.SMOKE,run.OUTPUT):
        for path in sorted(folder.glob('*/case_*')):
            row=read(path/'result.json');mode=row['policy_mode'];n=row['controls'];count+=1;controls+=n
            assert row['physics_unchanged'] and row['compiled_model_unchanged'] and row['physics_sha256']==PHYSICS and row['compiled_model_sha256']==MJB
            weightfile=run.OUTPUT/'checkpoint_0001.npz' if path.parent.name in ('training_0002','check_0001') else folder/'initial.npz'
            assert digest(weightfile)==row['policy_model_sha256']
            if str(weightfile) not in weight_cache:weight_cache[str(weightfile)]=load(weightfile)
            w=weight_cache[str(weightfile)];rng=np.random.default_rng(row['sampling_seed'])
            with np.load(path/'trajectory.npz',allow_pickle=False) as archive:
                z={k:archive[k] for k in archive.files if k not in ('qpos','qvel')};obs=z['observations'];assert len(obs)==n
                assert z['construction_preparation_sensors'].shape==z['preparation_sensors'].shape==(40,50)
                eq(obs[1:,34:48],(z['applied'][:-1]-home).astype(np.float32))
                initial=z['preparation_sensors'][-1].astype(float);logits=(initial-selector_model['mean'])/selector_model['std']@selector_model['weight']+selector_model['bias'];choice=int(np.argmax(logits));gains=selector_model['global_gains'][choice].copy()
                assert gains.tolist()==row['base_gains'] and choice==row['base_choice'] and logits.tolist()==row['initial_sensor_logits']
                for k in range(n):
                    ids=np.array([9,10,11]);current=obs[k].astype(np.float32);nom=nominal[k].astype(np.float32)
                    position=(current[6+ids]-nom[6+ids]).astype(float)/.05;velocity=(current[20+ids]-nom[20+ids]).astype(float)/.05
                    right=-.18*np.tanh(gains[:3]*position+gains[3:]*velocity) if k<529 else np.zeros(3)
                    eq(right,z['local_hip_extra_rad'][k],'right_feedback')
                    active=mode!='baseline' and 0<k<529;assert bool(z['policy_active'][k])==active
                    x=np.zeros(101);mean=noise=latent=np.zeros(10);logp=0.
                    if active:
                        x=feature(obs[k],obs[0],k);mean=net(w,x,'actor');noise=rng.normal(size=10) if mode=='sample' else np.zeros(10);latent=mean+1e-4*noise;logp=float(density(latent,mean))
                    requested=np.tanh(latent)
                    for key,val in [('policy_features',x),('policy_mean',mean),('policy_noise',noise),('policy_latent',latent),('policy_log_probability',logp),('policy_requested_action',requested)]:eq(val,z[key][k],key)
                    fixed=z['original_double_pre_slew_target_rad'][k];adjusted=fixed.copy()
                    if active:adjusted[legs]=np.clip(reference[k,legs]+.18*requested,lower[legs],upper[legs])
                    eq(adjusted,z['adjusted_double_pre_slew_target_rad'][k]);eq(adjusted[head],fixed[head])
                    if k:
                        prev=z['applied'][k-1];planned=np.clip(np.clip(adjusted,lower,upper),prev-5.24*.02,prev+5.24*.02);base=np.clip(np.clip(fixed,lower,upper),prev-5.24*.02,prev+5.24*.02)
                        eq(planned,z['planned_before_integration_rad'][k]);eq(base,z['same_state_unperturbed_planned_rad'][k])
                eq(z['planned_before_integration_rad'],z['applied']);assert float(np.abs(z['adjusted_double_pre_slew_target_rad']-reference[:n]).max())==row['maximum_combined_14_joint_correction_rad']
                assert abs(sum(z['original_step_reward'])-row['return_sum'])<1e-10
                assert row['success']==bool(row['valid'] and n==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8)
                oldpath=run.selector.OUTPUT/'candidate'/f"case_{row['case_seed']}";oldrow=read(oldpath/'result.json');assert row['initial_hash']==oldrow['initial_hash']
                with np.load(oldpath/'trajectory.npz',allow_pickle=False) as old:
                    eq(z['applied'][0],old['applied'][0])
                    if mode=='baseline':
                        baseline_count+=1
                        # qpos/qvel decoded only here for isolated old-field equality.
                        for key in old.files:eq(archive[key] if key in ('qpos','qvel') else z[key],old[key],key)
                        for key in oldrow:
                            if key in row:assert row[key]==oldrow[key],key
            assert rng.bit_generator.state==read(path/'rng.json')
            rows.append(dict(group=folder.name+'/'+path.parent.name,case=row['case_seed'],mode=mode,controls=n,success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],original_peaks=row['peaks'],combined_max=row['maximum_combined_14_joint_correction_rad']))
            print('R202_INTERRUPTION_PATH_VERIFIED',count,path.parent.name,row['case_seed'],n,flush=True)
    assert count==62 and controls==139323 and baseline_count==31
    for folder in (run.SMOKE,run.OUTPUT):
        startup=read(folder/'startup_closed.json');assert startup['smoke_gate_passed'] and startup['attempts']==6
        for name,cases in SPEC:assert report(folder/name,cases)==startup['reports'][name]
    for name,cases in SPEC:
        for case in cases:
            a=run.OUTPUT/name/f'case_{case}';b=run.SMOKE/name/f'case_{case}'
            with np.load(a/'trajectory.npz',allow_pickle=False) as za,np.load(b/'trajectory.npz',allow_pickle=False) as zb:
                assert za.files==zb.files
                # Full startup equality isolated from policy inference, including roots.
                for key in za.files:eq(za[key],zb[key],key)
            assert read(a/'result.json')==read(b/'result.json') and read(a/'rng.json')==read(b/'rng.json')
    baseline=report(run.OUTPUT/'development_baseline',CASES);train1=report(run.OUTPUT/'training_0001',CASES)
    assert baseline['successes']==18 and baseline['nominal_success'] and baseline['physical_failures']==0
    assert train1['successes']==8 and not train1['nominal_success'] and train1['physical_failures']==1
    learner=initial_state_and_reference_MC();print('R202_INITIAL_STATE_REFERENCE_MC_VERIFIED',flush=True)
    assert before=={str(d):inventory(d) for d in DIRECTORIES}
    idle()
    allfiles=set()
    for directory in (ROOT/'outputs',ROOT/'tmp'):
        for item in directory.iterdir():
            if 'r202' in item.name.lower():
                if item.is_file():allfiles.add(item)
                elif item.is_dir():allfiles.update(f for f in item.rglob('*') if f.is_file())
    allfiles.update(f for f in Path(run.__file__).parent.iterdir() if f.is_file() and 'r202' in f.name.lower())
    allfiles.add(Path(__file__))
    live=sum(f.stat().st_size for f in allfiles);forecast=(640+640+128)*1024**2
    assert live+forecast<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    return dict(verifier_source_sha256=digest(__file__),audit_complete=True,new_dynamic_attempts=0,
        original_formal_interrupted_by_user_authorization=True,formal_natural_exit_zero=False,
        independent_smoke_exit_zero_evidence='Immutable original launcher subprocess returncode-zero assertion and formal transition',
        standalone_smoke_or_launcher_natural_exit_receipt_present=False,
        closed_saved_paths=62,controls=controls,baseline_paths_all_old_fields_exact=31,
        all_causal_features_means_latents_noise_density_requests_rng_exact=True,
        selector_and_right_feedback_independently_exact=True,
        post_double_boundary_leg_replacement_and_head_exact=True,
        planning_after_first_joint_slew_recursive_previous_exact=True,
        first_applied_R157_boundary_exact=True,all_planned_equals_applied=True,
        constructor_and_reset_each_40x50=True,
        formal_and_independent_startup_all_arrays_results_rng_exact=True,
        initial_state_and_independent_reference_MC=learner,
        no_R202_final_candidate_or_fallback=True,budget_cap=162,unexecuted_budget=100,
        unexecuted_budget_may_be_used=False,baseline_successes=18,first_training_successes=8,
        first_training_standard_success=False,first_training_invalid=1,
        original_launcher_initial_formal_each16tests_with2skips_preserved=True,
        corrected_gate16tests_zero_skips_post_interruption_not_retroactive=True,
        source_manifest_counts=manifest_counts,source_inventory=before,
        source_evidence_hashes_unchanged=True,full_model_MJB_sha256=MJB,
        original_physics_sha256=PHYSICS,
        original_before_request_full_IMU_double_reconstructed=False,
        first_joint_slew_planning_independently_reconstructed=False,
        original_missing_MC_actor_value_and_final_not_filled=True,
        cross_implementation_bitwise_or_physical_safety_claim=False,rows=rows,
        live_bytes_all_R202_sources_logs_outputs_including_repair_verifier=live,
        future_archive_git_extra_bytes=forecast,live_plus_forecast_bytes=live+forecast,
        free_bytes=shutil.disk_usage(ROOT).free,archive_and_Git_not_yet_created=True,
        best_remains_R157_18_of_24=True,full_task_completed=False,hardware_readiness=False,qualification=False)

def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6]);idle()
    assert not DEST.exists() and not PROOF.exists() and shutil.disk_usage(ROOT).free>10*1024**3
    DEST.mkdir();copydir=DEST/'executed_sources';copydir.mkdir();manifest={}
    paths={Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        p=getattr(module,'__file__',None)
        if p and Path(p).suffix=='.py' and Path(p).is_file():paths.add(Path(p).resolve())
    for p in sorted(paths):
        name=hashlib.sha256(str(p).encode()).hexdigest()[:16]+'_'+p.name;shutil.copy2(p,copydir/name);manifest[str(p)]=dict(sha256=digest(p),copy=name)
    with (DEST/'source_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    try:
        result=verify();sources(manifest,copydir)
        with PROOF.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print('R202_INTERRUPTION_VERIFICATION_COMPLETE',result['closed_saved_paths'],result['controls'],digest(PROOF),flush=True)
    except Exception:
        import traceback
        with (DEST/'failure.json').open('x') as f:json.dump(dict(traceback=traceback.format_exc(),new_dynamic_attempts=0,proof_saved=PROOF.exists()),f,indent=2)
        raise
if __name__=='__main__':main()


