"""Readonly scalar/learner reconstruction and append-only terminal archival."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
import mujoco
from diagnostics import train_getup_total_target_ppo_r196 as run

ROOT=run.ROOT; REPO=ROOT/'github/openduck-mini-'
PREFLIGHT=ROOT/'outputs/getup_total_target_ppo_r196_preflight_20261010'
PROOF=ROOT/'tmp/getup_total_target_ppo_r196_terminal_verification_20261010.json'
DIRECTORIES=(PREFLIGHT,run.LAUNCH,run.SMOKE,run.OUTPUT)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def inventory(path):return {str(p.relative_to(path)):digest(p) for p in sorted(path.rglob('*')) if p.is_file()}
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def idle(base):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines():
        first=Path(line.split()[0]).name if line.split() else ''
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in line and any(s in line for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_')))
        assert 'publish_getup' not in line and 'git push' not in line


def sources(manifest,folder):
    for p,v in manifest.items():assert digest(p)==v['sha256']==digest(folder/v['copy'])


def net(weights,x,name,xp=np):
    for i in range(3):
        x=x@weights[f'{name}_w{i}']+weights[f'{name}_b{i}']
        if i<2:x=xp.tanh(x)
    return x


def sensor_features(obs,initial,k):
    return np.r_[np.tanh(obs[:50].astype(float)),np.tanh(initial[:50].astype(float)),min(k/528.,1.)]


def independent_learning(result):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jp
    import optax
    from flax import serialization
    p={k:jp.asarray(v) for k,v in load(run.OUTPUT/'initial.npz').items()}
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-6));state=optimizer.init(p)
    rng=np.random.default_rng(296);proof=[]
    initial_bytes=serialization.to_bytes(dict(weights=p,state=state))
    assert initial_bytes==(run.OUTPUT/'checkpoint_0000.learner.msgpack').read_bytes()
    def loss(params,x,z,oldlog,adv,ret,mask):
        mean=net(params,x,'actor',jp)
        lp=-.5*jp.sum(((z-mean)/1e-4)**2+2*np.log(1e-4)+np.log(2*np.pi),axis=-1)
        ratio=jp.exp(jp.clip(lp-oldlog,-40.,40.));clipped=jp.clip(ratio,.8,1.2)
        policy=-jp.sum(mask*jp.minimum(ratio*adv,clipped*adv))/jp.maximum(jp.sum(mask),1.)
        value=jp.mean((net(params,x,'critic',jp)[:,0]-ret)**2)
        kl=jp.sum(mask*(oldlog-lp))/jp.maximum(jp.sum(mask),1.)
        return policy+.5*value,jp.array([policy,value,kl])
    gradient=jax.jit(jax.value_and_grad(loss,has_aux=True))
    for iteration in (1,2):
        path=run.OUTPUT/f'training_{iteration:04d}';chunks=[[] for _ in range(6)]
        before={k:np.asarray(v).copy() for k,v in p.items()}
        for case in run.CASES:
            with np.load(path/f'case_{case}/trajectory.npz',allow_pickle=False) as archive:
                obs=archive['observations'];x=np.stack([sensor_features(o,obs[0],k) for k,o in enumerate(obs)])
                rewards=archive['original_step_reward'];ret=np.empty(len(rewards));total=0.
                for k in range(len(rewards)-1,-1,-1):total=rewards[k]+(.99**.2)*total;ret[k]=total
                mean=net(before,x,'actor');mask=archive['policy_active'].astype(float)
                np.testing.assert_allclose(mean[mask>0],archive['policy_mean'][mask>0],atol=1e-12,rtol=0)
                values=net(before,x,'critic')[:,0]
                vals=(x,archive['policy_latent'],archive['policy_log_probability'],ret-values,ret,mask)
                for target,value in zip(chunks,vals):target.append(value)
        arrays=[np.concatenate(v) for v in chunks];active=arrays[5]>0
        arrays[3]=(arrays[3]-arrays[3][active].mean())/(arrays[3][active].std()+1e-8)
        saved=load(path/'learner_batch.npz')
        for key,a in zip(('features','latent','old_log_probability','advantage','returns','actor_mask'),arrays):np.testing.assert_array_equal(a,saved[key])
        stats=[];norms=[];epochs=0;last=None
        for epoch in range(4):
            order=rng.permutation(len(arrays[0]))
            for start in range(0,len(order),512):
                ix=order[start:start+512]
                (_,s),g=gradient(p,*[jp.asarray(a[ix]) for a in arrays]);norms.append(float(optax.global_norm(g)));last=g
                update,state=optimizer.update(g,state,p);p=optax.apply_updates(p,update);stats.append(np.asarray(s))
            epochs+=1
            _,full=loss(p,*[jp.asarray(a) for a in arrays])
            if float(full[2])>.02:break
        saved=load(path/'learner_update.npz')
        np.testing.assert_array_equal(stats,saved['statistics']);np.testing.assert_array_equal(norms,saved['gradient_norms'])
        for k,v in last.items():np.testing.assert_array_equal(v,saved['last_gradient_'+k])
        checkpoint=load(run.OUTPUT/f'checkpoint_{iteration:04d}.npz')
        for k,v in p.items():np.testing.assert_array_equal(v,checkpoint[k])
        assert serialization.to_bytes(dict(weights=p,state=state))==(run.OUTPUT/f'checkpoint_{iteration:04d}.learner.msgpack').read_bytes()
        assert rng.bit_generator.state==read(run.OUTPUT/f'checkpoint_{iteration:04d}_rng.json')
        changes={k:float(np.linalg.norm(np.asarray(p[k])-before[k])) for k in before}
        h=result['history'][iteration-1];assert changes==h['changes'] and epochs==h['epochs'] and len(stats)==h['optimizer_updates']
        proof.append(dict(iteration=iteration,epochs=epochs,updates=len(stats),learner_batch_exact=True,gradient_statistics_exact=True,weights_exact=True,optimizer_serialization_bytes_exact=True,rng_exact=True,changes=changes))
    return proof


def verify():
    before={d.name:inventory(d) for d in DIRECTORIES}
    launch=read(run.LAUNCH/'result.json');result=read(run.OUTPUT/'results.json')
    assert launch['natural_exit'] and launch['smoke_exit_code']==launch['formal_exit_code']==0 and not launch['formal_not_run']
    assert result['terminal_result_saved'] and result['actual_dynamic_attempts']==156 and not result['original_development_gate']
    for folder in (run.LAUNCH,run.SMOKE,run.OUTPUT):
        c=read(folder/'contract.json');sources(c['sources'],folder/'executed_sources')
        if folder!=run.LAUNCH:
            assert c['model_files']==run.utility.model_files(run.local.prior.program.SCENE)
            for p,sha in c['hashes'].items():assert digest(p)==sha
            assert read(folder/'input_hashes_after.json')['source_and_model_files_unchanged']
            for f in folder.glob('worker_source_manifest_*.json'):
                dest=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''));sources(read(f),dest)
    assert read(PREFLIGHT/'exit.json')==dict(natural_exit=True,exit_code=0,new_dynamic=0)
    for folder in (PREFLIGHT,run.LAUNCH/'initial',run.LAUNCH/'formal'):
        log=(folder/'regression.log').read_text();assert 'Ran 20 tests' in log and '\nOK\n' in log
        for name in ('total_target_ppo_kernel_r196.py','train_getup_total_target_ppo_r196.py','test_getup_total_target_ppo_r196.py','launch_getup_total_target_ppo_r196.py'):
            candidates=list((folder/'executed_sources').glob('*_'+name));assert len(candidates)==1 and digest(candidates[0])==digest(Path(run.__file__).with_name(name))
    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    assert run.utility.model_digest(model)=='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
    legs=np.array([model.actuator(s+'_'+part).id for s in ('left','right') for part in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
    head=np.array([i for i in range(14) if i not in legs])
    joints=model.actuator_trnid[:,0];lower,upper=model.jnt_range[joints].T.copy()
    # StrictSim lower/upper are joint ranges mapped in actuator order.
    counts=controls=baseline_paths=0;rows=[];weight_cache={}
    for folder in (run.SMOKE,run.OUTPUT):
        for path in sorted(folder.glob('*/case_*')):
            row=read(path/'result.json');mode=row['policy_mode'];counts+=1;controls+=row['controls']
            assert row['physics_unchanged'] and row['compiled_model_unchanged']
            assert row['physics_sha256']=='4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae'
            assert row['compiled_model_sha256']=='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
            if path.parent.name=='training_0002':weightfile=run.OUTPUT/'checkpoint_0001.npz'
            elif path.parent.name in ('check_0001','check_0002'):weightfile=run.OUTPUT/f'checkpoint_{path.parent.name[-4:]}.npz'
            else:weightfile=folder/'initial.npz'
            assert digest(weightfile)==row['policy_model_sha256']
            if str(weightfile) not in weight_cache:weight_cache[str(weightfile)]=load(weightfile)
            w=weight_cache[str(weightfile)];rng=np.random.default_rng(row['sampling_seed'])
            with np.load(path/'trajectory.npz',allow_pickle=False) as archive:
                # Root data is not decoded for decisions; only separate equality below.
                z={k:archive[k] for k in archive.files if k not in ('qpos','qvel')}
                obs=z['observations'];initial=obs[0]
                for k in range(row['controls']):
                    active=mode!='baseline' and 0<k<529
                    assert bool(z['policy_active'][k])==active
                    mean=np.zeros(10);noise=np.zeros(10);latent=np.zeros(10);logp=0.
                    if active:
                        mean=net(w,sensor_features(obs[k],initial,k),'actor')
                        noise=rng.normal(size=10) if mode=='sample' else np.zeros(10)
                        latent=mean+1e-4*noise
                        logp=float(-.5*np.sum(((latent-mean)/1e-4)**2+2*np.log(1e-4)+np.log(2*np.pi)))
                    for key,val in [('policy_mean',mean),('policy_noise',noise),('policy_latent',latent),('policy_log_probability',logp)]:np.testing.assert_array_equal(val,z[key][k])
                    fixed=z['original_double_pre_slew_target_rad'][k];adjusted=fixed.copy()
                    # Reference file contains its original 2279 target sequence.
                    if active:
                        adjusted[legs]=np.clip(reference[k,legs]+.18*np.tanh(latent),lower[legs],upper[legs])
                    np.testing.assert_array_equal(adjusted,z['adjusted_double_pre_slew_target_rad'][k])
                    np.testing.assert_array_equal(adjusted[head],fixed[head])
                    if k:
                        previous=z['applied'][k-1]
                        planned=np.clip(np.clip(adjusted,lower,upper),previous-5.24*.02,previous+5.24*.02)
                        base=np.clip(np.clip(fixed,lower,upper),previous-5.24*.02,previous+5.24*.02)
                        np.testing.assert_array_equal(planned,z['planned_before_integration_rad'][k]);np.testing.assert_array_equal(base,z['same_state_unperturbed_planned_rad'][k])
                np.testing.assert_array_equal(z['planned_before_integration_rad'],z['applied'])
                assert abs(sum(z['original_step_reward'])-row['return_sum'])<1e-10
                oldpath=run.selector.OUTPUT/'candidate'/f"case_{row['case_seed']}";oldrow=read(oldpath/'result.json')
                assert row['initial_hash']==oldrow['initial_hash']
                if mode=='baseline':
                    baseline_paths+=1
                    with np.load(oldpath/'trajectory.npz',allow_pickle=False) as old:
                        for key in old.files:np.testing.assert_array_equal(archive[key] if key in ('qpos','qvel') else z[key],old[key])
                    for key in oldrow:
                        if key in row:assert row[key]==oldrow[key],key
            assert read(path/'rng.json')==rng.bit_generator.state
            rows.append(dict(group=folder.name+'/'+path.parent.name,case=row['case_seed'],mode=mode,controls=row['controls'],success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],original_peaks=row['peaks'],combined_max=row['maximum_combined_14_joint_correction_rad']))
    assert counts==162 and controls==365247 and baseline_paths==56
    run.compare_smoke(run.OUTPUT)
    baseline=result['baseline'];candidates=[]
    for h in result['history']:
        check=h['check'];old={r['case_seed']:r for r in baseline['rows']}
        good=bool(check['nominal_success'] and not check['physical_failures'] and check['successes']>baseline['successes'] and all(not old[r['case_seed']]['success'] or r['success'] for r in check['rows']))
        assert good==h['eligible']
        if good:candidates.append(h['iteration'])
    assert not candidates and result['selected_checkpoint'] is None and result['selected_R157_fallback']
    for case in run.CASES:
        a=run.OUTPUT/'candidate'/f'case_{case}';b=run.OUTPUT/'development_baseline'/f'case_{case}'
        with np.load(a/'trajectory.npz',allow_pickle=False) as za,np.load(b/'trajectory.npz',allow_pickle=False) as zb:
            assert za.files==zb.files
            for key in za.files:np.testing.assert_array_equal(za[key],zb[key])
    learner=independent_learning(result)
    assert before=={d.name:inventory(d) for d in DIRECTORIES}
    live=sum(f.stat().st_size for d in DIRECTORIES for f in d.rglob('*') if f.is_file())
    assert 3*live<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    return dict(terminal_result_saved=True,natural_exit=True,new_dynamic=0,closed_paths=162,controls=controls,baseline_paths_all_old_fields_bitwise=56,policy_mean_noise_latent_rng_independently_exact=True,post_double_boundary_replacement_independently_exact=True,planning_controls_after_first_independently_exact=True,first_control_original_R157_applied_parity=True,all_planned_equals_applied=True,learner=learner,original_before_request_IMU_double_reconstructed=False,whole_old_feedback_independently_recomputed=False,cross_implementation_bitwise_or_safety_claim=False,test_startup_mapping_manifest_not_saved=True,explicit_four_test_source_copies_verified=True,rows=rows,source_inventory=before,live_bytes=live,full_task_completed=False,hardware_readiness=False,qualification=False)


# Static target data only, no environment, MjData, forward or integration.
with np.load(run.local.prior.program.REFERENCE,allow_pickle=False) as _reference:reference=_reference['targets'].copy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6]);idle(a.base);proof=verify()
    if a.verify:
        assert not PROOF.exists();run.local.write_json(PROOF,proof);print(json.dumps({k:v for k,v in proof.items() if k not in ('source_inventory','rows')},indent=2));return
    assert proof==read(PROOF)
    before_git=sum(f.stat().st_size for f in (REPO/'.git').rglob('*') if f.is_file())
    for d in DIRECTORIES:
        target=REPO/'results'/d.name/'terminal_snapshot';assert not target.exists();shutil.copytree(d,target)
        assert inventory(target)==proof['source_inventory'][d.name]
        log=d.with_suffix('.log') if d in (run.SMOKE,run.OUTPUT) else ROOT/'tmp/getup_total_target_ppo_r196_launcher_20261010.log' if d==run.LAUNCH else None
        if log and log.exists():shutil.copy2(log,target/'process.log')
        run.local.write_json(target/'snapshot_metadata.json',dict(terminal_result_saved=True,natural_exit=True,new_dynamic=0,closed_paths=162,full_task_completed=False,hardware_readiness=False,qualification=False))
        run.local.write_json(target/'artifact_manifest.json',inventory(target))
        subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    for name in ('total_target_ppo_kernel_r196.py','train_getup_total_target_ppo_r196.py','test_getup_total_target_ppo_r196.py','launch_getup_total_target_ppo_r196.py',Path(__file__).name):
        target=REPO/'scripts'/name;assert not target.exists();shutil.copy2(Path(__file__).with_name(name),target)
        subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
    for name in ('GETUP_TOTAL_TARGET_PPO_PLAN_R196_20261010.md','GETUP_TOTAL_TARGET_PPO_TERMINAL_R196_20261010.md','publish_getup_total_target_ppo_terminal_r196_20261010.py'):
        target=REPO/('scripts/'+name if name.endswith('.py') else name);assert not target.exists();shutil.copy2(ROOT/'tmp'/name,target)
        subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
    target=REPO/'results'/run.OUTPUT.name/'terminal_verification.json';assert not target.exists();shutil.copy2(PROOF,target)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve finite R196 full-episode total-target PPO and all failed checkpoints'],cwd=REPO,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    after_git=sum(f.stat().st_size for f in (REPO/'.git').rglob('*') if f.is_file())
    archive=sum(f.stat().st_size for d in DIRECTORIES for f in (REPO/'results'/d.name).rglob('*') if f.is_file())
    assert proof['live_bytes']+archive+max(0,after_git-before_git)<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    print(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),live_bytes=proof['live_bytes'],archive_bytes=archive,git_increment_bytes=after_git-before_git,free_bytes=shutil.disk_usage(ROOT).free)),flush=True)


if __name__=='__main__':main()
