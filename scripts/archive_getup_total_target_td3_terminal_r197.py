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
from diagnostics import train_getup_total_target_td3_r197 as run

ROOT=run.ROOT; REPO=ROOT/'github/openduck-mini-'
PREFLIGHT=ROOT/'outputs/getup_total_target_td3_r197_preflight_20261010'
PROOF=ROOT/'tmp/getup_total_target_td3_r197_terminal_verification_20261010.json'
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
        if i<2 or name=='actor':x=xp.tanh(x)
    return x


def sensor_features(obs,initial,k):
    return np.r_[np.tanh(obs[:50].astype(float)),np.tanh(initial[:50].astype(float)),min(k/528.,1.)]


def independent_learning(result):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jp
    import optax
    from flax import serialization
    w=load(run.OUTPUT/'initial.npz')
    actor={k:jp.asarray(v) for k,v in w.items() if k.startswith('actor_')}
    critics={k:jp.asarray(v) for k,v in w.items() if not k.startswith('actor_')}
    ta=jax.tree.map(lambda v:v.copy(),actor);tq=jax.tree.map(lambda v:v.copy(),critics)
    ao=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-6))
    qo=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-4))
    ast=ao.init(actor);qst=qo.init(critics);count=0;rng=np.random.default_rng(297);proof=[];parts=[]
    def state_bytes():
        return serialization.to_bytes(dict(actor=actor,critics=critics,target_actor=ta,target_critics=tq,actor_state=ast,critic_state=qst,count=count))
    assert state_bytes()==(run.OUTPUT/'checkpoint_0000.learner.msgpack').read_bytes()
    assert rng.bit_generator.state==read(run.OUTPUT/'checkpoint_0000_rng.json')
    def target_value(pa,pq,nx,noise,r,boot,tail):
        na=jp.clip(net(pa,nx,'actor',jp)+jp.clip(1e-4*noise,-2e-4,2e-4),-1.,1.)
        qa=jp.concatenate([nx,na],axis=1)
        y=r+(.99**.2)*(boot*jp.minimum(net(pq,qa,'q1',jp)[:,0],net(pq,qa,'q2',jp)[:,0])+tail)
        return jax.lax.stop_gradient(y)
    def qloss(q,x,a,y):
        inp=jp.concatenate([x,a],axis=1)
        e1=net(q,inp,'q1',jp)[:,0]-y;e2=net(q,inp,'q2',jp)[:,0]-y
        loss=jp.mean(e1**2+e2**2)
        return loss,jp.array([loss,jp.mean(y),jp.max(jp.abs(y))])
    def aloss(p,q,x):
        a=net(p,x,'actor',jp)
        return -jp.mean(net(q,jp.concatenate([x,a],axis=1),'q1',jp)[:,0])
    target=jax.jit(target_value);qgradient=jax.jit(jax.value_and_grad(qloss,has_aux=True));agradient=jax.jit(jax.value_and_grad(aloss))
    for iteration in (1,2):
        path=run.OUTPUT/f'training_{iteration:04d}';episodes=[]
        before={k:np.asarray(v).copy() for k,v in {**actor,**critics}.items()}
        for case in run.CASES:
            with np.load(path/f'case_{case}/trajectory.npz',allow_pickle=False) as z:
                obs=z['observations'];n=len(obs);rewards=z['original_step_reward'];ret=np.empty(n);total=0.
                for k in range(n-1,-1,-1):total=rewards[k]+(.99**.2)*total;ret[k]=total
                ix=np.arange(1,min(n,529));x=np.stack([sensor_features(obs[k],obs[0],int(k)) for k in ix])
                expected=(np.arange(n)>0)&(np.arange(n)<529);np.testing.assert_array_equal(expected,z['policy_active'])
                mean=net(before,x,'actor');np.testing.assert_allclose(mean,z['policy_mean'][ix],atol=1e-12,rtol=0)
                nx=np.stack([sensor_features(obs[k+1] if k<n-1 else obs[k],obs[0],int(k+1)) for k in ix])
                boot=np.asarray([float(k<n-1 and k+1<529) for k in ix])
                tail=np.asarray([float(ret[k+1]) if k<n-1 and k+1==529 else 0. for k in ix])
                episodes.append(dict(features=x,action=z['policy_requested_action'][ix],reward=rewards[ix],next_features=nx,bootstrap=boot,home_tail=tail,terminal=ix==n-1,control=ix))
        new={key:np.concatenate([e[key] for e in episodes]) for key in episodes[0]}
        saved=load(path/'new_episode_replay.npz');assert saved.keys()==new.keys()
        for key in new:np.testing.assert_array_equal(new[key],saved[key])
        parts.append(new);replay={key:np.concatenate([p[key] for p in parts]) for key in new}
        assert read(path/'replay_manifest.json')==dict(training_rounds=list(range(1,iteration+1)),transitions=len(replay['reward']),old_data_used=False)
        stats=[];norms=[];indices=[];noises=[];lastq=lasta=None;acount=0
        for _ in range(64):
            ix=rng.integers(0,len(replay['reward']),size=512);noise=rng.normal(size=(512,10))
            indices.append(ix);noises.append(noise)
            x,a,r,nx,boot,tail=[jp.asarray(replay[k][ix]) for k in ('features','action','reward','next_features','bootstrap','home_tail')]
            y=target(ta,tq,nx,jp.asarray(noise),r,boot,tail)
            (_,stat),g=qgradient(critics,x,a,y);qnorm=float(optax.global_norm(g));lastq=g
            update,qst=qo.update(g,qst,critics);critics=optax.apply_updates(critics,update);count+=1
            loss=0.;anorm=0.;updated=count%2==0
            if updated:
                value,g=agradient(actor,critics,x);loss=float(value);anorm=float(optax.global_norm(g));lasta=g
                update,ast=ao.update(g,ast,actor);actor=optax.apply_updates(actor,update);acount+=1
                ta=jax.tree.map(lambda t,v:(1-.005)*t+.005*v,ta,actor)
                tq=jax.tree.map(lambda t,v:(1-.005)*t+.005*v,tq,critics)
            stats.append(np.r_[np.asarray(stat),loss,float(updated),count]);norms.append([qnorm,anorm])
        saved=load(path/'learner_update.npz')
        for key,value in [('statistics',stats),('gradient_norms',norms),('replay_indices',indices),('target_noise',noises)]:np.testing.assert_array_equal(value,saved[key])
        for key,value in {**lastq,**lasta}.items():np.testing.assert_array_equal(value,saved['last_gradient_'+key])
        checkpoint=load(run.OUTPUT/f'checkpoint_{iteration:04d}.npz')
        for key,value in {**actor,**critics}.items():np.testing.assert_array_equal(value,checkpoint[key])
        assert state_bytes()==(run.OUTPUT/f'checkpoint_{iteration:04d}.learner.msgpack').read_bytes()
        assert rng.bit_generator.state==read(run.OUTPUT/f'checkpoint_{iteration:04d}_rng.json')
        changes={key:float(np.linalg.norm(np.asarray(value)-before[key])) for key,value in {**actor,**critics}.items()}
        h=result['history'][iteration-1]
        assert changes==h['changes'] and h['critic_updates']==64 and h['actor_updates']==acount==32
        proof.append(dict(iteration=iteration,new_replay_transitions=len(new['reward']),pooled_replay_transitions=len(replay['reward']),critic_updates=64,actor_updates=32,replay_and_home_terminal_backup_exact=True,indices_target_noise_statistics_norms_last_gradients_exact=True,online_weights_exact=True,target_and_two_Adam_serialization_bytes_exact=True,rng_exact=True,changes=changes))
    return proof


def verify():
    before={d.name:inventory(d) for d in DIRECTORIES}
    launch=read(run.LAUNCH/'result.json');result=read(run.OUTPUT/'results.json')
    assert launch['natural_exit'] and launch['smoke_exit_code']==launch['formal_exit_code']==0 and not launch['formal_not_run']
    assert result['terminal_result_saved'] and result['actual_dynamic_attempts']==156 
    for folder in (run.LAUNCH,run.SMOKE,run.OUTPUT):
        c=read(folder/'contract.json');sources(c['sources'],folder/'executed_sources')
        if folder!=run.LAUNCH:
            assert c['model_files']==run.utility.model_files(run.local.prior.program.SCENE)
            for p,sha in c['hashes'].items():assert digest(p)==sha
            assert read(folder/'input_hashes_after.json')['source_and_model_files_unchanged']
            for f in folder.glob('worker_source_manifest_*.json'):
                dest=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''));sources(read(f),dest)
    assert read(ROOT/'tmp/getup_total_target_td3_r197_preflight_exit_20261010.json')==dict(local_session=44373,observed_natural_exit_code=0,tests=22,seconds=8.209,new_dynamic=0)
    for folder in (PREFLIGHT,run.LAUNCH/'initial',run.LAUNCH/'formal'):
        log=(folder/'regression.log').read_text();assert 'Ran 22 tests' in log and '\nOK\n' in log
        sources(read(folder/'source_manifest.json'),folder/'executed_sources')
        for name in ('total_target_td3_kernel_r197.py','train_getup_total_target_td3_r197.py','test_getup_total_target_td3_r197.py','launch_getup_total_target_td3_r197.py'):
            candidates=list((folder/'executed_sources').glob('*_'+name));assert len(candidates)==1 and digest(candidates[0])==digest(Path(run.__file__).with_name(name))
    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE))
    assert run.utility.model_digest(model)=='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
    legs=np.array([model.actuator(s+'_'+part).id for s in ('left','right') for part in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
    head=np.array([i for i in range(14) if i not in legs])
    joints=model.actuator_trnid[:,0];lower,upper=model.jnt_range[joints].T.copy()
    # StrictSim lower/upper are joint ranges mapped in actuator order.
    counts=controls=baseline_paths=0;rows=[];weight_cache={}
    selector_model=load(run.SMOKE/'frozen/initial_selector.npz')
    with np.load(run.SMOKE/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    for folder in (run.SMOKE,run.OUTPUT):
        for path in sorted(folder.glob('*/case_*')):
            row=read(path/'result.json');mode=row['policy_mode'];counts+=1;controls+=row['controls']
            assert row['physics_unchanged'] and row['compiled_model_unchanged']
            assert row['physics_sha256']=='4b9f4a9167e1614c22f7e6f28dd8508f7ccabb61d909a6c47dd46e40b861c6ae'
            assert row['compiled_model_sha256']=='53a24e16553a85c7e1ba354a692eca45094ba31d5d561f2d54f7a75e1d5914b3'
            if path.parent.name=='training_0002':weightfile=run.OUTPUT/'checkpoint_0001.npz'
            elif path.parent.name in ('check_0001','check_0002'):weightfile=run.OUTPUT/f'checkpoint_{path.parent.name[-4:]}.npz'
            elif path.parent.name=='candidate' and result['selected_checkpoint'] is not None:weightfile=run.OUTPUT/f"checkpoint_{result['selected_checkpoint']:04d}.npz"
            else:weightfile=folder/'initial.npz'
            assert digest(weightfile)==row['policy_model_sha256']
            if str(weightfile) not in weight_cache:weight_cache[str(weightfile)]=load(weightfile)
            w=weight_cache[str(weightfile)];rng=np.random.default_rng(row['sampling_seed'])
            with np.load(path/'trajectory.npz',allow_pickle=False) as archive:
                # Root data is not decoded for decisions; only separate equality below.
                z={k:archive[k] for k in archive.files if k not in ('qpos','qvel')}
                obs=z['observations'];initial=obs[0]
                gains,choice,logits=run.selector.old.predict(selector_model,z['preparation_sensors'][-1])
                assert gains.tolist()==row['base_gains'] and choice==row['base_choice'] and logits.tolist()==row['initial_sensor_logits']
                for k in range(row['controls']):
                    right=run.local.local_feedback(obs[k],nominal[k],np.array([9,10,11]),gains) if k<529 else np.zeros(3)
                    np.testing.assert_array_equal(right,z['local_hip_extra_rad'][k])
                    active=mode!='baseline' and 0<k<529
                    assert bool(z['policy_active'][k])==active
                    mean=np.zeros(10);noise=np.zeros(10);requested=np.zeros(10)
                    if active:
                        mean=net(w,sensor_features(obs[k],initial,k),'actor')
                        noise=rng.normal(size=10) if mode=='sample' else np.zeros(10)
                        requested=np.clip(mean+1e-4*noise,-1.,1.)
                    for key,val in [('policy_mean',mean),('policy_noise',noise),('policy_requested_action',requested)]:np.testing.assert_array_equal(val,z[key][k])
                    fixed=z['original_double_pre_slew_target_rad'][k];adjusted=fixed.copy()
                    # Reference file contains its original 2279 target sequence.
                    if active:
                        adjusted[legs]=np.clip(reference[k,legs]+.18*requested,lower[legs],upper[legs])
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
                with np.load(oldpath/'trajectory.npz',allow_pickle=False) as old:np.testing.assert_array_equal(z['applied'][0],old['applied'][0])
                if mode=='baseline':
                    baseline_paths+=1
                    with np.load(oldpath/'trajectory.npz',allow_pickle=False) as old:
                        for key in old.files:np.testing.assert_array_equal(archive[key] if key in ('qpos','qvel') else z[key],old[key])
                    for key in oldrow:
                        if key in row:assert row[key]==oldrow[key],key
            assert read(path/'rng.json')==rng.bit_generator.state
            rows.append(dict(group=folder.name+'/'+path.parent.name,case=row['case_seed'],mode=mode,controls=row['controls'],success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],original_peaks=row['peaks'],combined_max=row['maximum_combined_14_joint_correction_rad']))
    assert counts==162 and baseline_paths==(56 if result['selected_checkpoint'] is None else 31)
    run.compare_smoke(run.OUTPUT)
    baseline=result['baseline'];candidates=[]
    for h in result['history']:
        check=h['check'];old={r['case_seed']:r for r in baseline['rows']}
        good=bool(check['nominal_success'] and not check['physical_failures'] and check['successes']>baseline['successes'] and all(not old[r['case_seed']]['success'] or r['success'] for r in check['rows']))
        assert good==h['eligible']
        if good:candidates.append((check['successes'],min(r['return_sum'] for r in check['rows']),sum(r['return_sum'] for r in check['rows']),h['iteration']))
    selected=max(candidates)[3] if candidates else None
    assert result['selected_checkpoint']==selected and result['selected_R157_fallback']==(selected is None)
    if selected is None:
        for case in run.CASES:
            a=run.OUTPUT/'candidate'/f'case_{case}';b=run.OUTPUT/'development_baseline'/f'case_{case}'
            with np.load(a/'trajectory.npz',allow_pickle=False) as za,np.load(b/'trajectory.npz',allow_pickle=False) as zb:
                assert za.files==zb.files
                for key in za.files:np.testing.assert_array_equal(za[key],zb[key])
    candidate=result['candidate']
    assert result['original_development_gate']==bool(candidate['nominal_success'] and not candidate['physical_failures'] and candidate['successes']>=22 and candidate['successes']>18)
    learner=independent_learning(result)
    assert before=={d.name:inventory(d) for d in DIRECTORIES}
    live=sum(f.stat().st_size for d in DIRECTORIES for f in d.rglob('*') if f.is_file())
    assert 3*live<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    return dict(terminal_result_saved=True,natural_exit=True,new_dynamic=0,closed_paths=162,controls=controls,baseline_paths_all_old_fields_bitwise=baseline_paths,policy_mean_noise_clipped_action_rng_independently_exact=True,post_double_boundary_replacement_independently_exact=True,planning_controls_after_first_independently_exact=True,first_control_original_R157_applied_parity=True,all_planned_equals_applied=True,learner=learner,original_before_request_IMU_double_reconstructed=False,whole_old_feedback_independently_recomputed=False,cross_implementation_bitwise_or_safety_claim=False,test_startup_mapping_manifest_verified=True,explicit_four_test_source_copies_verified=True,rows=rows,source_inventory=before,live_bytes=live,full_task_completed=False,hardware_readiness=False,qualification=False)


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
        log=d.with_suffix('.log') if d in (run.SMOKE,run.OUTPUT) else ROOT/'tmp/getup_total_target_td3_r197_launcher_20261010.log' if d==run.LAUNCH else None
        if log and log.exists():shutil.copy2(log,target/'process.log')
        run.local.write_json(target/'snapshot_metadata.json',dict(terminal_result_saved=True,natural_exit=True,new_dynamic=0,closed_paths=162,full_task_completed=False,hardware_readiness=False,qualification=False))
        run.local.write_json(target/'artifact_manifest.json',inventory(target))
        subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    for name in ('total_target_td3_kernel_r197.py','train_getup_total_target_td3_r197.py','test_getup_total_target_td3_r197.py','launch_getup_total_target_td3_r197.py',Path(__file__).name):
        target=REPO/'scripts'/name;assert not target.exists();shutil.copy2(Path(__file__).with_name(name),target)
        subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
    for name in ('GETUP_TOTAL_TARGET_TD3_PLAN_R197_20261010.md','GETUP_TOTAL_TARGET_TD3_TERMINAL_R197_20261010.md','publish_getup_total_target_td3_terminal_r197_20261010.py'):
        target=REPO/('scripts/'+name if name.endswith('.py') else name);assert not target.exists();shutil.copy2(ROOT/'tmp'/name,target)
        subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
    receipt=ROOT/'tmp/getup_total_target_td3_r197_preflight_exit_20261010.json'
    target=REPO/'results'/PREFLIGHT.name/'preflight_exit_receipt.json';assert not target.exists();shutil.copy2(receipt,target)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    target=REPO/'results'/run.OUTPUT.name/'terminal_verification.json';assert not target.exists();shutil.copy2(PROOF,target)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve finite R197 twin-Q replay total-target training and all failed checkpoints'],cwd=REPO,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    after_git=sum(f.stat().st_size for f in (REPO/'.git').rglob('*') if f.is_file())
    archive=sum(f.stat().st_size for d in DIRECTORIES for f in (REPO/'results'/d.name).rglob('*') if f.is_file())
    assert proof['live_bytes']+archive+max(0,after_git-before_git)<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    print(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),live_bytes=proof['live_bytes'],archive_bytes=archive,git_increment_bytes=after_git-before_git,free_bytes=shutil.disk_usage(ROOT).free)),flush=True)


if __name__=='__main__':main()
