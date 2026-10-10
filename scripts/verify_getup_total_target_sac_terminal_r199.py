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
from diagnostics import train_getup_total_target_sac_r199 as run

ROOT=run.ROOT; REPO=ROOT/'github/openduck-mini-'
PREFLIGHT=ROOT/'outputs/getup_total_target_sac_r199_preflight_20261011'
PROOF=ROOT/'tmp/getup_total_target_sac_r199_terminal_verification_20261011.json'
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


def dist(w,x,xp=np):
    out=net(w,x,'actor',xp)
    return out[...,:10],xp.clip(float(np.log(1e-4))+out[...,10:],-12.,-4.)
def sample(mu,ls,noise,xp=np):
    z=mu+xp.exp(ls)*noise
    action=xp.tanh(z)
    jac=2*(np.log(2.)-z-xp.logaddexp(0.,-2*z))
    lp=xp.sum(-.5*(noise**2+np.log(2*np.pi))-ls-jac,axis=-1)
    return action,z,lp

def independent_learning(result):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jp
    import optax
    from flax import serialization
    w=load(run.OUTPUT/'initial.npz');initial_rng=np.random.default_rng(299)
    for name,n,last in [('actor',101,20),('q1',111,1),('q2',111,1)]:
        for i,(a,b) in enumerate([(n,32),(32,16),(16,last)]):
            weight=np.zeros((a,b)) if name=='actor' and i==2 else initial_rng.normal(0,1/np.sqrt(a),(a,b))
            np.testing.assert_array_equal(weight,w[f'{name}_w{i}']);np.testing.assert_array_equal(np.zeros(b),w[f'{name}_b{i}'])
    actor={k:jp.asarray(v) for k,v in w.items() if k.startswith('actor_')}
    critics={k:jp.asarray(v) for k,v in w.items() if not k.startswith('actor_')}
    tq=jax.tree.map(lambda v:v.copy(),critics);la=jp.asarray(np.log(1e-4))
    ao=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-6))
    qo=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-4))
    lo=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-4))
    ast=ao.init(actor);qst=qo.init(critics);lst=lo.init(la);count=0;rng=np.random.default_rng(299);proof=[];parts=[]
    def state_bytes():
        return serialization.to_bytes(dict(actor=actor,critics=critics,target_critics=tq,log_alpha=la,actor_state=ast,critic_state=qst,alpha_state=lst,count=count))
    assert state_bytes()==(run.OUTPUT/'checkpoint_0000.learner.msgpack').read_bytes()
    assert rng.bit_generator.state==read(run.OUTPUT/'checkpoint_0000_rng.json')
    def target_value(pa,pq,logalpha,nx,noise,r,boot,tail):
        mu,ls=dist(pa,nx,jp);a,_,lp=sample(mu,ls,noise,jp);inp=jp.concatenate([nx,a],axis=1)
        y=r+(.99**.2)*(boot*(jp.minimum(net(pq,inp,'q1',jp)[:,0],net(pq,inp,'q2',jp)[:,0])-jp.exp(logalpha)*lp)+tail)
        return jax.lax.stop_gradient(y)
    def qloss(q,x,a,y):
        inp=jp.concatenate([x,a],axis=1);e1=net(q,inp,'q1',jp)[:,0]-y;e2=net(q,inp,'q2',jp)[:,0]-y
        loss=.5*jp.mean(e1**2+e2**2)
        return loss,jp.array([loss,jp.mean(y),jp.max(jp.abs(y))])
    def aloss(p,q,logalpha,x,noise):
        mu,ls=dist(p,x,jp);a,_,lp=sample(mu,ls,noise,jp);inp=jp.concatenate([x,a],axis=1)
        value=jp.minimum(net(q,inp,'q1',jp)[:,0],net(q,inp,'q2',jp)[:,0])
        loss=jp.mean(jax.lax.stop_gradient(jp.exp(logalpha))*lp-value)
        return loss,jp.array([loss,jp.mean(lp),jp.mean(value)])
    def alphloss(logalpha,lp):
        return -jp.exp(logalpha)*jp.mean(jax.lax.stop_gradient(lp-75.))
    target=jax.jit(target_value);qgradient=jax.jit(jax.value_and_grad(qloss,has_aux=True))
    agradient=jax.jit(jax.value_and_grad(aloss,has_aux=True));lgradient=jax.jit(jax.value_and_grad(alphloss))
    for iteration in (1,2):
        path=run.OUTPUT/f'training_{iteration:04d}';episodes=[]
        before={k:np.asarray(v).copy() for k,v in {**actor,**critics}.items()};before_alpha=float(la)
        for case in run.CASES:
            with np.load(path/f'case_{case}/trajectory.npz',allow_pickle=False) as z:
                obs=z['observations'];n=len(obs);rewards=z['original_step_reward'];ret=np.empty(n);total=0.
                for k in range(n-1,-1,-1):total=rewards[k]+(.99**.2)*total;ret[k]=total
                ix=np.arange(1,min(n,529));x=np.stack([sensor_features(obs[k],obs[0],int(k)) for k in ix])
                expected=(np.arange(n)>0)&(np.arange(n)<529);np.testing.assert_array_equal(expected,z['policy_active'])
                mu,ls=dist(before,x);a,latent,lp=sample(mu,ls,z['policy_noise'][ix])
                for key,value in [('policy_location',mu),('policy_log_std',ls),('policy_mean',np.tanh(mu)),('policy_latent',latent),('policy_log_density',lp),('policy_requested_action',a)]:
                    np.testing.assert_allclose(value,z[key][ix],atol=1e-12,rtol=0)
                nx=np.stack([sensor_features(obs[k+1] if k<n-1 else obs[k],obs[0],int(k+1)) for k in ix])
                boot=np.asarray([float(k<n-1 and k+1<529) for k in ix]);tail=np.asarray([float(ret[k+1]) if k<n-1 and k+1==529 else 0. for k in ix])
                episodes.append(dict(features=x,action=z['policy_requested_action'][ix],reward=rewards[ix],next_features=nx,bootstrap=boot,home_tail=tail,terminal=ix==n-1,control=ix))
        new={key:np.concatenate([e[key] for e in episodes]) for key in episodes[0]};saved=load(path/'new_episode_replay.npz');assert saved.keys()==new.keys()
        for key in new:np.testing.assert_array_equal(new[key],saved[key])
        parts.append(new);replay={key:np.concatenate([p[key] for p in parts]) for key in new}
        assert read(path/'replay_manifest.json')==dict(training_rounds=list(range(1,iteration+1)),transitions=len(replay['reward']),old_data_used=False)
        stats=[];norms=[];indices=[];next_noise=[];actor_noise=[]
        for _ in range(64):
            ix=rng.integers(0,len(replay['reward']),size=512);nn=rng.normal(size=(512,10));pn=rng.normal(size=(512,10))
            indices.append(ix);next_noise.append(nn);actor_noise.append(pn)
            x,a,r,nx,boot,tail=[jp.asarray(replay[k][ix]) for k in ('features','action','reward','next_features','bootstrap','home_tail')]
            y=target(actor,tq,la,nx,jp.asarray(nn),r,boot,tail)
            (_,qs),qg=qgradient(critics,x,a,y);update,qst=qo.update(qg,qst,critics);critics=optax.apply_updates(critics,update)
            (_,ps),pg=agradient(actor,critics,la,x,jp.asarray(pn));update,ast=ao.update(pg,ast,actor);actor=optax.apply_updates(actor,update)
            mu,ls=dist(actor,x,jp);_,_,lp=sample(mu,ls,jp.asarray(pn),jp)
            aloss_value,ag=lgradient(la,lp);update,lst=lo.update(ag,lst,la);la=optax.apply_updates(la,update)
            tq=jax.tree.map(lambda t,v:(1-.005)*t+.005*v,tq,critics);count+=1
            stats.append(np.r_[np.asarray(qs),np.asarray(ps),float(aloss_value),float(la),float(jp.exp(la)),count])
            norms.append([float(optax.global_norm(qg)),float(optax.global_norm(pg)),float(np.abs(ag))])
        saved=load(path/'learner_update.npz')
        for key,value in [('statistics',stats),('gradient_norms',norms),('replay_indices',indices),('next_policy_noise',next_noise),('actor_policy_noise',actor_noise),('last_alpha_gradient',ag)]:np.testing.assert_array_equal(value,saved[key])
        for key,value in {**qg,**pg}.items():np.testing.assert_array_equal(value,saved['last_gradient_'+key])
        checkpoint=load(run.OUTPUT/f'checkpoint_{iteration:04d}.npz')
        for key,value in {**actor,**critics}.items():np.testing.assert_array_equal(value,checkpoint[key])
        assert state_bytes()==(run.OUTPUT/f'checkpoint_{iteration:04d}.learner.msgpack').read_bytes()
        assert rng.bit_generator.state==read(run.OUTPUT/f'checkpoint_{iteration:04d}_rng.json')
        changes={key:float(np.linalg.norm(np.asarray(value)-before[key])) for key,value in {**actor,**critics}.items()};h=result['history'][iteration-1]
        assert changes==h['changes'] and h['critic_updates']==h['actor_updates']==h['alpha_updates']==64
        assert float(la)-before_alpha==h['log_alpha_change']
        proof.append(dict(iteration=iteration,new_replay_transitions=len(new['reward']),pooled_replay_transitions=len(replay['reward']),critic_updates=64,actor_updates=64,alpha_updates=64,replay_and_home_terminal_backup_exact=True,indices_two_noises_statistics_norms_last_gradients_exact=True,online_weights_exact=True,target_and_three_Adam_and_alpha_serialization_bytes_exact=True,rng_exact=True,changes=changes,log_alpha_change=float(la)-before_alpha))
    return proof


def verify():
    before={d.name:inventory(d) for d in DIRECTORIES}
    print('R199_VERIFY_BEGIN',flush=True)
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
    preflight=read(ROOT/'tmp/getup_total_target_sac_r199_preflight_exit_20261011.json');assert preflight['observed_exit_code']==0 and preflight['new_dynamic']==0 and preflight['source_hashes_unchanged']
    for folder in (PREFLIGHT,run.LAUNCH/'initial',run.LAUNCH/'formal'):
        log=(folder/'regression.log').read_text();assert 'Ran 26 tests' in log and '\nOK\n' in log
        sources(read(folder/'source_manifest.json'),folder/'executed_sources')
        for name in ('total_target_sac_kernel_r199.py','train_getup_total_target_sac_r199.py','test_getup_total_target_sac_r199.py','launch_getup_total_target_sac_r199.py'):
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
                assert z['construction_preparation_sensors'].shape==z['preparation_sensors'].shape==(40,50)
                np.testing.assert_array_equal(obs[1:,34:48],(z['applied'][:-1]-model.keyframe('home').qpos[model.jnt_qposadr[joints]]).astype(np.float32))
                gains,choice,logits=run.selector.old.predict(selector_model,z['preparation_sensors'][-1])
                assert gains.tolist()==row['base_gains'] and choice==row['base_choice'] and logits.tolist()==row['initial_sensor_logits']
                for k in range(row['controls']):
                    right=run.local.local_feedback(obs[k],nominal[k],np.array([9,10,11]),gains) if k<529 else np.zeros(3)
                    np.testing.assert_array_equal(right,z['local_hip_extra_rad'][k])
                    active=mode!='baseline' and 0<k<529
                    assert bool(z['policy_active'][k])==active
                    mean=np.zeros(10);noise=np.zeros(10);requested=np.zeros(10);mu=np.zeros(10);ls=np.zeros(10);latent=np.zeros(10);lp=0.
                    if active:
                        mu,ls=dist(w,sensor_features(obs[k],initial,k));mean=np.tanh(mu)
                        noise=rng.normal(size=10) if mode=='sample' else np.zeros(10)
                        requested,latent,lp=sample(mu,ls,noise)
                    for key,val in [('policy_mean',mean),('policy_noise',noise),('policy_requested_action',requested),('policy_location',mu),('policy_log_std',ls),('policy_latent',latent),('policy_log_density',lp)]:np.testing.assert_array_equal(val,z[key][k])
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
                assert float(np.abs(z['adjusted_double_pre_slew_target_rad']-reference[:len(obs)]).max())==row['maximum_combined_14_joint_correction_rad']
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
    for folder in (run.SMOKE,run.OUTPUT):
        for name,mode,cases,parity in run.SMOKE_SPEC:
            p=folder/name;saved=read(p/'results.json');actual=[read(p/f'case_{c}'/'result.json') for c in cases]
            assert saved['rows']==actual
            assert saved['successes']==sum(bool(r['success']) for r in actual if r['case_seed'] is not None)
            assert saved['nominal_success']==next((bool(r['success']) for r in actual if r['case_seed'] is None),None)
            assert saved['physical_failures']==sum(not r['valid'] for r in actual)
            assert saved['return_sum']==sum(r['return_sum'] for r in actual)
    report_groups=[('development_baseline',result['baseline']),('candidate',result['candidate'])]
    for h in result['history']:report_groups.extend([(f"training_{h['iteration']:04d}",h['training']),(f"check_{h['iteration']:04d}",h['check'])])
    for name,saved in report_groups:
        actual=[read(run.OUTPUT/name/f'case_{c}'/'result.json') for c in run.CASES]
        assert saved==read(run.OUTPUT/name/'results.json') and saved['rows']==actual
        assert saved['successes']==sum(bool(r['success']) for r in actual if r['case_seed'] is not None)
        assert saved['nominal_success']==next(bool(r['success']) for r in actual if r['case_seed'] is None)
        assert saved['physical_failures']==sum(not r['valid'] for r in actual)
        assert saved['return_sum']==sum(r['return_sum'] for r in actual)
    assert result['history']==read(run.OUTPUT/'training_closed_history.json')
    baseline=result['baseline'];candidates=[]
    for h in result['history']:
        check=h['check'];old={r['case_seed']:r for r in baseline['rows']}
        good=bool(check['nominal_success'] and not check['physical_failures'] and check['successes']>baseline['successes'] and all(not old[r['case_seed']]['success'] or r['success'] for r in check['rows']))
        assert good==h['eligible']
        if good:candidates.append((check['successes'],min(r['return_sum'] for r in check['rows']),sum(r['return_sum'] for r in check['rows']),h['iteration']))
    selected=max(candidates,key=lambda c:(c[0],c[1],c[2],-c[3]))[3] if candidates else None
    assert result['selected_checkpoint']==selected and result['selected_R157_fallback']==(selected is None)
    if selected is None:
        for case in run.CASES:
            a=run.OUTPUT/'candidate'/f'case_{case}';b=run.OUTPUT/'development_baseline'/f'case_{case}'
            with np.load(a/'trajectory.npz',allow_pickle=False) as za,np.load(b/'trajectory.npz',allow_pickle=False) as zb:
                assert za.files==zb.files
                for key in za.files:np.testing.assert_array_equal(za[key],zb[key])
    candidate=result['candidate']
    assert result['original_development_gate']==bool(candidate['nominal_success'] and not candidate['physical_failures'] and candidate['successes']>=22 and candidate['successes']>18)
    print('R199_SCALAR_COMPLETE',counts,controls,flush=True)
    learner=independent_learning(result)
    print('R199_LEARNER_COMPLETE',flush=True)
    assert before=={d.name:inventory(d) for d in DIRECTORIES}
    live=sum(f.stat().st_size for d in DIRECTORIES for f in d.rglob('*') if f.is_file())
    assert 3*live<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    return dict(verifier_source_sha256=digest(__file__),terminal_result_saved=True,natural_exit=True,new_dynamic=0,closed_paths=162,controls=controls,baseline_paths_all_old_fields_bitwise=baseline_paths,policy_mu_logstd_mean_noise_latent_density_action_rng_independently_exact=True,post_double_boundary_replacement_independently_exact=True,planning_controls_after_first_independently_exact=True,first_control_original_R157_applied_parity=True,all_planned_equals_applied=True,learner=learner,original_before_request_IMU_double_reconstructed=False,whole_old_feedback_independently_recomputed=False,cross_implementation_bitwise_or_safety_claim=False,test_startup_mapping_manifest_verified=True,explicit_four_test_source_copies_verified=True,rows=rows,source_inventory=before,live_bytes=live,full_task_completed=False,hardware_readiness=False,qualification=False)


# Static target data only, no environment, MjData, forward or integration.
with np.load(run.local.prior.program.REFERENCE,allow_pickle=False) as _reference:reference=_reference['targets'].copy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6]);idle(a.base)
    assert not PROOF.exists()
    proof=verify()
    with PROOF.open('x') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in proof.items() if k not in ('source_inventory','rows')},indent=2),flush=True)
if __name__=='__main__':main()

