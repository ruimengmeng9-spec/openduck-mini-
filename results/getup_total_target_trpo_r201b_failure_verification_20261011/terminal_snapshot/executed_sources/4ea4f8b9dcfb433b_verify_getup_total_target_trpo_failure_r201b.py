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
from diagnostics import train_getup_total_target_trpo_r201 as run

ROOT=Path('/data/shijinsheng/open_duck')
REPO=ROOT/'github/openduck-mini-'
BASE='9f3e32e022cd48ad4dfe4f4302e26cc3e359d9db'
DEST=ROOT/'outputs/getup_total_target_trpo_r201b_failure_verification_20261011'
PROOF=ROOT/'tmp/getup_total_target_trpo_r201b_failure_verification_20261011.json'
DIRECTORIES=(run.NUMERIC,run.PREFLIGHT,run.LAUNCH,run.SMOKE,run.OUTPUT,run.NUMERIC_SOURCE)
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
    rng=np.random.default_rng(301);w={}
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
    saved=load(path/'complete_original_MC_batch.npz');assert saved.keys()==combined.keys()
    for k,v in combined.items():eq(v,saved[k],k)
    assert read(path/'advantage_normalization.json')==dict(active_mean=mean,active_std=std,denominator=scale,active_count=int(active.sum()),old_data_used=False,original_home_tail_included=True,invalid_original_terminal=True)
    return combined

def learning():
    w=fresh()
    for path in (run.SMOKE/'initial.npz',run.OUTPUT/'initial.npz',run.OUTPUT/'checkpoint_0000.npz'):
        saved=load(path);assert saved.keys()==w.keys()
        for k,v in w.items():eq(v,saved[k],k)
    actor={k:jp.asarray(v) for k,v in w.items() if k.startswith('actor_')}
    value={k:jp.asarray(v) for k,v in w.items() if k.startswith('value_')}
    theta,unravel=ravel_pytree(actor);optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-4));state=optimizer.init(value)
    rng=np.random.default_rng(301);actor_count=value_count=0
    def state_bytes():return serialization.to_bytes(dict(actor=actor,value=value,state=state,actor_updates=actor_count,value_updates=value_count))
    assert state_bytes()==(run.OUTPUT/'checkpoint_0000.learner.msgpack').read_bytes()
    assert rng.bit_generator.state==read(run.OUTPUT/'checkpoint_0000_rng.json')
    batch=mc_batch(1,w);mask=batch['active'];x,z,adv=map(jp.asarray,(batch['features'][mask],batch['latents'][mask],batch['normalized_advantage'][mask]))
    assert rng.bit_generator.state==read(run.OUTPUT/'training_0001/learner_rng_before.json')
    # Independently written formulas, not kernel Learner/CG/backtrack/update functions.
    flat_mean=jax.jit(lambda t,x:net(unravel(t),x,'actor',jp))
    def surrogate(t,x,z,oldlog,a):return jp.mean(jp.exp(density(z,flat_mean(t,x),jp)-oldlog)*a)
    surr=jax.jit(surrogate);grad=jax.jit(jax.grad(surrogate))
    def kl(t,x,old):return jp.mean(jp.sum(((old-flat_mean(t,x))/1e-4)**2,axis=-1)/2)
    kfun=jax.jit(kl);kg=jax.grad(kl)
    fisher=jax.jit(lambda t,v,x,old:jax.jvp(lambda q:kg(q,x,old),(t,),(v,))[1])
    oldtheta=np.asarray(theta).copy();oldmean=flat_mean(theta,x);oldlog=density(z,oldmean,jp);g=np.asarray(grad(theta,x,z,oldlog,adv))
    def product(v):return np.asarray(fisher(jp.asarray(oldtheta),jp.asarray(v),x,oldmean))+.01*v
    d=np.zeros_like(g);res=g.copy();search=g.copy();rr=float(res@res);trace=[]
    for iteration in range(10):
        if rr<=1e-20:break
        prod=product(search);curv=float(search@prod);assert np.isfinite(prod).all() and curv>0
        alpha=rr/curv;previous=res.copy();d=d+alpha*search;res=res-alpha*prod;next_rr=float(res@res);beta=next_rr/rr
        trace.append(dict(iteration=iteration,search=search.copy(),product=prod.copy(),previous_residual=previous,residual=res.copy(),direction=d.copy(),rr=rr,curvature=curv,alpha=alpha,next_rr=next_rr,beta=beta))
        search=res+beta*search;rr=next_rr
    damped=product(d);curv=float(d@damped);assert np.isfinite(curv) and curv>=0
    step=d*np.sqrt(2*.001/curv) if curv>0 else np.zeros_like(d)
    baseline=float(surr(theta,x,z,oldlog,adv));baseline_kl=float(kfun(theta,x,oldmean));assert np.isfinite(baseline) and baseline_kl==0
    prediction=float(g@step);chosen=oldtheta.copy();accepted=None;trials=[]
    if prediction>0 and np.isfinite(prediction):
        for j in range(10):
            fraction=.5**j;candidate=oldtheta+fraction*step;t=jp.asarray(candidate)
            actual_surrogate=float(surr(t,x,z,oldlog,adv));actual_kl=float(kfun(t,x,oldmean));gain=actual_surrogate-baseline
            good=bool(np.isfinite(actual_surrogate) and np.isfinite(actual_kl) and 0<=actual_kl<=.001 and gain>0 and gain>=.1*fraction*prediction)
            trials.append(dict(index=j,fraction=fraction,parameters=candidate.copy(),surrogate=actual_surrogate,mean_kl=actual_kl,actual_gain=gain,predicted_gain=fraction*prediction,accepted=good))
            if good:accepted=j;chosen=candidate;break
    path=run.OUTPUT/'training_0001';saved=load(path/'actor_full_decisions.npz')
    actual=dict(old_theta=oldtheta,old_mean=np.asarray(oldmean),old_log_probability=np.asarray(oldlog),gradient=g,damped_product=damped,full_step=step,chosen_theta=chosen)
    for key in ('search','product','previous_residual','residual','direction'):actual['cg_'+key]=np.stack([t[key] for t in trace]) if trace else np.empty((0,len(g)))
    actual['cg_scalars']=np.asarray([[t[k] for k in ('rr','curvature','alpha','next_rr','beta')] for t in trace]).reshape(-1,5)
    actual['trial_parameters']=np.stack([t['parameters'] for t in trials]) if trials else np.empty((0,len(g)))
    actual['trial_scalars']=np.asarray([[t[k] for k in ('fraction','surrogate','mean_kl','actual_gain','predicted_gain','accepted')] for t in trials]).reshape(-1,6)
    assert actual.keys()==saved.keys()
    for k,v in actual.items():eq(v,saved[k],k)
    theta=jp.asarray(chosen);actor=unravel(theta);actor_count=1
    actor_changes={k:float(np.linalg.norm(np.asarray(actor[k])-w[k])) for k in actor}
    actor_json=dict(actor_update_count=1,damped_curvature=curv,actor_changes=actor_changes,cg_iterations=len(trace),cg_scalars=[{n:t[n] for n in ('iteration','rr','curvature','alpha','next_rr','beta')} for t in trace],line_search=dict(baseline_surrogate=baseline,baseline_mean_kl=baseline_kl,predicted_gain=prediction,accepted_index=accepted),trials=[{k:v for k,v in t.items() if k!='parameters'} for t in trials],vector_evidence='actor_full_decisions.npz',sample_mean_KL_not_safety=True)
    assert actor_json==read(path/'actor_update.json')
    def value_loss(p,x,ret):return jp.mean((net(p,x,'value',jp)[:,0]-ret)**2)
    vg=jax.jit(jax.value_and_grad(value_loss));ixs=[];losses=[];norms=[]
    for _ in range(64):
        ix=rng.integers(0,len(batch['returns']),size=512);loss,gradient=vg(value,jp.asarray(batch['features'][ix]),jp.asarray(batch['returns'][ix]))
        ixs.append(ix);losses.append(float(loss));norms.append(float(optax.global_norm(gradient)))
        update,state=optimizer.update(gradient,state,value);value=optax.apply_updates(value,update);value_count+=1
    saved=load(path/'value_full_updates.npz')
    for k,v in [('indices',ixs),('losses',losses),('norms',norms)]:eq(v,saved[k],k)
    for k,v in gradient.items():eq(v,saved['last_gradient_'+k],k)
    value_changes={k:float(np.linalg.norm(np.asarray(v)-w[k])) for k,v in value.items()}
    assert read(path/'value_update.json')==dict(value_update_count=64,value_changes=value_changes,evidence='value_full_updates.npz')
    after={k:np.asarray(v) for k,v in {**actor,**value}.items()};cp=load(run.OUTPUT/'checkpoint_0001.npz')
    for k,v in after.items():eq(v,cp[k],k)
    assert state_bytes()==(run.OUTPUT/'checkpoint_0001.learner.msgpack').read_bytes()
    assert rng.bit_generator.state==read(run.OUTPUT/'checkpoint_0001_rng.json')==read(run.OUTPUT/'training_0002/learner_rng_before.json')
    batch2=mc_batch(2,after)
    history=read(run.OUTPUT/'training_closed_history.json');assert len(history)==1
    h=history[0];assert h['iteration']==1 and h['actor_updates']==1 and h['value_updates']==64 and h['accepted_backtrack']==accepted
    assert h['actor_changes']==actor_changes and h['value_changes']==value_changes and h['active_samples']==int(mask.sum()) and h['complete_samples']==len(batch['returns'])
    return dict(initial_and_checkpoint1_all_weights_and_full_Adam_count_msgpack_bytes_exact=True,first_complete_MC_and_preupdate_value_advantage_exact=True,first_active_samples=int(mask.sum()),first_complete_samples=len(batch['returns']),first_all_gradient_CG_vectors_scalars_trials_parameters_KL_surrogate_gains_exact=True,cg_iterations=len(trace),accepted_backtrack=accepted,accepted_mean_KL=trials[accepted]['mean_kl'] if accepted is not None else None,actor_updates_completed=1,value_updates_completed=64,value_all_indices_losses_norms_last_gradients_exact=True,learner_rng_exact=True,second_complete_MC_prevalue_advantage_and_before_rng_exact=True,second_complete_samples=len(batch2['returns']),second_active_samples=int(batch2['active'].sum()),second_original_CG_gradient_trials_saved=False,second_original_CG_gradient_trials_reconstructed_as_original=False,cg_direction_storage_note='cg_direction stores the full iterative direction stack; last row is final solve direction. Serializer overwrote the initial single-vector entry; no separate final-vector field exists.',actor_changes=actor_changes,value_changes=value_changes,actor_summary=actor_json,value_losses=losses,value_gradient_norms=norms)

def verify():
    before={str(d):inventory(d) for d in DIRECTORIES};manifest_counts=[]
    manifest_counts.append([str(run.NUMERIC),sources(read(run.NUMERIC/'source_manifest.json'),run.NUMERIC/'executed_sources')])
    failure=read(run.LAUNCH/'failure.json');assert failure['smoke_exit_code']==0 and failure['formal_exit_code']==1
    formal_failure=read(run.OUTPUT/'failure.json');assert 'Exact unchanged policy baseline required' in formal_failure['traceback']
    assert not (run.LAUNCH/'result.json').exists() and not (run.OUTPUT/'results.json').exists()
    for name in ('checkpoint_0002.npz','checkpoint_0002.learner.msgpack','check_0002','candidate'):assert not (run.OUTPUT/name).exists()
    for name in ('actor_full_decisions.npz','actor_update.json','value_full_updates.npz','value_update.json'):assert not (run.OUTPUT/'training_0002'/name).exists()
    for folder in (run.LAUNCH,run.SMOKE,run.OUTPUT):
        c=read(folder/'contract.json');manifest_counts.append([str(folder),sources(c['sources'],folder/'executed_sources')])
        if folder!=run.LAUNCH:
            for p,sha in c['model_files'].items():assert digest(p)==sha
            for p,sha in c['hashes'].items():assert digest(p)==sha
            assert read(folder/'input_hashes_after.json')['source_and_model_files_unchanged']
            for f in folder.glob('worker_source_manifest_*.json'):
                dest=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''));manifest_counts.append([str(f),sources(read(f),dest)])
    preflight=read(ROOT/'tmp/getup_total_target_trpo_r201_preflight_exit_20261011.json');assert preflight['natural_exit'] and preflight['exit_code']==0 and preflight['new_dynamic_attempts']==0 and preflight['source_hashes_unchanged']
    receipt_paths={'GETUP_TOTAL_TARGET_TRPO_PLAN_R201_20261011.json':ROOT/'tmp/GETUP_TOTAL_TARGET_TRPO_PLAN_R201_20261011.json','evidence_preparation_interface_20261011.json':ROOT/'tmp/getup_total_target_trpo_r201_interface_preparation_20261011.json'}
    for p,sha in preflight['source_hashes'].items():assert digest(receipt_paths[p] if p in receipt_paths else Path(run.__file__).with_name(p) if not Path(p).is_absolute() else Path(p))==sha
    for folder in (run.PREFLIGHT,run.LAUNCH/'initial',run.LAUNCH/'formal'):
        log=(folder/'regression.log').read_text();assert 'Ran 16 tests' in log and '\nOK\n' in log
        manifest_counts.append([str(folder),sources(read(folder/'source_manifest.json'),folder/'executed_sources')])
        for f in (folder/'worker_fixture').glob('worker_source_manifest_*.json'):
            dest=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''));manifest_counts.append([str(f),sources(read(f),dest)])
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
            print('R201_FAILURE_PATH_VERIFIED',count,path.parent.name,row['case_seed'],n,flush=True)
    assert count==112 and controls==251297 and baseline_count==31
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
    baseline=report(run.OUTPUT/'development_baseline',CASES);train1=report(run.OUTPUT/'training_0001',CASES);check=report(run.OUTPUT/'check_0001',CASES);train2=report(run.OUTPUT/'training_0002',CASES)
    history=read(run.OUTPUT/'training_closed_history.json');assert len(history)==1 and history[0]['training']==train1 and history[0]['check']==check
    old={r['case_seed']:r for r in baseline['rows']};eligible=bool(check['nominal_success'] and not check['physical_failures'] and check['successes']>baseline['successes'] and all(not old[r['case_seed']]['success'] or r['success'] for r in check['rows']))
    assert not eligible and history[0]['eligible']==eligible
    learner=learning();print('R201_FAILURE_LEARNER_VERIFIED',flush=True)
    assert before=={str(d):inventory(d) for d in DIRECTORIES}
    idle();live=sum(f.stat().st_size for d in DIRECTORIES for f in d.rglob('*') if f.is_file())
    extra=[Path(__file__),*Path(run.__file__).parent.glob('*trpo*r201*.py'),ROOT/'tmp/GETUP_TOTAL_TARGET_TRPO_PLAN_R201_20261011.json',ROOT/'tmp/getup_total_target_trpo_r201_interface_preparation_20261011.json',ROOT/'tmp/getup_total_target_trpo_r201_preflight_exit_20261011.json',ROOT/'tmp/getup_total_target_trpo_r201_launcher_20261011.log',Path(str(run.SMOKE)+'.log'),Path(str(run.OUTPUT)+'.log')]
    live+=sum(p.stat().st_size for p in set(extra) if p.is_file())+sum(p.stat().st_size for p in DEST.rglob('*') if p.is_file())
    assert 3*live<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    return dict(verifier_source_sha256=digest(__file__),audit_complete=True,new_dynamic_attempts=0,formal_failed_naturally_exit_code=1,independent_smoke_exit_code=0,launcher_separate_exit_receipt_present=False,closed_saved_paths=112,controls=controls,baseline_paths_all_old_fields_exact=31,all_causal_features_means_latents_noise_density_requests_rng_exact=True,selector_and_right_feedback_independently_exact=True,post_double_boundary_leg_replacement_and_head_exact=True,planning_after_first_joint_slew_recursive_previous_exact=True,first_applied_R157_boundary_exact=True,all_planned_equals_applied=True,constructor_and_reset_each_40x50=True,formal_and_independent_startup_all_arrays_results_rng_exact=True,learner=learner,first_checkpoint_common_eligible=False,no_R201_final_candidate_or_fallback=True,budget_cap=162,unexecuted_budget=50,unexecuted_budget_may_be_used=False,baseline_successes=baseline['successes'],first_training_successes=train1['successes'],first_check_successes=check['successes'],second_training_successes=train2['successes'],source_manifest_counts=manifest_counts,source_inventory=before,source_evidence_hashes_unchanged=True,full_model_MJB_sha256=MJB,original_physics_sha256=PHYSICS,original_before_request_full_IMU_double_reconstructed=False,first_joint_slew_planning_independently_reconstructed=False,original_failure_missing_second_CG_not_filled=True,cross_implementation_bitwise_or_physical_safety_claim=False,rows=rows,live_bytes_including_known_sources_receipts_logs_and_verifier=live,free_bytes=shutil.disk_usage(ROOT).free,archive_and_Git_not_yet_created=True,best_remains_R157_18_of_24=True,full_task_completed=False,hardware_readiness=False,qualification=False)

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
        print('R201_FAILURE_VERIFICATION_COMPLETE',result['closed_saved_paths'],result['controls'],digest(PROOF),flush=True)
    except Exception:
        import traceback
        with (DEST/'failure.json').open('x') as f:json.dump(dict(traceback=traceback.format_exc(),new_dynamic_attempts=0,proof_saved=PROOF.exists()),f,indent=2)
        raise
if __name__=='__main__':main()
