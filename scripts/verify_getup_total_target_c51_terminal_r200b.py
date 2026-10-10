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
from diagnostics import train_getup_total_target_c51_r200 as run

ROOT=run.ROOT; REPO=ROOT/'github/openduck-mini-'
PREFLIGHT=ROOT/'outputs/getup_total_target_c51_r200_preflight_20261011'
PROOF=ROOT/'tmp/getup_total_target_c51_r200_terminal_verification_20261011.json'
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



def categorical(w,x,xp=np):
    logits=net(w,x,'value',xp).reshape(x.shape[:-1]+(21,51))
    shifted=logits-xp.max(logits,axis=-1,keepdims=True)
    masses=xp.exp(shifted);masses=masses/xp.sum(masses,axis=-1,keepdims=True)
    return xp.sum(masses*xp.asarray(np.linspace(-128.,128.,51)),axis=-1),masses

BOOK=np.zeros((21,10))
for _j in range(10):
    BOOK[1+2*_j,_j]=1e-4;BOOK[2+2*_j,_j]=-1e-4

def independent_learning(result):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jp
    import optax
    from flax import serialization
    w=load(run.OUTPUT/'initial.npz');initial_rng=np.random.default_rng(300)
    for i,(a,b) in enumerate([(101,32),(32,16),(16,1071)]):
        weight=np.zeros((a,b)) if i==2 else initial_rng.normal(0,1/np.sqrt(a),(a,b))
        np.testing.assert_array_equal(weight,w[f'value_w{i}'])
        np.testing.assert_array_equal(np.zeros(b),w[f'value_b{i}'])
    assert len(w)==6
    online={k:jp.asarray(v) for k,v in w.items()};target_weights=jax.tree.map(lambda v:v.copy(),online)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-4))
    state=optimizer.init(online);count=0;rng=np.random.default_rng(300);proof=[];parts=[]
    def state_bytes():return serialization.to_bytes(dict(online=online,target=target_weights,state=state,count=count))
    assert state_bytes()==(run.OUTPUT/'checkpoint_0000.learner.msgpack').read_bytes()
    assert rng.bit_generator.state==read(run.OUTPUT/'checkpoint_0000_rng.json')
    atoms=jp.asarray(np.linspace(-128.,128.,51))
    def target_value(t,nx,r,boot,tail):
        q,p=categorical(t,nx,jp);greedy=jp.argmax(q,axis=1)
        selected=p[jp.arange(len(nx)),greedy]
        raw=r[:,None]+(.99**.2)*(tail[:,None]+boot[:,None]*atoms[None,:])
        clipped=jp.clip(raw,-128.,128.)
        location=(clipped+128.)/(256./50)
        basis=jp.clip(1-jp.abs(location[:,:,None]-jp.arange(51)[None,None,:]),0.,1.)
        mass=jp.sum(selected[:,:,None]*basis,axis=1)
        low=jp.sum(selected*(raw<-128.),axis=1);high=jp.sum(selected*(raw>128.),axis=1)
        return jax.tree.map(jax.lax.stop_gradient,(mass,raw,low,high,greedy,selected))
    def objective(parameters,x,c,mass):
        logits=net(parameters,x,'value',jp).reshape(x.shape[:-1]+(21,51))[jp.arange(len(x)),c]
        return -jp.mean(jp.sum(jax.lax.stop_gradient(mass)*jax.nn.log_softmax(logits,axis=1),axis=1))
    target=jax.jit(target_value);gradient=jax.jit(jax.value_and_grad(objective))
    for iteration in (1,2):
        path=run.OUTPUT/f'training_{iteration:04d}';episodes=[]
        before={k:np.asarray(v).copy() for k,v in online.items()}
        frozen_target={k:np.asarray(v).copy() for k,v in target_weights.items()}
        for case in run.CASES:
            with np.load(path/f'case_{case}/trajectory.npz',allow_pickle=False) as z:
                obs=z['observations'];n=len(obs);rewards=z['original_step_reward'];ret=np.empty(n);total=0.
                for k in range(n-1,-1,-1):total=rewards[k]+(.99**.2)*total;ret[k]=total
                ix=np.arange(1,min(n,529));x=np.stack([sensor_features(obs[k],obs[0],int(k)) for k in ix])
                expected=(np.arange(n)>0)&(np.arange(n)<529);np.testing.assert_array_equal(expected,z['policy_active'])
                # Recompute from actual causal observations; recorded code is only a comparison.
                decisions=[];row=read(path/f'case_{case}/result.json');sample_rng=np.random.default_rng(row['sampling_seed'])
                for index,feature in zip(ix,x):
                    q,p=categorical(before,feature);greedy=int(np.argmax(q))
                    gate=float(sample_rng.random());draw=int(sample_rng.integers(21));chosen=draw if gate<.1 else greedy
                    for key,value in [('policy_code',chosen),('policy_greedy_code',greedy),('policy_exploration_gate',gate),('policy_exploration_code',draw),('policy_expected_values',q),('policy_selected_probabilities',p[chosen]),('policy_requested_action',BOOK[chosen])]:
                        np.testing.assert_array_equal(value,z[key][index])
                    decisions.append(chosen)
                assert sample_rng.bit_generator.state==read(path/f'case_{case}/rng.json')
                nx=np.stack([sensor_features(obs[k+1] if k<n-1 else obs[k],obs[0],int(k+1)) for k in ix])
                boot=np.asarray([float(k<n-1 and k+1<529) for k in ix])
                tail=np.asarray([float(ret[k+1]) if k<n-1 and k+1==529 else 0. for k in ix])
                episodes.append(dict(features=x,action_code=np.asarray(decisions,dtype=np.int64),reward=rewards[ix],next_features=nx,bootstrap=boot,home_tail=tail,terminal=ix==n-1,control=ix))
        new={key:np.concatenate([e[key] for e in episodes]) for key in episodes[0]}
        saved=load(path/'new_episode_replay.npz');assert saved.keys()==new.keys()
        for key in new:np.testing.assert_array_equal(new[key],saved[key])
        parts.append(new);replay={key:np.concatenate([p[key] for p in parts]) for key in new}
        assert read(path/'replay_manifest.json')==dict(training_rounds=list(range(1,iteration+1)),transitions=len(replay['reward']),old_data_used=False)
        saved=load(path/'learner_update.npz');stats=[];norms=[];indices=[];projection_error=0.
        for step in range(64):
            ix=rng.integers(0,len(replay['reward']),size=512);indices.append(ix)
            x,c,r,nx,boot,tail=[jp.asarray(replay[k][ix]) for k in ('features','action_code','reward','next_features','bootstrap','home_tail')]
            mass,raw,low,high,greedy,selected=target(target_weights,nx,r,boot,tail)
            for key,value in [('projected_masses',mass),('unbounded_target_atoms',raw),('next_greedy_codes',greedy),('next_selected_probabilities',selected)]:
                np.testing.assert_array_equal(value,saved[key][step])
            np.testing.assert_allclose(np.asarray(mass).sum(axis=1),1.,atol=2e-14,rtol=0)
            # Alternate adjacent-bin accumulation checks the projection independently of triangular evaluation.
            raw_np=np.asarray(raw);prob_np=np.asarray(selected)
            location=(np.clip(raw_np,-128.,128.)+128.)/(256./50)
            floor=np.floor(location).astype(int);ceil=np.ceil(location).astype(int)
            alternate=np.zeros((512,51));rows=np.repeat(np.arange(512),51)
            same=floor==ceil
            np.add.at(alternate,(rows,floor.ravel()),(prob_np*np.where(same,1.,ceil-location)).ravel())
            np.add.at(alternate,(rows,ceil.ravel()),(prob_np*np.where(same,0.,location-floor)).ravel())
            projection_error=max(projection_error,float(np.abs(alternate-np.asarray(mass)).max()))
            np.testing.assert_allclose(alternate,mass,atol=1e-14,rtol=0)
            loss,g=gradient(online,x,c,mass);change,state=optimizer.update(g,state,online)
            online=optax.apply_updates(online,change);count+=1
            stats.append([float(loss),float(jp.mean(low)),float(jp.mean(high)),float(jp.min(raw)),float(jp.max(raw)),count])
            norms.append(float(optax.global_norm(g)))
            for key,value in target_weights.items():np.testing.assert_array_equal(value,frozen_target[key])
        target_weights=jax.tree.map(lambda v:v.copy(),online)
        for key,value in [('statistics',stats),('gradient_norms',norms),('replay_indices',indices)]:
            np.testing.assert_array_equal(value,saved[key])
        for key,value in g.items():np.testing.assert_array_equal(value,saved['last_gradient_'+key])
        checkpoint=load(run.OUTPUT/f'checkpoint_{iteration:04d}.npz')
        for key,value in online.items():np.testing.assert_array_equal(value,checkpoint[key])
        assert state_bytes()==(run.OUTPUT/f'checkpoint_{iteration:04d}.learner.msgpack').read_bytes()
        assert rng.bit_generator.state==read(run.OUTPUT/f'checkpoint_{iteration:04d}_rng.json')
        changes={key:float(np.linalg.norm(np.asarray(value)-before[key])) for key,value in online.items()}
        h=result['history'][iteration-1]
        assert changes==h['changes'] and h['categorical_updates']==64 and h['hard_target_copies']==1
        proof.append(dict(iteration=iteration,new_replay_transitions=len(new['reward']),pooled_replay_transitions=len(replay['reward']),categorical_updates=64,hard_target_copies=1,target_frozen_for_all_64_updates_exact=True,replay_and_original_home_terminal_backup_exact=True,all_indices_projected_masses_unbounded_atoms_greedy_probabilities_exact=True,alternate_adjacent_bin_projection_max_error=projection_error,statistics=stats,gradient_norms=norms,last_gradients_exact=True,online_target_Adam_count_serialization_bytes_exact=True,rng_exact=True,changes=changes,old_data_used=False))
    return proof

def verify():
    before={d.name:inventory(d) for d in DIRECTORIES}
    print('R200_VERIFY_BEGIN',flush=True)
    launch=read(run.LAUNCH/'result.json');result=read(run.OUTPUT/'results.json')
    assert not any(list(d.rglob('failure.json')) for d in DIRECTORIES)
    assert launch['source_hashes_unchanged'] and launch['smoke_gate_passed']
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
    preflight=read(ROOT/'tmp/getup_total_target_c51_r200_preflight_exit_20261011.json');assert preflight['natural_exit'] and preflight['exit_code']==0 and preflight['new_dynamic_attempts']==0 and preflight['source_hashes_unchanged']
    for p,sha in preflight['source_hashes'].items():assert digest(p)==sha
    for folder in (PREFLIGHT,run.LAUNCH/'initial',run.LAUNCH/'formal'):
        log=(folder/'regression.log').read_text();assert 'Ran 25 tests' in log and '\nOK\n' in log
        sources(read(folder/'source_manifest.json'),folder/'executed_sources')
        for f in (folder/'worker_fixture').glob('worker_source_manifest_*.json'):
            dest=f.with_name(f.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''))
            sources(read(f),dest)
        for name in ('total_target_c51_kernel_r200.py','train_getup_total_target_c51_r200.py','test_getup_total_target_c51_r200.py','launch_getup_total_target_c51_r200.py'):
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
                    requested=np.zeros(10);chosen=greedy=draw=0;gate_draw=0.
                    expected_values=np.zeros(21);selected_probabilities=np.zeros(51)
                    if active:
                        q,p=categorical(w,sensor_features(obs[k],initial,k));greedy=int(np.argmax(q))
                        gate_draw=float(rng.random()) if mode=='sample' else 0.
                        draw=int(rng.integers(21)) if mode=='sample' else 0
                        chosen=draw if mode=='sample' and gate_draw<.1 else greedy
                        requested=BOOK[chosen].copy();expected_values=q;selected_probabilities=p[chosen]
                    for key,val in [('policy_requested_action',requested),('policy_code',chosen),('policy_greedy_code',greedy),('policy_expected_values',expected_values),('policy_selected_probabilities',selected_probabilities),('policy_exploration_gate',gate_draw),('policy_exploration_code',draw)]:
                        np.testing.assert_array_equal(val,z[key][k])
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
    for name,mode,cases,parity in run.SMOKE_SPEC:
        for case in cases:
            a=run.OUTPUT/name/f'case_{case}';b=run.SMOKE/name/f'case_{case}'
            za=load(a/'trajectory.npz');zb=load(b/'trajectory.npz');assert za.keys()==zb.keys()
            for key in za:np.testing.assert_array_equal(za[key],zb[key])
            assert read(a/'result.json')==read(b/'result.json') and read(a/'rng.json')==read(b/'rng.json')
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
    print('R200_SCALAR_COMPLETE',counts,controls,flush=True)
    learner=independent_learning(result)
    print('R200_LEARNER_COMPLETE',flush=True)
    assert before=={d.name:inventory(d) for d in DIRECTORIES}
    live=sum(f.stat().st_size for d in DIRECTORIES for f in d.rglob('*') if f.is_file())
    assert 3*live<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    return dict(verifier_source_sha256=digest(__file__),terminal_result_saved=True,natural_exit=True,new_dynamic=0,closed_paths=162,controls=controls,baseline_paths_all_old_fields_bitwise=baseline_paths,policy_code_greedy_expected_returns_probabilities_two_draw_rng_request_independently_exact=True,post_double_boundary_replacement_independently_exact=True,planning_controls_after_first_independently_exact=True,first_control_original_R157_applied_parity=True,all_planned_equals_applied=True,learner=learner,original_before_request_IMU_double_reconstructed=False,whole_old_feedback_independently_recomputed=False,cross_implementation_bitwise_or_safety_claim=False,test_startup_mapping_manifest_verified=True,explicit_four_test_source_copies_verified=True,rows=rows,source_inventory=before,live_bytes=live,full_task_completed=False,hardware_readiness=False,qualification=False)


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

