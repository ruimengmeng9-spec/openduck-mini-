"""Read-only scalar, companion return, complete REINFORCE and Adam audit."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_full_episode_pg_r181 as run

OUTPUT=run.ROOT/'outputs/getup_full_episode_pg_terminal_audit_r182_20261010'
SMOKE=run.ROOT/'outputs/getup_full_episode_pg_terminal_audit_r182_smoke_20261010'
FIELDS=('observations','preparation_sensors','local_hip_extra_rad','pg_features','pg_hidden','pg_noise_amplitude','pg_mean_latent','pg_sampled_latent','pg_innovation','pg_extra_rad','original_step_rewards','planned_before_integration_rad','same_state_unperturbed_planned_rad','original_double_pre_slew_target_rad','adjusted_double_pre_slew_target_rad','applied')


def read(path):
    return json.loads(Path(path).read_text())


def actor(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}


def eq_actor(a,b):
    assert set(a)==set(b)
    for key in a:np.testing.assert_array_equal(a[key],b[key],err_msg=key)


def scalar(path,nominal,model,save=None):
    row=read(path/'result.json');p={k:np.asarray(v,dtype=float) for k,v in row['parameters'].items()}
    with np.load(path/'trajectory.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in FIELDS}
    obs=data['observations'];hidden=np.zeros(run.HIDDEN);records={k:[] for k in ('pg_features','pg_hidden','pg_noise_amplitude','pg_mean_latent','pg_sampled_latent','pg_extra_rad','local_hip_extra_rad')}
    rng=np.random.default_rng(row['exploration_seed'] if row['exploration_seed'] is not None else 0)
    gains,choice,logits=run.selector.old.predict(model,data['preparation_sensors'][-1])
    np.testing.assert_array_equal(gains,row['base_gains']);assert choice==row['base_choice'];np.testing.assert_array_equal(logits,row['initial_sensor_logits'])
    for k,x in enumerate(obs):
        if k<529:
            f,amp,_=run.causal_features(x,nominal[k],obs[0],nominal[0],k)
            innovation=rng.normal(size=14) if row['exploration_seed'] is not None else np.zeros(14)
            np.testing.assert_array_equal(innovation,data['pg_innovation'][k])
            extra,hidden,mean,latent=run.actor_step(p,f,hidden,amp,innovation)
            base=run.local.local_feedback(x,nominal[k],[9,10,11],gains)
        else:
            f=np.zeros(run.FEATURES);amp=0.;extra=np.zeros(14);hidden=np.zeros(run.HIDDEN);mean=np.zeros(14);latent=np.zeros(14);base=np.zeros(3)
            np.testing.assert_array_equal(data['pg_innovation'][k],np.zeros(14))
        for key,value in zip(records,(f,hidden,amp,mean,latent,extra,base)):records[key].append(np.copy(value))
    for key,value in records.items():np.testing.assert_array_equal(value,data[key],err_msg=str(path)+' '+key)
    for key in ('pg_features','pg_hidden','pg_noise_amplitude','pg_mean_latent','pg_sampled_latent','pg_extra_rad'):
        np.testing.assert_array_equal(data[key][0],np.zeros_like(data[key][0]))
        if len(obs)>529:np.testing.assert_array_equal(data[key][529:],np.zeros_like(data[key][529:]))
        if row['case_seed'] is None:np.testing.assert_array_equal(data[key],np.zeros_like(data[key]))
    np.testing.assert_array_equal(data['pg_sampled_latent'],data['pg_mean_latent']+run.SIGMA*data['pg_noise_amplitude'][:,None]*data['pg_innovation'])
    np.testing.assert_array_equal(data['planned_before_integration_rad'],data['applied'])
    reward_sum=sum(float(v) for v in data['original_step_rewards'])
    assert reward_sum==row['full_episode_original_reward_sum']
    assert abs(reward_sum-row['return_sum'])<=1e-10
    assert np.abs(data['pg_hidden']).max()<=1 and row['maximum_combined_14_joint_correction_rad']<=.18+1e-12
    direct=data['adjusted_double_pre_slew_target_rad']-data['original_double_pre_slew_target_rad']
    post=data['planned_before_integration_rad']-data['same_state_unperturbed_planned_rad']
    if save is not None:
        save.mkdir(parents=True,exist_ok=False);np.savez_compressed(save/'signals.npz',**data,same_state_pre_slew_direct_new_rad=direct,same_state_post_slew_direct_new_rad=post)
    return dict(path=str(path),case_seed=row['case_seed'],success=row['success'],valid=row['valid'],controls=row['controls'],initial_hash=row['initial_hash'],peaks=row['peaks'],entry_time_s=row['entry_time_s'],strict_tail_s=row['strict_tail_s'],exploration_seed=row['exploration_seed'],scalar_bitwise=True,innovation_rng_bitwise=True,reward_sum=reward_sum,maximum_extra=float(np.abs(data['pg_extra_rad']).max()),maximum_hidden=float(np.abs(data['pg_hidden']).max()),maximum_pre_slew_direct=float(np.abs(direct).max()),maximum_post_slew_direct=float(np.abs(post).max()),combined_cap=row['maximum_combined_14_joint_correction_rad'])


def terminal_parity():
    pairs=[]
    for case in run.CASES:
        paths=[run.OUTPUT/name/f'case_{case}' for name in ('development_candidate','development_baseline')]+[run.selector.OUTPUT/'candidate'/f'case_{case}']
        rows=[read(p/'result.json') for p in paths]
        assert rows[0]['initial_hash']==rows[1]['initial_hash']==rows[2]['initial_hash']
        # Root arrays are decoded only here for old complete-field equality,
        # never by scalar/gradient reconstruction or as runtime policy input.
        with np.load(paths[0]/'trajectory.npz',allow_pickle=False) as c,np.load(paths[1]/'trajectory.npz',allow_pickle=False) as b,np.load(paths[2]/'trajectory.npz',allow_pickle=False) as old:
            assert c.files==b.files
            for key in c.files:np.testing.assert_array_equal(c[key],b[key],err_msg=key)
            for key in old.files:np.testing.assert_array_equal(b[key],old[key],err_msg=key)
        assert rows[0]['peaks']==rows[1]['peaks']==rows[2]['peaks']
        assert all(r['success']==rows[0]['success'] and r['valid']==rows[0]['valid'] for r in rows)
        pairs.append(dict(case_seed=case,initial_hash=rows[0]['initial_hash'],all_candidate_baseline_fields_bitwise=True,baseline_R157_original_fields_bitwise=True,original_labels_peaks_equal=True))
    return pairs


def learner_audit(output,closed):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jp
    import optax
    from flax import serialization
    p={k:jp.asarray(v) for k,v in run.initial_actor().items()}
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.001));state=optimizer.init(p)
    rng=np.random.default_rng(run.SEED);value_grad=jax.jit(jax.value_and_grad(run.score_loss));results=[]
    for number in range(5):
        cp=run.OUTPUT/'checkpoints'/f'update_{number:04d}'
        eq_actor({k:np.asarray(v) for k,v in p.items()},actor(cp/'actor.npz'))
        # Byte equality also verifies complete optimizer moments/count, not
        # merely final weights or aggregate gradient norms.
        assert serialization.to_bytes(dict(actor=p,optimizer=state))==(cp/'learner.msgpack').read_bytes()
        assert rng.bit_generator.state==read(cp/'rng.json')
        assert read(cp/'history.json')==closed['history'][:number]
        if number==4:break
        history=closed['history'][number];features=[];latents=[];amplitudes=[];returns=[];seeds=[];paths=[]
        for repeat in range(2):
            draw=[int(v) for v in rng.integers(1,2**62,size=len(run.CASES))];seeds.append(draw)
            report=read(run.OUTPUT/'training'/f'update_{number+1:04d}'/f'replicate_{repeat}'/'results.json')
            assert report==history['closed_training_reports'][repeat]
            returns.append([r['return_sum'] for r in report['rows']])
            for case,seed,row in zip(run.CASES,draw,report['rows']):
                assert row['case_seed']==case and row['exploration_seed']==seed
                eq_actor(p,{k:np.asarray(v) for k,v in row['parameters'].items()})
                path=run.OUTPUT/'training'/f'update_{number+1:04d}'/f'replicate_{repeat}'/f'case_{case}';paths.append(str(path))
                with np.load(path/'trajectory.npz',allow_pickle=False) as z:
                    n=min(529,len(z['pg_features']));f=np.zeros((529,run.FEATURES));l=np.zeros((529,14));a=np.zeros(529)
                    f[:n]=z['pg_features'][:n];l[:n]=z['pg_sampled_latent'][:n];a[:n]=z['pg_noise_amplitude'][:n]
                    expected=np.asarray(run.differentiable_means(p,jp.asarray(f)))
                    np.testing.assert_allclose(expected[:n],z['pg_mean_latent'][:n],atol=1e-12,rtol=1e-12)
                features.append(f);latents.append(l);amplitudes.append(a)
        assert seeds==history['exploration_seeds'] and all(a!=b for a,b in zip(*seeds))
        r=np.asarray(returns);advantages=np.concatenate([r[0]-r[1],r[1]-r[0]])
        np.testing.assert_array_equal(r,history['return_pair_matrix']);np.testing.assert_array_equal(advantages,history['companion_advantages'])
        loss,grad=value_grad(p,jp.asarray(features),jp.asarray(latents),jp.asarray(amplitudes),jp.asarray(advantages))
        assert float(loss)==history['score_loss']
        change,state=optimizer.update(grad,state,p);previous=p;p=optax.apply_updates(p,change)
        norms={k:float(np.linalg.norm(np.asarray(v))) for k,v in grad.items()}
        changes={k:float(np.max(np.abs(np.asarray(p[k])-np.asarray(previous[k])))) for k in p}
        assert norms==history['gradient_norms'] and changes==history['maximum_parameter_changes']
        dest=output/'learner_reconstruction'/f'update_{number+1:04d}';dest.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(dest/'gradient.npz',**{k:np.asarray(v) for k,v in grad.items()})
        np.savez_compressed(dest/'batch_returns_advantages.npz',returns=r,advantages=advantages,seeds=np.asarray(seeds,dtype=np.int64))
        result=dict(update=number+1,score_loss=float(loss),gradient_norms=norms,maximum_parameter_changes=changes,complete_actor_optimizer_msgpack_bitwise=True,companion_return_advantages_bitwise=True,complete_rng_history_exact=True,numpy_jax_mean_tolerance=1e-12,paths=paths)
        run.local.write_json(dest/'result.json',result);results.append(result)
        print('R182_LEARNER_UPDATE_EXACT',number+1,flush=True)
    assert rng.bit_generator.state==closed['rng']
    return results


def groups(smoke):
    startup=[run.OUTPUT/'zero_parity',run.OUTPUT/'nonzero_smoke',*[run.OUTPUT/'exploration_smoke'/f'replicate_{i}' for i in range(2)]]
    if smoke:return startup
    return startup+[run.OUTPUT/'development_baseline',*[run.OUTPUT/'training'/f'update_{n:04d}'/f'replicate_{r}' for n in range(1,5) for r in range(2)],*[run.OUTPUT/'development_checkpoints'/f'update_{n:04d}' for n in range(1,5)],run.OUTPUT/'development_candidate']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();output=SMOKE if args.smoke else OUTPUT
    assert not output.exists();terminal=read(run.OUTPUT/'results.json');closed=read(run.OUTPUT/'training_closed.json')
    assert terminal['terminal_result_saved'] and closed['updates']==4 and len(closed['history'])==4
    eq_actor(actor(run.OUTPUT/'selected_actor.npz'),run.initial_actor())
    tracked=[p for p in run.OUTPUT.rglob('*') if p.is_file() and 'startup_closed_01' not in p.parts]
    tracked+=[Path(__file__),Path(run.__file__),run.local.prior.program.SCENE,run.local.prior.program.STAND,run.local.prior.program.REFERENCE]
    before={str(p):run.prior.digest(p) for p in tracked}
    output.mkdir();src=output/'executed_sources';src.mkdir()
    for p in (Path(__file__),Path(run.__file__)):shutil.copy2(p,src/p.name)
    with np.load(run.OUTPUT/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    model=actor(run.OUTPUT/'frozen/initial_selector.npz');baseline={r['case_seed']:r for r in terminal['baseline']['rows']}
    programs=[];count=0
    for group in groups(args.smoke):
        report=read(group/'results.json');rows=[]
        for row in report['rows']:
            save=None
            if group==run.OUTPUT/'nonzero_smoke':save=output/'probe_signals'/f"case_{row['case_seed']}"
            if not args.smoke and group==run.OUTPUT/'development_checkpoints/update_0002':save=output/'highest_nonzero_full_development_signals'/f"case_{row['case_seed']}"
            rows.append(scalar(group/f"case_{row['case_seed']}",nominal,model,save));count+=1
        programs.append(dict(path=str(group),successes=report['successes'],physical_failures=report['physical_failures'],rescued=[r['case_seed'] for r in report['rows'] if r['case_seed'] is not None and r['success'] and not baseline[r['case_seed']]['success']],regressed=[r['case_seed'] for r in report['rows'] if r['case_seed'] is not None and not r['success'] and baseline[r['case_seed']]['success']],rows=rows))
        print('R182_SCALAR_GROUP_EXACT',str(group),len(rows),flush=True)
    assert count==(8 if args.smoke else 358)
    pairs=terminal_parity();learner=[] if args.smoke else learner_audit(output,closed)
    after={str(p):run.prior.digest(p) for p in tracked};assert before==after
    result=dict(read_only=True,smoke=args.smoke,scalar_attempts=count,programs=programs,terminal_pairing=pairs,learner_updates=learner,feedback_scalar_whitelist=list(FIELDS),source_hashes_unchanged=True,hashes_before=before,root_arrays_only_separate_fullfield_parity=True,no_environment_MjData_forward_integration_force_inference=True,no_new_dynamic_replays=True,standalone_joint_slew_planning_not_independently_recomputed=True,original_labels_physics_not_reclassified=True,terminal_result_saved=True,full_task_completed=False,hardware_readiness=False,qualification_executed=False)
    run.local.write_json(output/'results.json',result);print('R182_READ_ONLY_CLOSED',count,len(pairs),len(learner),flush=True)


if __name__=='__main__':main()
