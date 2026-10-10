"""Whole-episode recurrent score-function optimization, simulation only.

Every learned input/recurrent/output parameter participates in autodiff.
Independent companion episode returns form a case-conditioned baseline.
No teacher loss, forward model, CEM, short-tail success objective or lookup.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import numpy as np
from diagnostics import train_getup_recurrent_readout_r175 as prior

local=prior.local;selector=prior.selector;planner=prior.planner;ROOT=prior.ROOT
OUTPUT=ROOT/'outputs/getup_full_episode_pg_r181_left_20261010'
SMOKE=ROOT/'outputs/getup_full_episode_pg_r181_smoke_20261010'
CASES=prior.CASES;HIDDEN=8;FEATURES=150;SIGMA=.0001;SEED=281
MODEL=None


def initial_actor():
    rng=np.random.default_rng(SEED)
    return dict(input_raw=rng.normal(0,.3,(HIDDEN,FEATURES)),
                recurrence_raw=rng.normal(0,.1,(HIDDEN,HIDDEN)),output_raw=np.zeros((14,HIDDEN)))


def check_actor(p):
    for key,shape in [('input_raw',(HIDDEN,FEATURES)),('recurrence_raw',(HIDDEN,HIDDEN)),('output_raw',(14,HIDDEN))]:
        if key not in p or np.shape(p[key])!=shape or not np.isfinite(p[key]).all():raise ValueError('Finite complete recurrent actor required')


def matrices(p,xp=np):
    a=.5*xp.tanh(p['input_raw'])/np.sqrt(FEATURES)
    t=xp.tanh(p['recurrence_raw']);b=.5*t/(1+xp.sum(xp.abs(t),axis=1,keepdims=True))
    c=.1*xp.tanh(p['output_raw'])
    return a,b,c


def causal_features(current,nominal,initial,initial_nominal,control):
    values=[np.asarray(v,dtype=np.float32) for v in (current,nominal,initial,initial_nominal)]
    if any(v.shape!=(55,) or not np.isfinite(v).all() for v in values):raise ValueError('Finite causal actual sensor55 required')
    if not isinstance(control,(int,np.integer)) or not 0<=control<529:raise ValueError('Recovery control index required')
    x,n,x0,n0=values
    initial_error=(x0[:50].astype(float)-n0[:50].astype(float))/prior.SCALE
    delta=((x[:50].astype(float)-n[:50].astype(float))-(x0[:50].astype(float)-n0[:50].astype(float)))/prior.SCALE
    phi=np.tanh(delta);context=np.tanh(initial_error)
    # Current causal error-change, its interaction with causal initial context,
    # and a fixed smooth phase modulation. No case-specific centers or library.
    f=np.concatenate([phi,phi*context,phi*np.sin(np.pi*control/529.)])
    amplitude=float(np.sqrt(np.mean(phi*phi)))
    return f,amplitude,delta


def actor_step(p,features,state,amplitude,innovation):
    check_actor(p);f=np.asarray(features,dtype=float);h=np.asarray(state,dtype=float);eps=np.asarray(innovation,dtype=float)
    if f.shape!=(FEATURES,) or h.shape!=(HIDDEN,) or eps.shape!=(14,) or not all(np.isfinite(v).all() for v in (f,h,eps)):raise ValueError('Finite causal actor state required')
    if np.abs(f).max()>1 or np.abs(h).max()>1 or not np.isfinite(amplitude) or not 0<=amplitude<=1:raise ValueError('Bounded actor features/state required')
    a,b,c=matrices(p);hidden=np.tanh(a@f+b@h);mean=c@hidden
    latent=mean+SIGMA*amplitude*eps;extra=.18*np.tanh(latent)
    return extra,hidden,mean,latent


def init_worker(frozen):
    global MODEL
    path=Path(frozen)
    with np.load(path/'initial_selector.npz',allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}
    with np.load(path/'snapshot.npz',allow_pickle=False) as z:local.WEIGHTS={k:z[k].copy() for k in z.files}
    with np.load(path/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:local.NOMINAL=z['observations'].copy()


def evaluate(job):
    parameters,case,directory,parity,exploration_seed=job
    p={k:np.asarray(v,dtype=float) for k,v in parameters.items()};check_actor(p);assert case in CASES
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,SEED,True)
    env.reset(case);sim=env.sim;fingerprint=prior.physics_hash(sim)
    ids=np.array([sim.model.actuator(n).id for n in local.JOINTS]);sensors=np.array([9,10,11])
    gains,choice,logits=selector.old.predict(MODEL,env.preparation_sensors[-1])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    initial=env.observe().copy();initial_nominal=local.NOMINAL[0].copy();hidden=np.zeros(HIDDEN)
    rng=np.random.default_rng(exploration_seed if exploration_seed is not None else 0)
    original=sim.step_target;records=[];extras=[];states=[];fs=[];amps=[];means=[];latents=[];innovations=[];rewards=[];frozen_history=[]
    planned=[];base_planned=[];fixed_targets=[];new_targets=[];maxima=[]
    def controlled(target):
        nonlocal hidden
        k=env.controls;observed=env.observe().copy()
        np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
        frozen=local.local_feedback(observed,local.NOMINAL[k],sensors,gains) if k<529 else np.zeros(3)
        fixed=local.merge_target(target,env.targets[k],frozen,ids,sim.lower,sim.upper)
        if k<529:
            features,amplitude,_=causal_features(observed,local.NOMINAL[k],initial,initial_nominal,k)
            innovation=rng.normal(size=14) if exploration_seed is not None else np.zeros(14)
            extra,hidden,mean,latent=actor_step(p,features,hidden,amplitude,innovation)
        else:
            features=np.zeros(FEATURES);amplitude=0.;innovation=np.zeros(14);extra=np.zeros(14);hidden=np.zeros(HIDDEN);mean=np.zeros(14);latent=np.zeros(14)
        adjusted=local.merge_target(fixed,env.targets[k],extra,np.arange(14),sim.lower,sim.upper)
        assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
        if k==0:
            for v in (features,extra,hidden,mean,latent):np.testing.assert_array_equal(v,np.zeros_like(v))
            assert amplitude==0 and adjusted is fixed
        wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper);unperturbed=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
        fs.append(features.copy());amps.append(amplitude);extras.append(extra.copy());states.append(hidden.copy());means.append(mean.copy());latents.append(latent.copy());innovations.append(innovation.copy());frozen_history.append(frozen.copy())
        planned.append(wanted.copy());base_planned.append(unperturbed.copy());fixed_targets.append(fixed.copy());new_targets.append(adjusted.copy());maxima.append(float(np.abs(adjusted-env.targets[k]).max()))
        applied=original(adjusted);np.testing.assert_array_equal(applied,wanted);np.testing.assert_array_equal(sim.prev,wanted);return applied
    sim.step_target=controlled
    try:
        while True:
            obs=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False);rewards.append(float(step[1]))
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    finally:sim.step_target=original
    row=step[5]
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],preparation_sensors=env.preparation_sensors,local_hip_extra_rad=frozen_history,
        pg_extra_rad=extras,pg_hidden=states,pg_features=fs,pg_noise_amplitude=amps,pg_mean_latent=means,pg_sampled_latent=latents,pg_innovation=innovations,original_step_rewards=rewards,
        planned_before_integration_rad=planned,same_state_unperturbed_planned_rad=base_planned,original_double_pre_slew_target_rad=fixed_targets,adjusted_double_pre_slew_target_rad=new_targets)
    reference=selector.OUTPUT/'candidate'/f'case_{case}';old=json.loads((reference/'result.json').read_text());assert row['initial_hash']==old['initial_hash']
    with np.load(reference/'trajectory.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.asarray(arrays['applied'])[0],z['applied'][0])
        if parity or case is None:
            equality={k:bool(np.array_equal(np.asarray(arrays[k]),z[k])) for k in z.files};assert all(equality.values()),equality;assert row['peaks']==old['peaks'];row['complete_R157_parity']=equality
    if case is None:
        for v in (extras,states,fs,amps,means,latents):np.testing.assert_array_equal(v,np.zeros_like(v))
    assert np.isclose(sum(rewards),row['return_sum'],atol=1e-10,rtol=0)
    row.update(parameters={k:v.tolist() for k,v in p.items()},exploration_seed=exploration_seed,base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),full_path=True,
        maximum_combined_14_joint_correction_rad=max(maxima),maximum_pg_extra_rad=float(np.abs(extras).max()),maximum_hidden=float(np.abs(states).max()),
        online_planned_target_equals_actual_applied_bitwise=True,physics_sha256=fingerprint,physics_unchanged=fingerprint==prior.physics_hash(sim),root_edits_during_recovery=0,
        no_root_truth_or_case_metadata_in_controller=True,teacher_labels_used=False,full_episode_original_reward_sum=sum(rewards))
    assert row['physics_unchanged']
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row


def group(pool,p,cases,path,parity=False,seeds=None):
    seeds=[None]*len(cases) if seeds is None else seeds
    report=local.aggregate(list(pool.map(evaluate,[(p,case,str(path/f'case_{case}'),parity,seed) for case,seed in zip(cases,seeds)])))
    local.write_json(path/'results.json',report);return report


def differentiable_means(p,features):
    import jax
    import jax.numpy as jp
    a,b,c=matrices(p,jp)
    def step(h,f):
        h=jp.tanh(a@f+b@h);return h,c@h
    return jax.lax.scan(step,jp.zeros(HIDDEN),features)[1]


def score_loss(p,features,latents,amplitudes,advantages):
    import jax
    import jax.numpy as jp
    means=jax.vmap(differentiable_means,in_axes=(None,0))(p,features)
    positive=amplitudes>0
    sd=SIGMA*jp.where(positive,amplitudes,1.)
    logp=-.5*jp.sum(((latents-means)/sd[...,None])**2,axis=-1)
    # Variance and Jacobian terms are independent of actor parameters for a
    # given sampled sensor history. Degenerate exact-zero decisions are omitted.
    sequence_scores=jp.sum(jp.where(positive,logp,0.),axis=1)
    return -jp.mean(advantages*sequence_scores)


def compare_smoke(output):
    for name,cases in [('zero_parity',[None,769000,773004]),('nonzero_smoke',[None,769002,773004])]:
        for case in cases:
            fresh=output/name/f'case_{case}';old=SMOKE/name/f'case_{case}'
            with np.load(fresh/'trajectory.npz',allow_pickle=False) as a,np.load(old/'trajectory.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            x=json.loads((fresh/'result.json').read_text());y=json.loads((old/'result.json').read_text());assert x['initial_hash']==y['initial_hash'] and x['peaks']==y['peaks'] and x['success']==y['success']
    for repeat in range(2):
        fresh=output/'exploration_smoke'/f'replicate_{repeat}'/'case_769000';old=SMOKE/'exploration_smoke'/f'replicate_{repeat}'/'case_769000'
        with np.load(fresh/'trajectory.npz',allow_pickle=False) as a,np.load(old/'trajectory.npz',allow_pickle=False) as b:
            assert a.files==b.files
            for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
        x=json.loads((fresh/'result.json').read_text());y=json.loads((old/'result.json').read_text());assert x['initial_hash']==y['initial_hash'] and x['peaks']==y['peaks'] and x['success']==y['success']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>12*1024**3
    output.mkdir();src=output/'executed_sources';src.mkdir()
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if name.startswith('diagnostics.') and f and Path(f).suffix=='.py':shutil.copy2(f,src/Path(f).name)
    for name in [Path(__file__).name,'test_getup_full_episode_pg_r181.py','launch_getup_full_episode_pg_r181.py']:shutil.copy2(Path(__file__).with_name(name),src/name)
    frozen=output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:shutil.copy2(source,frozen/name);assert prior.digest(source)==prior.digest(frozen/name)
    p=initial_actor();probe={k:v.copy() for k,v in p.items()};probe['output_raw']=np.random.default_rng(1810281).choice([-1.,1.],(14,HIDDEN))*.005
    np.savez_compressed(frozen/'initial_actor.npz',**p);np.savez_compressed(frozen/'fixed_probe.npz',**probe)
    local.write_json(output/'contract.json',dict(seed=SEED,smoke=args.smoke,simulation_only=True,updates=4,workers=6,independent_replicates_per_case=2,training_cases=CASES,training_attempt_budget=200,
        hypothesis='Train all nonlinear recurrent actor matrices from whole original episode returns, rather than frozen feature CEM or short629-step PPO/teacher-action fitting; independent same-case companion return reduces baseline variance without online labels.',
        actor_parameters=1376,hidden=HIDDEN,features=FEATURES,input_parameterization='.5*tanh(Araw)/sqrt150',recurrence_parameterization='.5*tanh(Braw)/(1+rowL1)',output_parameterization='.1*tanh(Craw)',
        learning_method='On-policy whole-episode REINFORCE; per-case independent companion baseline; one Adam update per closed50-attempt batch; learning rate .001; gradient global norm1; no critic, sample reuse, CEM, teacher loss or forward model.',
        fixed_iid_latent_std=SIGMA,exploration_state_amplitude='RMS tanh causal actual50 error-change; exactly zero on initial/nominal',training_objective='Undiscounted sum of unmodified original per-step rewards through original complete episode, including original home; no new reward or short success objective',
        gradient_scope='Input, recurrence and output all learnable; hidden gradients may initially be zero with zero output head; record actual per-block gradients and changes, no claim all blocks actually changed before measurement.',
        runtime_inputs='Current/causal initial actual native50, past internal recurrent state, fixed same-phase nominal sensor and phase; no case/seed/label/root/future/lookup. Exploration seed is experiment RNG only, never policy input.',
        controls=2279,control_hz=50,physics_hz=500,entry_deadline_s=12,strict_tail_s=30,combined_14_joint_cap_rad=.18,home_original=True,physics_reward_acceptance_original=True,
        nominal_scalar_exact_zero=True,first_target_exact_original=True,zero_object_identity_preserved=True,no_mid_episode_root_edits=True,qualification_never_loaded=True,automatic_expanded_or_qualification=False,
        full_task_completed=False,hardware_readiness=False,hashes={str(f):prior.digest(f) for f in [*src.iterdir(),*frozen.iterdir(),local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE]}))
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
        parity=group(pool,p,[None,769000,773004],output/'zero_parity',True)
        nonzero=group(pool,probe,[None,769002,773004],output/'nonzero_smoke')
        exploration=[group(pool,p,[769000],output/'exploration_smoke'/f'replicate_{repeat}',seeds=[1810001+repeat]) for repeat in range(2)]
        assert exploration[0]['rows'][0]['initial_hash']==exploration[1]['rows'][0]['initial_hash']
        for repeat in range(2):
            with np.load(output/'exploration_smoke'/f'replicate_{repeat}'/'case_769000'/'trajectory.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['pg_sampled_latent'],z['pg_mean_latent']+SIGMA*z['pg_noise_amplitude'][:,None]*z['pg_innovation'])
        if not args.smoke:compare_smoke(output)
        local.write_json(output/'startup_closed.json',dict(parity=parity,nonzero=nonzero,exploration=exploration,independent_smoke_bitwise_equal=not args.smoke,terminal_result_saved=False,training_updates_saved=0))
        print('R181_ZERO_NOMINAL_STARTUP_PARITY_PASS',flush=True)
        if args.smoke:
            local.write_json(output/'results.json',dict(smoke=True,parity=parity,nonzero=nonzero,exploration=exploration,terminal_result_saved=True,full_task_completed=False,hardware_readiness=False));return
        import jax
        jax.config.update('jax_enable_x64',True)
        import jax.numpy as jp
        import optax
        from flax import serialization
        p={k:jp.asarray(v) for k,v in p.items()};optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(.001));state=optimizer.init(p)
        rng=np.random.default_rng(SEED);history=[];selected={k:np.asarray(v) for k,v in p.items()};best=None
        value_grad=jax.jit(jax.value_and_grad(score_loss))
        def checkpoint(number):
            dest=output/'checkpoints'/f'update_{number:04d}';dest.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(dest/'actor.npz',**{k:np.asarray(v) for k,v in p.items()})
            (dest/'learner.msgpack').write_bytes(serialization.to_bytes(dict(actor=p,optimizer=state)))
            local.write_json(dest/'rng.json',rng.bit_generator.state);local.write_json(dest/'history.json',history)
        checkpoint(0)
        baseline=group(pool,selected,CASES,output/'development_baseline',True);best=baseline
        for update in range(1,5):
            assert shutil.disk_usage(ROOT).free>10*1024**3,'Preserve evidence without cleanup'
            current={k:np.asarray(v) for k,v in p.items()};batch=output/'training'/f'update_{update:04d}';reports=[];features=[];latents=[];amplitudes=[];returns=[];seeds=[]
            for repeat in range(2):
                draw=[int(v) for v in rng.integers(1,2**62,size=len(CASES))];seeds.append(draw)
                report=group(pool,current,CASES,batch/f'replicate_{repeat}',seeds=draw);reports.append(report);returns.append([r['return_sum'] for r in report['rows']])
                for case in CASES:
                    with np.load(batch/f'replicate_{repeat}'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as z:
                        n=min(529,len(z['pg_features']));f=np.zeros((529,FEATURES));l=np.zeros((529,14));a=np.zeros(529)
                        f[:n]=z['pg_features'][:n];l[:n]=z['pg_sampled_latent'][:n];a[:n]=z['pg_noise_amplitude'][:n]
                        expected=np.asarray(differentiable_means(p,jp.asarray(f)))
                        np.testing.assert_allclose(expected[:n],z['pg_mean_latent'][:n],atol=1e-12,rtol=1e-12)
                        features.append(f);latents.append(l);amplitudes.append(a)
            r=np.asarray(returns);advantages=np.concatenate([r[0]-r[1],r[1]-r[0]])
            loss,grad=value_grad(p,jp.asarray(features),jp.asarray(latents),jp.asarray(amplitudes),jp.asarray(advantages))
            assert np.isfinite(float(loss)) and all(np.isfinite(np.asarray(g)).all() for g in grad.values())
            change,state=optimizer.update(grad,state,p);previous=p;p=optax.apply_updates(p,change);check_actor({k:np.asarray(v) for k,v in p.items()})
            norms={k:float(np.linalg.norm(np.asarray(v))) for k,v in grad.items()};changes={k:float(np.max(np.abs(np.asarray(p[k])-np.asarray(previous[k])))) for k in p}
            report=group(pool,{k:np.asarray(v) for k,v in p.items()},CASES,output/'development_checkpoints'/f'update_{update:04d}')
            if local.rank(report)>local.rank(best):best=report;selected={k:np.asarray(v) for k,v in p.items()}
            history.append(dict(update=update,exploration_seeds=seeds,closed_training_reports=reports,return_pair_matrix=r.tolist(),companion_advantages=advantages.tolist(),score_loss=float(loss),gradient_norms=norms,maximum_parameter_changes=changes,complete_development=report))
            checkpoint(update);local.write_json(output/'progress.json',dict(closed_update=update,updates=4,best=best,terminal_result_saved=False))
            print('R181_FULL_UPDATE',update,report['successes'],report['physical_failures'],norms,changes,flush=True)
        np.savez_compressed(output/'selected_actor.npz',**selected)
        local.write_json(output/'training_closed.json',dict(history=history,rng=rng.bit_generator.state,updates=4,training_attempts=200,whole_episode_original_returns=True))
        candidate=group(pool,selected,CASES,output/'development_candidate')
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
        local.write_json(output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,original_development_gate=gate,terminal_result_saved=True,formal_dynamic_attempts=358,expanded_development_run=False,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
        print('R181_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)


if __name__=='__main__':main()
