"""R110 causal-context teacher distillation, simulation only.

Case IDs are used solely to choose offline data and physical initializations.
The actor receives current sensors + episode-start sensors, never case IDs.
The original environment, complete path, physics and acceptance are unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np

from diagnostics.getup_reference_env_r100 import ReferenceEpisode, TRAIN
from diagnostics.probe_getup_anchor_r101 import ROOT, SCENE, STAND, REFERENCE
from diagnostics.getup_independent_native import digest
from diagnostics.train_getup_joint_anchor_r102 import write_json, aggregate

TEACHERS=ROOT/'github/openduck-mini-/results/getup_case_teachers_r109_left_20261005/verified_teacher_snapshot_0001'
RECOVERY=529


def causal_input(current, initial):
    current=np.asarray(current,dtype=np.float32);initial=np.asarray(initial,dtype=np.float32)
    if current.shape!=(55,) or initial.shape!=(55,) or not np.isfinite(current).all() or not np.isfinite(initial).all():
        raise ValueError('Only two finite 55-dimensional sensor observations are accepted')
    return np.concatenate((current,initial))


def normalize(x, mean, std):
    if not np.isfinite(x).all() or np.any(std<1e-4):raise ValueError('Invalid training-only normalization')
    return np.clip((np.asarray(x,dtype=np.float32)-mean)/std,-10.,10.).astype(np.float32)


def scalar_network(weights, x):
    x=normalize(x,weights['input_mean'],weights['input_std'])
    if x.shape!=(110,):raise ValueError('Scalar execution is required')
    for layer in ('hidden0','hidden1','mean'):
        x=np.tanh(x@weights[layer+'_kernel']+weights[layer+'_bias'])
    if not np.isfinite(x).all():raise ValueError('Nonfinite actor output')
    return x


def policy_action(weights, current, initial, control):
    if not 0<=control<2279:raise ValueError('Control index outside frozen path')
    if control>=RECOVERY:return np.zeros(10,dtype=np.float32)
    # Both sides use the identical scalar arithmetic, not a cached batch output.
    actual=scalar_network(weights,causal_input(current,initial))
    nominal=scalar_network(weights,causal_input(weights['nominal_observations'][control],weights['nominal_observations'][0]))
    return np.clip(actual-nominal,-1.,1.)


def load_data(folder=TEACHERS):
    expected=(None,*TRAIN);xs=[];ys=[];metadata=[];nominal=None
    for case in expected:
        path=folder/f'case_{case}';result=json.loads((path/'result.json').read_text())
        assert result['case_seed']==case and result['success'] and result['qualified_teacher'] and result['valid']
        assert result['controls']==2279 and result['entry_time_s']<=12. and result['strict_tail_s']>=30.-1e-8
        with np.load(path/'trajectory.npz',allow_pickle=False) as data:
            obs=data['observations'].astype(np.float32);actions=data['normalized_residual'].astype(np.float32)
        assert obs.shape==(2279,55) and actions.shape==(2279,10)
        assert np.isfinite(obs).all() and np.isfinite(actions).all() and np.max(np.abs(actions))<=1.+1e-6
        if case is None:
            nominal=obs.copy();np.testing.assert_array_equal(actions,np.zeros_like(actions))
        xs.append(np.concatenate((obs[:RECOVERY],np.repeat(obs[:1],RECOVERY,axis=0)),axis=1))
        ys.append(actions[:RECOVERY])
        metadata.append(dict(case_seed=case,initial_hash=result['initial_hash'],trajectory_sha256=digest(path/'trajectory.npz')))
    x=np.concatenate(xs);y=np.concatenate(ys)
    # All 25 cases have equal 529-row contribution; no home-hold domination.
    mean=x.mean(0,dtype=np.float64).astype(np.float32)
    std=np.maximum(x.std(0,dtype=np.float64),1e-4).astype(np.float32)
    anchors=np.concatenate((nominal[:RECOVERY],np.repeat(nominal[:1],RECOVERY,axis=0)),axis=1)
    return x,y,anchors,mean,std,nominal,metadata


def fit(output, updates, checkpoints):
    import jax
    import jax.numpy as jnp
    from flax import linen as nn, serialization
    import optax

    class Actor(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.tanh(nn.Dense(128,name='hidden0')(x))
            x=nn.tanh(nn.Dense(64,name='hidden1')(x))
            return nn.tanh(nn.Dense(10,name='mean',kernel_init=nn.initializers.zeros)(x))

    x,y,a,mean,std,nominal,metadata=load_data()
    nx=normalize(x,mean,std);na=normalize(a,mean,std)
    actor=Actor();key=jax.random.PRNGKey(210);key,initkey=jax.random.split(key)
    params=actor.init(initkey,jnp.zeros(110))['params']
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(3e-4))
    opt_state=optimizer.init(params);rng=np.random.default_rng(210)
    @jax.jit
    def update(params,state,bx,ba,by):
        def loss(p):
            pred=jnp.clip(actor.apply({'params':p},bx)-actor.apply({'params':p},ba),-1.,1.)
            return jnp.mean((pred-by)**2)
        value,grads=jax.value_and_grad(loss)(params)
        changes,state=optimizer.update(grads,state,params)
        return optax.apply_updates(params,changes),state,value

    def export(p):
        weights={layer+'_'+kind:np.array(p[layer][kind]) for layer in ('hidden0','hidden1','mean') for kind in ('kernel','bias')}
        weights.update(input_mean=mean,input_std=std,nominal_observations=nominal)
        return weights

    output.mkdir(exist_ok=False);history=[]
    np.savez_compressed(output/'zero_initial_actor.npz',**export(params))
    write_json(output/'data_manifest.json',dict(teachers=metadata,rows=len(x),input_dim=110,action_dim=10,
        recovery_controls=RECOVERY,normalization_only_teacher_data=True,independent_data_loaded=False,
        temporal_samples_not_independent_recovery_validation=True))
    for step in range(1,updates+1):
        indices=rng.integers(len(x),size=512);time_indices=indices%RECOVERY
        params,opt_state,loss=update(params,opt_state,nx[indices],na[time_indices],y[indices])
        value=float(loss)
        if not np.isfinite(value):raise ValueError('Nonfinite distillation loss')
        if step%50==0 or step in checkpoints:
            history.append(dict(update=step,batch_mse=value));write_json(output/'history.json',history)
        if step in checkpoints:
            weights=export(params)
            for k in range(RECOVERY):np.testing.assert_array_equal(policy_action(weights,nominal[k],nominal[0],k),np.zeros(10))
            scalar=np.stack([policy_action(weights,x[i,:55],x[i,55:],int(i%RECOVERY)) for i in range(len(x))])
            batch=np.clip(np.array(actor.apply({'params':params},nx))-np.tile(np.array(actor.apply({'params':params},na)),(25,1)),-1.,1.)
            mismatch=float(np.abs(scalar-batch).max());assert mismatch<2e-5
            model=output/f'update_{step:05d}.npz';np.savez_compressed(model,**weights)
            (output/f'learner_{step:05d}.msgpack').write_bytes(serialization.to_bytes(dict(params=params,opt_state=opt_state,key=key)))
            write_json(output/f'rng_{step:05d}.json',rng.bit_generator.state)
            report=dict(update=step,training_teacher_mse=float(np.mean((scalar-y)**2)),scalar_batch_max_difference=mismatch,
                nominal_scalar_exact_zero=True,model_sha256=digest(model),closed_loop_success_not_established=True)
            write_json(output/f'report_{step:05d}.json',report)
            write_json(output/'progress.json',report);print('R110_CHECKPOINT',json.dumps(report),flush=True)
    write_json(output/'results.json',dict(completed_updates=updates,checkpoints=checkpoints,training_fit_only=True))
    return [str(output/f'update_{s:05d}.npz') for s in checkpoints]


def full_rollout(job):
    model,case,directory,parity=job
    weights=None
    if model:
        with np.load(model,allow_pickle=False) as data:weights={k:data[k].copy() for k in data.files}
    env=ReferenceEpisode(str(SCENE),str(STAND),REFERENCE,210,qualification=True);env.reset(case)
    context=env.observe().copy();records=[];anchor_error=[]
    while True:
        obs=env.observe();k=env.controls
        action=np.zeros(10,dtype=np.float32) if weights is None else policy_action(weights,obs,context,k)
        if weights is not None:
            anchor_error.append(float(np.linalg.norm(normalize(causal_input(obs,context),weights['input_mean'],weights['input_std']))))
        row=env.step(action,auto_reset=False)
        records.append((obs.copy(),action.copy(),env.sim.data.time,env.sim.data.qpos.copy(),env.sim.data.qvel.copy(),env.sim.prev.copy(),env.tail>0))
        if row[2]:break
    result=row[5];result.update(model=model,initial_context_causal=True,
        success=bool(result['valid'] and result['controls']==2279 and result['entry_time_s'] is not None
            and result['entry_time_s']<=12. and result['strict_tail_s']>=30.-1e-8),
        reference_states_injected=False,qualification_trial=True)
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],initial_context=context,
        time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],
        applied=[r[5] for r in records],strict=[r[6] for r in records],normalized_input_norm=anchor_error)
    if case is None:
        np.testing.assert_array_equal(arrays['normalized_residual'],np.zeros((len(records),10)))
        result['nominal_scalar_exact_zero']=True
    if parity:
        expected=json.loads((Path(parity)/'result.json').read_text());assert result['initial_hash']==expected['initial_hash']
        with np.load(Path(parity)/'trajectory.npz') as old:
            for name in ('qpos','qvel','applied','strict'):np.testing.assert_array_equal(np.array(arrays[name]),old[name])
        result['original_full_path_bitwise_parity']=True
    np.savez_compressed(dest/'trajectory.npz',**arrays);write_json(dest/'result.json',result)
    return result


def evaluate(pool,model,seeds,dest,parity=False):
    jobs=[(model,s,str(dest/f'case_{s}'),str(TEACHERS/'case_None') if parity and s is None else None) for s in seeds]
    rows=list(pool.map(full_rollout,jobs));report=aggregate(rows)
    write_json(dest/'results.json',report);return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=6);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();assert 1<=args.workers<=6
    args.output.mkdir(exist_ok=False)
    sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_distill_r110.py','launch_getup_distill_r110.py',
        'getup_reference_env_r100.py','probe_getup_anchor_r101.py','train_getup_joint_anchor_r102.py',
        'getup_independent_native.py','validate_getup_fullpath_r27.py','train_getup_fullpath_r27.py',
        'getup_fullfallen_env_r32.py','getup_fullfallen_contract_r32.py','search_getup_reference_feedback_r64.py',
        'search_getup_sustained_bridge_r42.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    updates=10 if args.smoke else 2000;checkpoints=[10] if args.smoke else [250,500,1000,2000]
    write_json(args.output/'contract.json',dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        hypothesis='A new sensor-conditioned network distilled from all complete teachers may resolve frozen-feature feedback tradeoffs',
        teacher_data=str(TEACHERS),case_identifiers_not_actor_inputs=True,causal_initial_context=True,
        actor_input_dim=110,actor_output_dim=10,hidden_sizes=[128,64],updates=updates,checkpoints=checkpoints,
        seed=210,learning_rate=3e-4,batch_size=512,normalization_training_data_only=True,
        full_path_controls=2279,physics_hz=500,control_hz=50,combined_residual_cap_rad=.18,
        home_feedback_unchanged=True,root_edits_after_initialization=0,physics_rewards_acceptance_unchanged=True,
        entry_deadline_s=12,strict_standing_s=30,development_gate=22,qualification_seeds_initially_unused=True,
        workers=args.workers,smoke=args.smoke,hashes={str(f):digest(f) for f in [SCENE,STAND,REFERENCE,TEACHERS/'snapshot.json',*sources.iterdir()]}))
    models=fit(args.output/'training',updates,checkpoints)
    # JAX stays in the parent; spawned physics workers import only NumPy.
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        if args.smoke:
            zero=list(pool.map(full_rollout,[(str(args.output/'training/zero_initial_actor.npz'),s,
                str(args.output/'zero_actor_parity'/f'case_{s}'),str(TEACHERS/f'case_{s}')) for s in (None,769000)]))
            write_json(args.output/'zero_actor_parity.json',zero)
            nominal=evaluate(pool,models[-1],[None],args.output/'nominal_parity',parity=True)
            nonstandard=evaluate(pool,models[-1],[769000],args.output/'nonstandard_interface')
            write_json(args.output/'results.json',dict(smoke=True,zero_actor_parity=zero,nominal=nominal,nonstandard=nonstandard,
                full_task_completed=False,independent_qualification_run=False))
        else:
            baseline=evaluate(pool,None,[None,*TRAIN],args.output/'baseline',parity=True)
            assert baseline['nominal_success'] and baseline['successes']==12 and baseline['physical_failures']==0
            reports=[]
            for model in models:
                report=evaluate(pool,model,[None,*TRAIN],args.output/('development_'+Path(model).stem),parity=True)
                report['model']=model;reports.append(report)
                write_json(args.output/'progress.json',dict(phase='full_development',baseline=baseline,candidates=reports))
                print('R110_FULL_DEVELOPMENT',Path(model).stem,report['successes'],report['physical_failures'],flush=True)
                assert all(r['initial_hash']==b['initial_hash'] for r,b in zip(report['rows'],baseline['rows']))
            eligible=[r for r in reports if r['nominal_success'] and r['successes']>=22 and r['successes']>13 and r['physical_failures']==0]
            independent=None;frozen=None
            if eligible:
                best=max(eligible,key=lambda r:(r['successes'],r['return_sum']));frozen=args.output/'frozen_candidate.npz'
                shutil.copy2(best['model'],frozen)
                write_json(args.output/'frozen_candidate.json',dict(source=best['model'],sha256=digest(frozen),before_independent_evaluation=True))
                seeds=list(range(3160000,3160040))
                candidate=evaluate(pool,str(frozen),seeds,args.output/'qualification_candidate')
                paired=evaluate(pool,None,seeds,args.output/'qualification_baseline')
                assert all(c['initial_hash']==b['initial_hash'] for c,b in zip(candidate['rows'],paired['rows']))
                counts=[sum(r['success'] for r in candidate['rows'][i:i+20]) for i in (0,20)]
                independent=dict(candidate=candidate,baseline=paired,group_successes=counts,
                    left_stage_passed=all(n>=18 for n in counts) and candidate['physical_failures']==0)
            write_json(args.output/'results.json',dict(baseline=baseline,candidates=reports,eligible_models=[r['model'] for r in eligible],
                independent=independent,independent_qualification_run=independent is not None,
                left_stage_passed=bool(independent and independent['left_stage_passed']),full_task_completed=False,
                simulation_only=True,hardware_readiness=False))
    print('R110_TERMINAL',flush=True)


if __name__=='__main__':main()
