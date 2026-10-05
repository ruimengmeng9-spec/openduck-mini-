"""One bounded offline DAgger round after read-only drift evidence R111.

Teachers are selected only to relabel stored training trajectories. The new
deployed-in-simulation actor remains the same case-ID-free causal function.
Off-trajectory teacher labels are bounded targets, not proven recoveries.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_distill_r110 as base
from diagnostics.train_getup_joint_anchor_r102 import write_json, aggregate
from diagnostics.getup_independent_native import digest

ROOT=base.ROOT;RUN=ROOT/'outputs/getup_distill_r110_left_20261005'
AUDIT=ROOT/'outputs/getup_distill_audit_r111_20261005'
WARM=RUN/'training/update_02000.npz'


def dataset():
    original,labels,anchors,mean,std,nominal,teachers=base.load_data()
    audited=json.loads((AUDIT/'results.json').read_text());assert not audited['independent_qualification_run']
    current=[];targets=[];times=[];manifest=[]
    for group in audited['groups']:
        checkpoint=Path(group['model']).stem
        assert digest(group['model'])==group['model_sha256']
        for row in group['rows']:
            case=row['case_seed'];assert case in base.TRAIN
            trace=RUN/('development_'+checkpoint)/f'case_{case}/trajectory.npz'
            relabeled=AUDIT/f'{checkpoint}_case_{case}.npz';assert digest(trace)==row['trajectory_sha256']
            with np.load(trace) as data:
                obs=data['observations'].copy();context=data['initial_context'].copy()
            with np.load(relabeled) as data:y=data['current_state_expert_labels'].copy()
            n=min(len(obs),base.RECOVERY);assert y.shape==(n,10)
            x=np.concatenate((obs[:n],np.repeat(context[None],n,axis=0)),axis=1).astype(np.float32)
            assert np.isfinite(x).all() and np.isfinite(y).all() and np.abs(y).max()<=1.+1e-6
            current.append(x);targets.append(y.astype(np.float32));times.append(np.arange(n))
            manifest.append(dict(checkpoint=checkpoint,case_seed=case,rows=n,source_sha256=digest(trace),
                relabel_sha256=digest(relabeled),labels_are_not_proven_off_trajectory_recoveries=True))
    x=np.concatenate(current);y=np.concatenate(targets);t=np.concatenate(times)
    return original,labels,anchors,mean,std,nominal,x,y,t,teachers,manifest


def fit(output,smoke):
    import jax
    import jax.numpy as jnp
    from flax import linen as nn,serialization
    import optax
    class Actor(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.tanh(nn.Dense(128,name='hidden0')(x));x=nn.tanh(nn.Dense(64,name='hidden1')(x))
            return nn.tanh(nn.Dense(10,name='mean')(x))
    original,labels,anchors,mean,std,nominal,x,y,t,teachers,manifest=dataset()
    with np.load(WARM) as data:w={k:data[k].copy() for k in data.files}
    for name,value in [('input_mean',mean),('input_std',std),('nominal_observations',nominal)]:np.testing.assert_array_equal(w[name],value)
    actor=Actor();params={layer:{kind:jnp.array(w[layer+'_'+kind]) for kind in ('kernel','bias')} for layer in ('hidden0','hidden1','mean')}
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-4));state=optimizer.init(params)
    rng=np.random.default_rng(212);key=jax.random.PRNGKey(212)
    n_original=base.normalize(original,mean,std);n_extra=base.normalize(x,mean,std);n_anchor=base.normalize(anchors,mean,std)
    @jax.jit
    def update(params,state,bx,ba,by):
        def loss(p):return jnp.mean((jnp.clip(actor.apply({'params':p},bx)-actor.apply({'params':p},ba),-1.,1.)-by)**2)
        loss_value,grads=jax.value_and_grad(loss)(params);change,state=optimizer.update(grads,state,params)
        return optax.apply_updates(params,change),state,loss_value
    def export():
        out={layer+'_'+kind:np.array(params[layer][kind]) for layer in ('hidden0','hidden1','mean') for kind in ('kernel','bias')}
        out.update(input_mean=mean,input_std=std,nominal_observations=nominal);return out
    output.mkdir(exist_ok=False);initial=export()
    for name in w:np.testing.assert_array_equal(initial[name],w[name])
    np.savez_compressed(output/'warm_actor.npz',**initial)
    write_json(output/'data_manifest.json',dict(original_teachers=teachers,relabeled_sources=manifest,
        original_rows=len(original),visited_rows=len(x),normalization_frozen_R110_training_only=True,
        actor_input_contains_no_teacher_identity=True,offline_case_teacher_selection_only=True,
        optimizer_and_rng_fresh=True,not_exact_resume=True,warm_model_sha256=digest(WARM)))
    updates=20 if smoke else 4000;checkpoints=[20] if smoke else [1000,2000,4000];history=[]
    for step in range(1,updates+1):
        oi=rng.integers(len(original),size=256);ei=rng.integers(len(x),size=256)
        bx=np.concatenate((n_original[oi],n_extra[ei]));ba=np.concatenate((n_anchor[oi%529],n_anchor[t[ei]]))
        by=np.concatenate((labels[oi],y[ei]));params,state,loss=update(params,state,bx,ba,by)
        value=float(loss);assert np.isfinite(value)
        if step%50==0 or step in checkpoints:
            history.append(dict(update=step,batch_mse=value));write_json(output/'history.json',history)
        if step in checkpoints:
            weights=export();model=output/f'update_{step:05d}.npz'
            for k in range(529):np.testing.assert_array_equal(base.policy_action(weights,nominal[k],nominal[0],k),np.zeros(10))
            np.savez_compressed(model,**weights)
            (output/f'learner_{step:05d}.msgpack').write_bytes(serialization.to_bytes(dict(params=params,opt_state=state,key=key)))
            write_json(output/f'rng_{step:05d}.json',rng.bit_generator.state)
            original_pred=np.stack([base.policy_action(weights,original[i,:55],original[i,55:],int(i%529)) for i in range(len(original))])
            visited_pred=np.stack([base.policy_action(weights,x[i,:55],x[i,55:],int(t[i])) for i in range(len(x))])
            report=dict(update=step,original_teacher_mse=float(np.mean((original_pred-labels)**2)),
                visited_teacher_mse=float(np.mean((visited_pred-y)**2)),nominal_scalar_exact_zero=True,
                model_sha256=digest(model),closed_loop_success_not_established=True)
            write_json(output/f'report_{step:05d}.json',report);write_json(output/'progress.json',report)
            print('R112_CHECKPOINT',json.dumps(report),flush=True)
    write_json(output/'results.json',dict(completed_updates=updates,checkpoints=checkpoints,one_offline_aggregation_round=True))
    return [str(output/f'update_{s:05d}.npz') for s in checkpoints]


def rollout(job):
    row=base.full_rollout(job)
    row.pop('qualification_trial',None)
    row.update(full_path_trial=True,independent_qualification_trial=row['case_seed'] in range(3160000,3160040))
    write_json(Path(job[2])/'result.json',row);return row


def evaluate(pool,model,seeds,dest,parity=False):
    rows=list(pool.map(rollout,[(model,s,str(dest/f'case_{s}'),str(base.TEACHERS/'case_None') if parity and s is None else None) for s in seeds]))
    result=aggregate(rows);write_json(dest/'results.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_aggregate_r112.py','launch_getup_aggregate_r112.py','train_getup_distill_r110.py',
        'audit_getup_distill_r111.py','getup_reference_env_r100.py','validate_getup_fullpath_r27.py','getup_independent_native.py',
        'getup_fullfallen_env_r32.py','getup_fullfallen_contract_r32.py','search_getup_reference_feedback_r64.py',
        'search_getup_sustained_bridge_r42.py','train_getup_fullpath_r27.py','train_getup_joint_anchor_r102.py','probe_getup_anchor_r101.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    write_json(args.output/'contract.json',dict(simulation_only=True,smoke=args.smoke,input_dim=110,output_dim=10,
        hypothesis='Offline current-state relabeling may mitigate demonstrated teacher-to-student distribution drift',
        fixed_seed=212,learning_rate=1e-4,updates=20 if args.smoke else 4000,minibatch_original_visited_ratio=[256,256],
        warm_start_actor_only=True,fresh_optimizer_and_rng=True,not_exact_checkpoint_resume=True,
        normalization_frozen_R110_training_only=True,teacher_identity_only_offline_relabeling=True,actor_case_ID_free=True,
        labels_not_proven_offtrajectory_recovery=True,no_new_oracle_policy_deployment=True,
        unchanged_physics_rewards_gates=True,root_edits_after_initialization=0,
        control_hz=50,physics_hz=500,full_path_controls=2279,combined_residual_cap_rad=.18,
        home_feedback_unchanged=True,entry_deadline_s=12,strict_tail_s=30,development_gate=22,
        independent_seeds_initially_unused=True,workers=6,hardware_readiness=False,full_task_completed=False,
        hashes={str(f):digest(f) for f in [WARM,AUDIT/'results.json',base.SCENE,base.STAND,base.REFERENCE,*sources.iterdir()]}))
    models=fit(args.output/'training',args.smoke)
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        if args.smoke:
            standard=evaluate(pool,models[-1],[None],args.output/'standard_parity',True)
            interface=evaluate(pool,models[-1],[769000],args.output/'nonstandard_interface')
            write_json(args.output/'results.json',dict(smoke=True,standard=standard,interface=interface,
                independent_qualification_run=False,full_task_completed=False))
        else:
            old=json.loads((RUN/'results.json').read_text());baseline=old['baseline']
            assert baseline['successes']==12 and baseline['nominal_success'] and baseline['physical_failures']==0
            reports=[]
            for model in models:
                r=evaluate(pool,model,[None,*base.TRAIN],args.output/('development_'+Path(model).stem),True)
                assert all(c['initial_hash']==b['initial_hash'] for c,b in zip(r['rows'],baseline['rows']))
                r['model']=model;reports.append(r);write_json(args.output/'progress.json',dict(phase='full_development',candidates=reports))
                print('R112_FULL_DEVELOPMENT',Path(model).stem,r['successes'],r['physical_failures'],flush=True)
            eligible=[r for r in reports if r['nominal_success'] and r['successes']>=22 and r['physical_failures']==0]
            independent=None
            if eligible:
                best=max(eligible,key=lambda r:(r['successes'],r['return_sum']));frozen=args.output/'frozen_candidate.npz';shutil.copy2(best['model'],frozen)
                write_json(args.output/'frozen_candidate.json',dict(source=best['model'],sha256=digest(frozen),before_independent_evaluation=True))
                seeds=list(range(3160000,3160040));candidate=evaluate(pool,str(frozen),seeds,args.output/'qualification_candidate')
                paired=evaluate(pool,None,seeds,args.output/'qualification_baseline')
                assert all(c['initial_hash']==b['initial_hash'] for c,b in zip(candidate['rows'],paired['rows']))
                counts=[sum(r['success'] for r in candidate['rows'][i:i+20]) for i in (0,20)]
                independent=dict(candidate=candidate,baseline=paired,group_successes=counts,
                    left_stage_passed=min(counts)>=18 and candidate['physical_failures']==0)
            write_json(args.output/'results.json',dict(candidates=reports,baseline_reused_from_R110=baseline,
                baseline_source_sha256=digest(RUN/'results.json'),independent=independent,
                independent_qualification_run=independent is not None,left_stage_passed=bool(independent and independent['left_stage_passed']),
                hardware_readiness=False,full_task_completed=False))
    print('R112_TERMINAL',flush=True)


if __name__=='__main__':main()
