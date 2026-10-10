"""R113 initial-sensor-conditioned program distillation, simulation only.

The model predicts two program flags and six bounded residual nodes. Frozen
R102 feedback reads actual current sensors. Case identity and teacher recipe
are supervised labels only, never inference inputs or a lookup table.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics.getup_reference_env_r100 import ReferenceEpisode,TRAIN
from diagnostics.probe_getup_anchor_r101 import ROOT,SCENE,STAND,REFERENCE,MODEL
from diagnostics.train_getup_joint_anchor_r102 import action_for,write_json,aggregate
from diagnostics.getup_independent_native import digest

TEACHERS=ROOT/'github/openduck-mini-/results/getup_case_teachers_r109_left_20261005/verified_teacher_snapshot_0001'
NODES=np.array([0,50,100,150,200,300,400,529])


def checked_context(context):
    x=np.asarray(context,dtype=np.float32)
    if x.shape!=(55,) or not np.isfinite(x).all():raise ValueError('One finite actual 55-dimensional initial observation required')
    return x


def normalize(x,mean,std):
    if not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<1e-4):
        raise ValueError('Invalid training-only normalization')
    return np.clip((np.asarray(x,dtype=np.float32)-mean)/std,-10.,10.).astype(np.float32)


def scalar_predict(weights,context):
    x=normalize(checked_context(context),weights['context_mean'],weights['context_std'])
    for name in ('hidden0','hidden1'):x=np.tanh(x@weights[name+'_kernel']+weights[name+'_bias'])
    logits=x@weights['flags_kernel']+weights['flags_bias']
    knots=np.tanh(x@weights['knots_kernel']+weights['knots_bias']).reshape(6,10)
    if not np.isfinite(logits).all() or not np.isfinite(knots).all():raise ValueError('Nonfinite program')
    return logits,knots


def predict_program(weights,context):
    logits,raw=scalar_predict(weights,context)
    _,nominal=scalar_predict(weights,weights['nominal_context'])
    # Binary heads are fixed learned functions of sensors, not case branches.
    profile=int(logits[0]>=0.);has_knots=bool(logits[1]>=0.)
    knots=np.clip(raw-nominal,-1.,1.) if has_knots else np.zeros((6,10),dtype=np.float32)
    return profile,knots,dict(logits=logits.tolist(),has_knots=has_knots)


def interpolate_nodes(knots,control):
    x=np.asarray(knots,dtype=np.float64)
    if x.shape!=(6,10) or not np.isfinite(x).all() or np.max(np.abs(x))>1.:raise ValueError('Six bounded ten-joint nodes required')
    if not 0<=control<2279:raise ValueError('Control outside original path')
    if control>=529:return np.zeros(10)
    values=np.vstack((np.zeros((1,10)),x,np.zeros((1,10))))
    return np.array([np.interp(control,NODES,values[:,j]) for j in range(10)])


def execute_program(weights,current,control,profile,knots):
    obs=checked_context(current)
    if profile not in (0,1):raise ValueError('Only zero or original R102 feedback allowed')
    planned=interpolate_nodes(knots,control)
    if control>=529:return np.zeros(10)
    frozen={name:weights['feedback_'+name] for name in ('hidden0_kernel','hidden0_bias','hidden1_kernel','hidden1_bias','mean_kernel','mean_bias')}
    base=np.zeros(10) if profile==0 else action_for(frozen,obs,weights['feedback_anchors'][control],weights['feedback_gains'])
    return np.clip(base+planned,-1.,1.)


def load_program_data():
    contexts=[];flags=[];nodes=[];records=[]
    for case in (None,*TRAIN):
        folder=TEACHERS/f'case_{case}';row=json.loads((folder/'result.json').read_text())
        assert row['case_seed']==case and row['qualified_teacher'] and row['success'] and row['valid'] and row['controls']==2279
        assert row['strict_tail_s']>=30.-1e-8 and row['entry_time_s']<=12.
        with np.load(folder/'trajectory.npz',allow_pickle=False) as data:context=checked_context(data['observations'][0]).copy()
        with np.load(folder/'teacher_parameters.npz',allow_pickle=False) as data:profile=int(data['profile']);knots=data['knots'].copy()
        assert profile in (0,1) and knots.shape==(6,10) and np.isfinite(knots).all() and np.abs(knots).max()<=1.
        contexts.append(context);flags.append([profile,bool(np.any(knots!=0.))]);nodes.append(knots)
        records.append(dict(case_seed=case,initial_hash=row['initial_hash'],trajectory_sha256=digest(folder/'trajectory.npz'),
            parameter_sha256=digest(folder/'teacher_parameters.npz')))
    x=np.stack(contexts);y=np.array(flags,dtype=np.float32);knots=np.stack(nodes).astype(np.float32)
    assert y[0,0]==0 and y[0,1]==0 and np.array_equal(knots[0],np.zeros((6,10)))
    mean=x.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(x.std(0,dtype=np.float64),1e-4).astype(np.float32)
    with np.load(MODEL,allow_pickle=False) as data:frozen={'feedback_'+k:data[k].copy() for k in data.files}
    with np.load(TEACHERS/'frozen_profiles.npz',allow_pickle=False) as data:
        frozen.update(feedback_anchors=data['anchors'].copy(),feedback_gains=data['gains'].copy())
    frozen.update(context_mean=mean,context_std=std,nominal_context=x[0].copy())
    return x,y,knots,frozen,records


def fit(output,smoke):
    import jax
    import jax.numpy as jnp
    from flax import linen as nn,serialization
    import optax
    class Encoder(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.tanh(nn.Dense(64,name='hidden0')(x));x=nn.tanh(nn.Dense(32,name='hidden1')(x))
            return nn.Dense(2,name='flags')(x),nn.tanh(nn.Dense(60,name='knots')(x)).reshape(x.shape[:-1]+(6,10))
    x,flags,nodes,frozen,records=load_program_data();nx=normalize(x,frozen['context_mean'],frozen['context_std'])
    encoder=Encoder();key=jax.random.PRNGKey(213);key,initkey=jax.random.split(key)
    params=encoder.init(initkey,jnp.zeros(55))['params']
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-3));state=optimizer.init(params)
    @jax.jit
    def update(params,state):
        def objective(p):
            logits,raw=encoder.apply({'params':p},jnp.array(nx))
            anchored=jnp.clip(raw-raw[0],-1.,1.)
            classification=jnp.mean(optax.sigmoid_binary_cross_entropy(logits,jnp.array(flags)))
            regression=jnp.mean((anchored-jnp.array(nodes))**2)
            return classification+100.*regression
        loss,grads=jax.value_and_grad(objective)(params);delta,state=optimizer.update(grads,state,params)
        return optax.apply_updates(params,delta),state,loss
    def export():
        w={name+'_'+kind:np.array(params[name][kind]) for name in ('hidden0','hidden1','flags','knots') for kind in ('kernel','bias')}
        w.update(frozen);return w
    output.mkdir(exist_ok=False);np.savez_compressed(output/'initial_encoder.npz',**export())
    write_json(output/'data_manifest.json',dict(teachers=records,examples=25,context_dim=55,
        predicted_binary_flags=2,predicted_nodes_shape=[6,10],case_identifiers_labels_metadata_only=True,
        inference_contains_no_context_library=True,normalization_only_known_training_data=True))
    updates=20 if smoke else 5000;checkpoints=[20] if smoke else [500,1500,5000];history=[]
    for step in range(1,updates+1):
        params,state,loss=update(params,state);value=float(loss);assert np.isfinite(value)
        if step%50==0 or step in checkpoints:
            history.append(dict(update=step,full_batch_objective=value));write_json(output/'history.json',history)
        if step in checkpoints:
            w=export();scalar=np.stack([scalar_predict(w,c)[0] for c in x]);raw=np.stack([scalar_predict(w,c)[1] for c in x])
            batched,bnodes=encoder.apply({'params':params},jnp.array(nx))
            mismatch=max(float(np.max(np.abs(scalar-np.array(batched)))),float(np.max(np.abs(raw-np.array(bnodes)))))
            assert mismatch<5e-5
            np.testing.assert_array_equal(predict_program(w,x[0])[1],np.zeros((6,10)))
            predicted=[predict_program(w,c) for c in x];effective=np.stack([p[1] for p in predicted])
            file=output/f'update_{step:05d}.npz';np.savez_compressed(file,**w)
            (output/f'learner_{step:05d}.msgpack').write_bytes(serialization.to_bytes(dict(params=params,opt_state=state,key=key)))
            write_json(output/f'rng_{step:05d}.json',dict(jax_key=np.array(key).tolist(),full_batch_deterministic_no_sampling=True,seed=213))
            report=dict(update=step,profile_classification_correct=sum(p[0]==int(f[0]) for p,f in zip(predicted,flags)),
                knot_gate_correct=sum(p[2]['has_knots']==bool(f[1]) for p,f in zip(predicted,flags)),
                effective_node_mse=float(np.mean((effective-nodes)**2)),effective_node_max_error=float(np.abs(effective-nodes).max()),
                scalar_batch_max_difference=mismatch,nominal_nodes_exact_zero=True,model_sha256=digest(file),
                classification_and_parameter_fit_not_recovery_validation=True)
            write_json(output/f'report_{step:05d}.json',report);write_json(output/'progress.json',report)
            print('R113_CHECKPOINT',json.dumps(report),flush=True)
    write_json(output/'results.json',dict(completed_updates=updates,checkpoints=checkpoints,training_fit_only=True))
    return [str(output/f'update_{s:05d}.npz') for s in checkpoints]


def full_trial(job):
    model,case,directory,parity,override=job
    with np.load(model,allow_pickle=False) as data:w={k:data[k].copy() for k in data.files}
    env=ReferenceEpisode(str(SCENE),str(STAND),REFERENCE,213,qualification=True);env.reset(case)
    initial=env.observe().copy();profile,knots,info=predict_program(w,initial)
    if override is not None:
        profile=int(override['profile']);knots=np.asarray(override['knots']);info=dict(fixed_program_override=True)
    records=[]
    while True:
        obs=env.observe();k=env.controls;action=execute_program(w,obs,k,profile,knots)
        row=env.step(action,auto_reset=False)
        records.append((obs.copy(),action.copy(),env.sim.data.time,env.sim.data.qpos.copy(),env.sim.data.qvel.copy(),env.sim.prev.copy(),env.tail>0))
        if row[2]:break
    result=row[5];result.update(model=model,profile=profile,knots=knots.tolist(),predicted_program_info=info,
        actual_initial_context_only=True,teacher_identity_not_inference_input=True,
        success=bool(result['valid'] and result['controls']==2279 and result['entry_time_s'] is not None and result['entry_time_s']<=12. and result['strict_tail_s']>=30.-1e-8),
        full_path_trial=True,independent_qualification_trial=case in range(3160000,3160040),
        reference_states_injected=False,override_mode=('interface_teacher' if parity else 'fixed_zero_baseline') if override is not None else None)
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],initial_context=initial,
        time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records])
    if case is None:
        np.testing.assert_array_equal(arrays['normalized_residual'],np.zeros((len(records),10)));result['nominal_scalar_exact_zero']=True
    if parity:
        old_result=json.loads((Path(parity)/'result.json').read_text());assert result['initial_hash']==old_result['initial_hash']
        with np.load(Path(parity)/'trajectory.npz') as old:
            for name in ('qpos','qvel','applied','strict'):np.testing.assert_array_equal(np.array(arrays[name]),old[name])
        result['original_full_path_bitwise_parity']=True
    np.savez_compressed(dest/'trajectory.npz',**arrays);write_json(dest/'result.json',result)
    return result


def evaluate(pool,model,seeds,dest):
    jobs=[(model,s,str(dest/f'case_{s}'),str(TEACHERS/'case_None') if s is None else None,None) for s in seeds]
    rows=list(pool.map(full_trial,jobs));result=aggregate(rows);write_json(dest/'results.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_program_r113.py','launch_getup_program_r113.py','getup_reference_env_r100.py',
        'probe_getup_anchor_r101.py','train_getup_joint_anchor_r102.py','getup_independent_native.py','validate_getup_fullpath_r27.py',
        'train_getup_fullpath_r27.py','getup_fullfallen_env_r32.py','getup_fullfallen_contract_r32.py',
        'search_getup_reference_feedback_r64.py','search_getup_sustained_bridge_r42.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    write_json(args.output/'contract.json',dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        hypothesis='Predicting teacher program parameters may avoid near-zero stepwise imitation errors and retain exact fixed feedback where appropriate',
        seed=213,context_dim=55,current_sensor_dim=55,encoder_hidden_sizes=[64,32],predicted_binary_flags=2,predicted_nodes_shape=[6,10],
        learned_case_identifier_or_lookup_table=False,teacher_profile_and_nodes_supervision_only=True,
        initial_context_after_original_fall_settle=True,parameters_cached_without_root_state_writes=True,
        full_batch_examples=25,learning_rate=1e-3,gradient_norm_cap=1,updates=20 if args.smoke else 5000,
        binary_loss_weight=1,node_mse_weight=100,checkpoints=[20] if args.smoke else [500,1500,5000],
        normalization_only_existing_training_data=True,discrete_flags_from_learned_sensor_logits=True,
        control_hz=50,physics_hz=500,full_path_controls=2279,combined_correction_cap_rad=.18,
        original_feedback_and_home_hold_preserved=True,root_edits_after_initialization=0,physics_rewards_acceptance_unchanged=True,
        entry_deadline_s=12,strict_tail_s=30,development_gate=22,workers=6,smoke=args.smoke,
        independent_seeds_initially_unused=True,hashes={str(f):digest(f) for f in [SCENE,STAND,REFERENCE,MODEL,TEACHERS/'snapshot.json',TEACHERS/'frozen_profiles.npz',*sources.iterdir()]}))
    models=fit(args.output/'training',args.smoke)
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        if args.smoke:
            jobs=[]
            for s in (None,769000,769001,773004):
                with np.load(TEACHERS/f'case_{s}/teacher_parameters.npz') as data:override=dict(profile=int(data['profile']),knots=data['knots'].tolist())
                jobs.append((models[-1],s,str(args.output/'interface_teacher_parity'/f'case_{s}'),str(TEACHERS/f'case_{s}'),override))
            parity=list(pool.map(full_trial,jobs));write_json(args.output/'interface_teacher_parity.json',parity)
            nominal=evaluate(pool,models[-1],[None],args.output/'predicted_nominal_parity')
            nonstandard=evaluate(pool,models[-1],[769000],args.output/'predicted_nonstandard_interface')
            write_json(args.output/'results.json',dict(smoke=True,interface_teacher_parity=parity,nominal=nominal,
                nonstandard=nonstandard,interface_teacher_replays_not_unified_policy_performance=True,
                independent_qualification_run=False,full_task_completed=False))
        else:
            prior=ROOT/'outputs/getup_distill_r110_left_20261005/results.json';baseline=json.loads(prior.read_text())['baseline']
            assert baseline['successes']==12 and baseline['nominal_success'] and baseline['physical_failures']==0
            reports=[]
            for model in models:
                report=evaluate(pool,model,[None,*TRAIN],args.output/('development_'+Path(model).stem))
                assert all(c['initial_hash']==b['initial_hash'] for c,b in zip(report['rows'],baseline['rows']))
                report['model']=model;reports.append(report);write_json(args.output/'progress.json',dict(phase='full_development',candidates=reports))
                print('R113_FULL_DEVELOPMENT',Path(model).stem,report['successes'],report['physical_failures'],flush=True)
            eligible=[r for r in reports if r['nominal_success'] and r['successes']>=22 and r['physical_failures']==0]
            independent=None
            if eligible:
                best=max(eligible,key=lambda r:(r['successes'],r['return_sum']));frozen=args.output/'frozen_candidate.npz';shutil.copy2(best['model'],frozen)
                write_json(args.output/'frozen_candidate.json',dict(source=best['model'],sha256=digest(frozen),before_independent_evaluation=True))
                seeds=list(range(3160000,3160040));candidate=evaluate(pool,str(frozen),seeds,args.output/'qualification_candidate')
                # Zero program is a standard-model-independent fixed baseline.
                jobs=[(str(frozen),s,str(args.output/'qualification_baseline'/f'case_{s}'),None,dict(profile=0,knots=np.zeros((6,10)).tolist())) for s in seeds]
                rows=list(pool.map(full_trial,jobs));paired=aggregate(rows);write_json(args.output/'qualification_baseline/results.json',paired)
                assert all(c['initial_hash']==b['initial_hash'] for c,b in zip(candidate['rows'],paired['rows']))
                counts=[sum(r['success'] for r in candidate['rows'][i:i+20]) for i in (0,20)]
                independent=dict(candidate=candidate,baseline=paired,group_successes=counts,
                    left_stage_passed=min(counts)>=18 and candidate['physical_failures']==0)
            write_json(args.output/'results.json',dict(candidates=reports,baseline_reused_from_R110=baseline,baseline_source_sha256=digest(prior),
                independent=independent,independent_qualification_run=independent is not None,
                left_stage_passed=bool(independent and independent['left_stage_passed']),simulation_only=True,hardware_readiness=False,full_task_completed=False))
    print('R113_TERMINAL',flush=True)


if __name__=='__main__':main()
