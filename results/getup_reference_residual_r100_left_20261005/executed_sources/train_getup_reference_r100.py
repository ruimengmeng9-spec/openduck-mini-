"""R100 causal full-reference residual PPO, simulation only, fresh optimizer."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np

from diagnostics.getup_reference_env_r100 import GAMMA,CAP,TRAIN,pipe_worker,numpy_action
from diagnostics.train_getup_mixed_ppo_r98 import FullVector
from diagnostics.getup_fullfallen_contract_r32 import generalized_advantage
from diagnostics.getup_temporal_noise_r33 import sample_latent,conditional_log_prob,reset_noise
from diagnostics.getup_independent_native import digest


def export_actor(params,path):
    import onnx
    import onnxruntime as ort
    from onnx import helper,numpy_helper,TensorProto
    weights={name+'_'+key:np.asarray(params[name][key],np.float32)
             for name in ('hidden0','hidden1','mean') for key in ('kernel','bias')}
    np.savez_compressed(path.with_suffix('.npz'),**weights)
    nodes=[];initializers=[];previous='obs'
    for name in ('hidden0','hidden1','mean'):
        for key in ('kernel','bias'):
            initializers.append(numpy_helper.from_array(weights[name+'_'+key],name+'_'+key))
        nodes.extend([helper.make_node('MatMul',[previous,name+'_kernel'],[name+'_mul']),
                      helper.make_node('Add',[name+'_mul',name+'_bias'],[name+'_pre']),
                      helper.make_node('Tanh',[name+'_pre'],[name+'_out'])])
        previous=name+'_out'
    graph=helper.make_graph(nodes,'simulation_getup_r100_residual',
        [helper.make_tensor_value_info('obs',TensorProto.FLOAT,[None,55])],
        [helper.make_tensor_value_info(previous,TensorProto.FLOAT,[None,10])],initializers)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',13)])
    model.ir_version=10
    helper.set_model_props(model,{'simulation_only':'true','not_a_walking_actor':'true',
        'decoder':'combined legacy feedback + 0.18 * output clipped to +/-0.18rad',
        'requires_frozen_reference':'getup_path_audit_r99_20261005/frozen_path.npz'})
    onnx.checker.check_model(model);onnx.save(model,str(path.with_suffix('.onnx')))
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(path.with_suffix('.onnx')),sess_options=opts,
                                providers=['CPUExecutionProvider'])
    obs=np.random.default_rng(100).normal(size=(64,55)).astype(np.float32)
    np.testing.assert_allclose(session.run(None,{'obs':obs})[0],numpy_action(weights,obs),
                               atol=2e-6,rtol=2e-5)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=64)
    p.add_argument('--horizon',type=int,default=256)
    p.add_argument('--envs',type=int,default=8)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=200)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    args=p.parse_args()
    if args.iterations<1 or args.horizon<1 or not 0<args.learning_rate<=1e-4:p.error('Invalid budget')
    root=Path('/data/shijinsheng/open_duck')
    scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference=root/'outputs/getup_path_audit_r99_20261005/frozen_path.npz'
    args.output.mkdir(parents=True,exist_ok=False)
    names=(Path(__file__).name,'getup_reference_env_r100.py','test_getup_reference_r100.py',
           'getup_temporal_noise_r33.py','getup_fullfallen_contract_r32.py',
           'getup_fullfallen_env_r32.py','train_getup_mixed_ppo_r98.py','train_getup_native_ppo.py',
           'search_getup_reference_feedback_r64.py','search_getup_sustained_bridge_r42.py',
           'getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py')
    sources=args.output/'executed_sources';sources.mkdir()
    for name in names:shutil.copy2(Path(__file__).with_name(name),sources/name)
    contract=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        stage='left-side complete fall reference-residual discovery',seed=args.seed,
        iterations=args.iterations,horizon=args.horizon,envs=args.envs,workers=args.workers,
        learning_rate=args.learning_rate,optimizer_reinitialized=True,actor_initial_output_zero=True,
        observation_size=55,action_size=10,observation_layout=['original native 50D','causal IMU reference error 4D','reference recovery progress 1D'],
        action_cap_rad=CAP,combined_correction_cap_rad=CAP,original_home_phase_unchanged=True,
        train_seeds=list(TRAIN),nominal_probability=.1,prefix_library_used=False,
        episode_controls=629,motor_dt_s=.02,decision_dt_s=.02,physics_dt_s=.002,
        physical_limits_and_strict_acceptance_unchanged=True,root_edits_after_initialization=0,
        training_success_is_not_acceptance=True,final_hold_required_s=30.,entry_deadline_s=12.,
        gamma=GAMMA,gae_lambda=.95**.2,latent_std=.03,temporal_noise_rho=.9,
        hashes={name:digest(sources/name) for name in names},
        scene_sha256=digest(scene),standing_actor_sha256=digest(stand),reference_sha256=digest(reference))
    (args.output/'controller_contract.json').write_text(json.dumps(contract,indent=2))
    import jax
    import jax.numpy as jp
    import flax.linen as nn
    from flax import serialization
    import optax
    class Actor(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.tanh(nn.Dense(64,name='hidden0')(x));x=nn.tanh(nn.Dense(32,name='hidden1')(x))
            return nn.Dense(10,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros,name='mean')(x)
    class Critic(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.tanh(nn.Dense(64)(x));x=nn.tanh(nn.Dense(32)(x))
            return nn.Dense(1)(x)[...,0]
    actor,critic=Actor(),Critic();key=jax.random.PRNGKey(args.seed)
    key,a,b=jax.random.split(key,3)
    params=dict(actor=actor.init(a,jp.zeros((1,55)))['params'],critic=critic.init(b,jp.zeros((1,55)))['params'])
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(args.learning_rate))
    optstate=optimizer.init(params);logstd=jp.full((10,),np.log(.03))
    @jax.jit
    def sample(par,obs,prior,rng):
        mean=actor.apply({'params':par['actor']},obs)
        z,noise=sample_latent(mean,logstd,prior,jax.random.normal(rng,mean.shape),xp=jp)
        return jp.tanh(z),z,conditional_log_prob(z,mean,logstd,prior,xp=jp),critic.apply({'params':par['critic']},obs),noise
    @jax.jit
    def values(par,obs):return critic.apply({'params':par['critic']},obs)
    @jax.jit
    def update(par,state,obs,z,oldlog,adv,returns,prior):
        def loss(candidate):
            mean=actor.apply({'params':candidate['actor']},obs)
            logp=conditional_log_prob(z,mean,logstd,prior,xp=jp)
            ratio=jp.exp(logp-oldlog)
            policy=-jp.mean(jp.minimum(ratio*adv,jp.clip(ratio,.8,1.2)*adv))
            value=jp.mean((values(candidate,obs)-returns)**2)
            return policy+.5*value,jp.stack([policy,value,jp.mean(oldlog-logp)])
        (_,stats),grads=jax.value_and_grad(loss,has_aux=True)(par)
        updates,state=optimizer.update(grads,state,par)
        return optax.apply_updates(par,updates),state,stats
    rng=np.random.default_rng(args.seed);history=[];records=[];started=time.monotonic()
    def save(name):
        prefix=args.output/name
        prefix.with_suffix('.msgpack').write_bytes(serialization.to_bytes(params))
        prefix.with_suffix('.learner.msgpack').write_bytes(serialization.to_bytes(dict(params=params,optstate=optstate,key=key)))
        prefix.with_suffix('.rng.json').write_text(json.dumps(rng.bit_generator.state,indent=2))
        export_actor(params['actor'],prefix)
        (args.output/'episodes.json').write_text(json.dumps(records,indent=2))
    save('initial');vector=None
    print('R100_DEVICES',jax.devices(),flush=True)
    try:
        vector=FullVector(scene,stand,args.workers,args.envs,args.seed,pipe_worker,(str(reference),))
        previous=np.zeros((args.envs,10),np.float32)
        for iteration in range(1,args.iterations+1):
            observations=[];latents=[];logs=[];vals=[];priors=[];rewards=[];dones=[];terms=[];boots=[];episodes=[]
            for _ in range(args.horizon):
                obs=vector.obs.copy();prior=previous.copy();key,sub=jax.random.split(key)
                action,z,lp,v,noise=sample(params,jp.asarray(obs),jp.asarray(prior),sub)
                _,r,done,term,boot,info,_=vector.step(np.asarray(action))
                previous=np.asarray(reset_noise(np.asarray(noise),done),np.float32)
                for target,value in ((observations,obs),(latents,z),(logs,lp),(vals,v),(priors,prior),
                                     (rewards,r),(dones,done),(terms,term),(boots,boot)):target.append(np.asarray(value))
                episodes.extend(info)
            nextv=np.asarray(values(params,jp.asarray(np.asarray(boots).reshape(-1,55)))).reshape(args.horizon,args.envs)
            adv,ret=generalized_advantage(np.asarray(rewards),np.asarray(vals),nextv,np.asarray(dones),np.asarray(terms),gamma=GAMMA,lam=.95**.2)
            fa=adv.ravel();fa=(fa-fa.mean())/(fa.std()+1e-8)
            arrays=[np.asarray(observations).reshape(-1,55),np.asarray(latents).reshape(-1,10),
                    np.asarray(logs).ravel(),fa,ret.ravel(),np.asarray(priors).reshape(-1,10)]
            stats=[]
            for epoch in range(4):
                order=rng.permutation(len(fa))
                for offset in range(0,len(order),256):
                    ix=order[offset:offset+256]
                    params,optstate,s=update(params,optstate,*[jp.asarray(x[ix]) for x in arrays]);stats.append(np.asarray(s))
                if np.mean(stats,axis=0)[2]>.02:break
            s=np.mean(stats,axis=0)
            if not np.isfinite(s).all() or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
                raise RuntimeError('Nonfinite update; checkpoint refused')
            records.extend(episodes)
            row=dict(iteration=iteration,controls=iteration*args.horizon*args.envs,
                actual_full_fall_episodes=len(episodes),training_successes=sum(e['training_success'] for e in episodes),
                physical_failures=sum(not e['valid'] for e in episodes),
                mean_return=float(np.mean([e['return_sum'] for e in episodes])) if episodes else None,
                policy_loss=float(s[0]),value_loss=float(s[1]),approx_kl=float(s[2]),wall_seconds=time.monotonic()-started)
            history.append(row);(args.output/'training_history.json').write_text(json.dumps(history,indent=2))
            print(json.dumps(row),flush=True)
            if iteration%16==0 or iteration==args.iterations:save(f'checkpoint_{iteration:04d}')
        save('final')
        summary=dict(completed_iterations=args.iterations,controls=args.iterations*args.envs*args.horizon,
            actual_full_fall_episodes=len(records),training_successes=sum(e['training_success'] for e in records),
            full_task_completed=False,simulation_only=True,hardware_readiness=False,
            wall_seconds=time.monotonic()-started,model_sha256=digest(args.output/'final.onnx'))
        (args.output/'training_summary.json').write_text(json.dumps(summary,indent=2))
        print('R100_TRAINING_TERMINAL',json.dumps(summary),flush=True)
    finally:
        if vector is not None:vector.close()


if __name__=='__main__':main()
