"""Full-fallen PPO with time-correlated, conditionally scored exploration.

Reuses the 50D native actor layout and mathematical PPO/export helpers, but
replaces the old easy reset/early fall termination with actual ground starts.
Physical audits and final independent recovery acceptance remain unchanged.
R32's independent decision noise is replaced by an AR(1) latent process;
stored previous noise is part of the exact PPO likelihood. This is a new
independent experiment; it never mutates a running R32 model or controller.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np

from diagnostics.getup_fullfallen_env_r32 import GAMMA, DECISION_CONTROLS, pipe_worker
from diagnostics.getup_independent_native import digest
from diagnostics.train_getup_native_ppo import NativeVector, export_actor as export_base_actor
from diagnostics.getup_fullfallen_contract_r32 import generalized_advantage
from diagnostics.getup_temporal_noise_r33 import (
    RHO, INNOVATION_SCALE, sample_latent, conditional_log_prob, reset_noise)


def export_actor(params,path):
    """Check exported deterministic targets against exact trained dense weights."""
    import onnx
    import onnxruntime as ort
    export_base_actor(params,path)
    model=onnx.load(str(path));model.graph.name='independent_fullfallen_r33_temporal'
    onnx.helper.set_model_props(model,dict(stage='actual_four_orientation_fullfallen_temporal_exploration',
        simulation_only='true',final_acceptance_required='80 independent trials and 30s loaded hold'))
    onnx.save(model,str(path))
    opts=ort.SessionOptions();opts.intra_op_num_threads=opts.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
    inputs=np.random.default_rng(9032).normal(size=(64,50)).astype(np.float32)
    target=inputs
    for name in ('hidden0','hidden1','mean'):
        target=np.tanh(target@np.asarray(params[name]['kernel'],np.float32)
                       +np.asarray(params[name]['bias'],np.float32))
    actual=session.run(None,{'obs':inputs})[0]
    np.testing.assert_allclose(actual,target,atol=2e-6,rtol=2e-5)


class FullVector(NativeVector):
    def __init__(self,scene,stand,workers,envs,seed):
        if envs%workers:
            raise ValueError('Environment count must be divisible by workers')
        context=mp.get_context('spawn')
        self.connections=[];self.processes=[];self.chunk=envs//workers
        try:
            for rank in range(workers):
                parent,child=context.Pipe()
                process=context.Process(target=pipe_worker,args=(child,str(scene),str(stand),self.chunk,seed+1000*rank))
                process.start();child.close()
                self.connections.append(parent);self.processes.append(process)
            self.obs=np.concatenate([self.receive(c) for c in self.connections])
        except BaseException:
            self.close()
            raise

    def step(self,actions):
        for i,c in enumerate(self.connections):
            c.send(('step',actions[i*self.chunk:(i+1)*self.chunk]))
        rows=[row for c in self.connections for row in self.receive(c)]
        self.obs=np.stack([r[0] for r in rows])
        return (self.obs,np.array([r[1] for r in rows],np.float32),
                np.array([r[2] for r in rows]),np.array([r[3] for r in rows]),
                np.stack([r[4] for r in rows]),[r[5] for r in rows if r[5] is not None],
                sum(r[6] for r in rows))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=256)
    p.add_argument('--envs',type=int,default=16)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--horizon',type=int,default=128)
    p.add_argument('--seed',type=int,default=1333)
    p.add_argument('--initialize-actor',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    scene=args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand=args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    qualification=json.loads((scene.parent/'qualification.json').read_text())
    if qualification['standing_passed_runs']!=qualification['standing_runs']:
        raise RuntimeError('Existing collision model lacks standing qualification')
    from diagnostics.validate_getup_fullpath_r27 import StrictSim
    sim=StrictSim(scene,stand)
    contract=dict(stage='R33 actual prone/supine/left/right fallen -> loaded strict home standing',
        simulation_only=True,hardware_readiness=False,default_controller_replaced=False,
        observation_size=50,action_size=14,root_edits_after_initialization=0,
        observation_layout=['gyro:3','framezaxis_upvector:3','joint_delta_home:14',
                            'joint_velocity_x0.05:14','previous_target_delta_home:14','foot_contact:2'],
        reset_probabilities=dict(standing=.2,prone=.2,supine=.2,left_side=.2,right_side=.2),
        reset_perturbations=dict(tilt_rad=.05,joint_rad=.02,velocity=.01),
        training_reset_seed_range=[0,999999],heldout_reset_seed_base=1300000,
        physics_dt_s=.002,motor_dt_s=.02,decision_dt_s=DECISION_CONTROLS*.02,
        repeated_motor_controls=DECISION_CONTROLS,substep_physical_audits=True,
        fall_orientation_is_not_a_termination=True,episode_controls=300,
        training_label='last continuous 1s strictly loaded; discovery only, NOT final completion',
        final_gate='each four fallen orientations >=18/20 independent perturbed starts, then continuous 30s strict standing',
        action_decoder='midpoint+tanh(latent)*half_range, physical clamp, original 5.24rad/s slew',
        lower_rad=sim.lower.tolist(),upper_rad=sim.upper.tolist(),home_rad=sim.home.tolist(),
        reward='potential difference + qualified-standing bonus; no extra motion penalty during discovery',
        optimizer_reinitialized=True,critic_reinitialized=True,initial_latent_std=.5,
        exploration=dict(kind='AR1 latent Gaussian; prior noise included in PPO conditional density',
                         rho=RHO,innovation_scale=INNOVATION_SCALE,
                         reset_noise_on_each_done=True,deterministic_deployment_noise_zero=True),
        seed=args.seed,num_envs=args.envs,horizon=args.horizon,iterations=args.iterations,
        initialized_actor=str(args.initialize_actor) if args.initialize_actor else None,
        official_method_reference='https://github.com/pollen-robotics/microduck_rl/blob/main/src/mjlab_microduck/tasks/microduck_standup_env_cfg.py',
        copied_microduck_mechanical_parameters=False,
        hashes={str(f):digest(f) for f in (Path(__file__),Path(__file__).with_name('getup_temporal_noise_r33.py'),
                                          Path(__file__).with_name('getup_fullfallen_env_r32.py'),
                                          Path(__file__).with_name('getup_fullfallen_contract_r32.py'),
                                          Path(__file__).with_name('train_getup_fullpath_r27.py'),scene,stand)})
    if args.initialize_actor:
        contract['hashes'][str(args.initialize_actor)]=digest(args.initialize_actor)
    (args.output/'controller_contract.json').write_text(json.dumps(contract,indent=2))
    source=args.output/'executed_sources';source.mkdir()
    for name in ('train_getup_temporal_ppo_r33.py','getup_temporal_noise_r33.py',
                 'getup_fullfallen_env_r32.py','train_getup_native_ppo.py',
                 'getup_fullfallen_contract_r32.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py'):
        shutil.copy2(Path(__file__).with_name(name),source/name)
    del sim
    import jax
    import jax.numpy as jp
    import flax.linen as nn
    from flax import serialization
    import optax
    print('POLICY_DEVICES',jax.devices(),flush=True)
    class Actor(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.tanh(nn.Dense(128,name='hidden0')(x))
            x=nn.tanh(nn.Dense(64,name='hidden1')(x))
            mean=nn.Dense(14,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros,name='mean')(x)
            std=self.param('log_std',nn.initializers.constant(np.log(.5)),(14,))
            return mean,jp.clip(std,-2.,.1)
    class Critic(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.tanh(nn.Dense(128)(x));x=nn.tanh(nn.Dense(64)(x))
            return nn.Dense(1)(x)[...,0]
    actor,critic=Actor(),Critic()
    key=jax.random.PRNGKey(args.seed);key,a,b=jax.random.split(key,3)
    params=dict(actor=actor.init(a,jp.zeros((1,50)))['params'],critic=critic.init(b,jp.zeros((1,50)))['params'])
    if args.initialize_actor:
        old=serialization.from_bytes(params,args.initialize_actor.read_bytes())
        params['actor']=old['actor']
    else:
        low,high,home=(np.asarray(contract[k]) for k in ('lower_rad','upper_rad','home_rad'))
        normal=(home-(low+high)/2)/((high-low)/2)
        params['actor']['mean']['bias']=jp.asarray(np.arctanh(np.clip(normal,-.98,.98)),dtype=jp.float32)
    params['actor']['log_std']=jp.full((14,),np.log(.5),dtype=jp.float32)
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(1e-4))
    optstate=optimizer.init(params)
    export_actor(params['actor'],args.output/'initial.onnx')
    (args.output/'initial.msgpack').write_bytes(serialization.to_bytes(params))
    @jax.jit
    def sample(par,obs,previous_noise,rng):
        mean,std=actor.apply({'params':par['actor']},obs)
        z,noise=sample_latent(mean,std,previous_noise,jax.random.normal(rng,mean.shape),xp=jp)
        logp=conditional_log_prob(z,mean,std,previous_noise,xp=jp)
        return jp.tanh(z),z,logp,critic.apply({'params':par['critic']},obs),noise
    @jax.jit
    def values(par,obs):
        return critic.apply({'params':par['critic']},obs)
    @jax.jit
    def update(par,state,obs,z,oldlog,adv,returns,previous_noise):
        def loss(candidate):
            mean,std=actor.apply({'params':candidate['actor']},obs)
            logp=conditional_log_prob(z,mean,std,previous_noise,xp=jp)
            ratio=jp.exp(logp-oldlog)
            policy=-jp.mean(jp.minimum(ratio*adv,jp.clip(ratio,.8,1.2)*adv))
            value=jp.mean((critic.apply({'params':candidate['critic']},obs)-returns)**2)
            entropy=jp.sum(std+np.log(INNOVATION_SCALE)+.5*np.log(2*np.pi*np.e))
            return policy+.5*value-.003*entropy,jp.stack([policy,value,entropy,jp.mean(oldlog-logp)])
        (_,stats),grads=jax.value_and_grad(loss,has_aux=True)(par)
        updates,state=optimizer.update(grads,state,par)
        return optax.apply_updates(par,updates),state,stats
    vector=None;history=[];actual_controls=0;episode_records=[];rng=np.random.default_rng(args.seed);started=time.monotonic()
    try:
        vector=FullVector(scene,stand,args.workers,args.envs,args.seed)
        previous_noise=np.zeros((args.envs,14),dtype=np.float32)
        for iteration in range(args.iterations):
            obs_rows=[];latents=[];oldlogs=[];oldvalues=[];rewards=[];dones=[];terms=[];bootstrap=[];episodes=[];prior_noises=[]
            for _ in range(args.horizon):
                obs=vector.obs.copy();key,sub=jax.random.split(key)
                prior=previous_noise.copy()
                action,z,lp,v,new_noise=sample(params,jp.asarray(obs),jp.asarray(prior),sub)
                _,r,done,term,boot,info,controls=vector.step(np.asarray(action))
                previous_noise=np.asarray(reset_noise(np.asarray(new_noise),done),dtype=np.float32)
                actual_controls+=controls
                obs_rows.append(obs);latents.append(np.asarray(z));oldlogs.append(np.asarray(lp));oldvalues.append(np.asarray(v));prior_noises.append(prior)
                rewards.append(r);dones.append(done);terms.append(term);bootstrap.append(boot);episodes.extend(info)
            next_v=np.asarray(values(params,jp.asarray(np.asarray(bootstrap).reshape(-1,50)))).reshape(args.horizon,args.envs)
            adv,ret=generalized_advantage(np.array(rewards),np.array(oldvalues),next_v,np.array(dones),np.array(terms),gamma=GAMMA)
            fa=adv.ravel();fa=(fa-fa.mean())/(fa.std()+1e-8)
            arrays=[np.asarray(obs_rows).reshape(-1,50),np.asarray(latents).reshape(-1,14),
                    np.asarray(oldlogs).ravel(),fa,ret.ravel(),np.asarray(prior_noises).reshape(-1,14)]
            stats=[]
            for epoch in range(4):
                order=rng.permutation(len(fa))
                for offset in range(0,len(order),256):
                    ix=order[offset:offset+256]
                    params,optstate,s=update(params,optstate,*[jp.asarray(x[ix]) for x in arrays]);stats.append(np.asarray(s))
                if float(np.mean(np.asarray(stats),axis=0)[3])>.02:
                    break
            s=np.mean(stats,axis=0)
            if not np.isfinite(s).all() or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
                raise RuntimeError('Nonfinite update; refusing checkpoint')
            by_pose={pose:dict(episodes=sum(e['pose']==pose for e in episodes),
                discovery_successes=sum(e['pose']==pose and e['training_success'] for e in episodes),
                physical_failures=sum(e['pose']==pose and not e['valid'] for e in episodes))
                for pose in ('standing','prone','supine','left_side','right_side')}
            row=dict(iteration=iteration+1,decision_steps=(iteration+1)*args.envs*args.horizon,
                actual_motor_controls=actual_controls,episode_count=len(episodes),by_pose=by_pose,
                mean_return=float(np.mean([e['return_sum'] for e in episodes])) if episodes else None,
                policy_loss=float(s[0]),value_loss=float(s[1]),gaussian_entropy=float(s[2]),approx_kl=float(s[3]),
                wall_seconds=time.monotonic()-started)
            history.append(row);episode_records.extend(episodes)
            (args.output/'training_history.json').write_text(json.dumps(history,indent=2))
            print(json.dumps(row),flush=True)
            if (iteration+1)%16==0 or iteration+1==args.iterations:
                prefix=args.output/f'checkpoint_{iteration+1:04d}'
                prefix.with_suffix('.msgpack').write_bytes(serialization.to_bytes(params))
                export_actor(params['actor'],prefix.with_suffix('.onnx'))
                (args.output/'episodes.json').write_text(json.dumps(episode_records,indent=2))
        export_actor(params['actor'],args.output/'final.onnx')
        (args.output/'final.msgpack').write_bytes(serialization.to_bytes(params))
        summary=dict(completed_iterations=args.iterations,decision_steps=args.iterations*args.envs*args.horizon,
            actual_motor_controls=actual_controls,wall_seconds=time.monotonic()-started,
            full_task_completed=False,independent_30s_acceptance_not_yet_run=True,
            simulation_only=True,hardware_readiness=False,onnx_sha256=digest(args.output/'final.onnx'))
        (args.output/'training_summary.json').write_text(json.dumps(summary,indent=2))
        print('FULLFALLEN_PPO_TERMINAL',json.dumps(summary),flush=True)
    finally:
        if vector is not None:vector.close()


if __name__=='__main__':
    main()


