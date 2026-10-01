"""R18 independent PPO: bounded home-posture residual, native 50 Hz motors.

No MJX surrogate, no walking-actor replacement, and no physical parameter
changes. Actor observes IMU/joint/previous-target/contact flags, not force or
root-height privileged data. This is NOT a full fallen-recovery task.
"""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np

from diagnostics.getup_macro_curriculum_r15 import pipe_worker,macro_advantage
from diagnostics.getup_independent_native import digest


class NativeVector:
    def __init__(self,scene,stand,starts,workers,envs,seed,tilt,repeat):
        if envs%workers:
            raise ValueError('env count must be divisible by workers')
        context=mp.get_context('spawn')
        self.connections=[]; self.processes=[]; self.chunk=envs//workers
        for rank in range(workers):
            parent,child=context.Pipe()
            process=context.Process(target=pipe_worker,
                args=(child,str(scene),str(stand),starts,self.chunk,seed+1000*rank,tilt,repeat))
            process.start(); child.close()
            self.connections.append(parent); self.processes.append(process)
        try:
            self.obs=np.concatenate([self.receive(c) for c in self.connections])
        except Exception:
            self.close()
            raise

    @staticmethod
    def receive(connection):
        if not connection.poll(120):
            raise RuntimeError('native worker exceeded 120-second response timeout')
        tag,payload=connection.recv()
        if tag=='error':
            raise RuntimeError(payload)
        return payload

    def step(self,actions):
        for i,c in enumerate(self.connections):
            c.send(('step',actions[i*self.chunk:(i+1)*self.chunk]))
        rows=[row for c in self.connections for row in self.receive(c)]
        self.obs=np.stack([r[0] for r in rows])
        return (self.obs,np.array([r[1] for r in rows],np.float32),
                np.array([r[2] for r in rows]),np.array([r[3] for r in rows]),
                np.stack([r[4] for r in rows]),[r[5] for r in rows if r[5] is not None],np.array([r[6] for r in rows]))

    def close(self):
        for c in self.connections:
            try: c.send(('close',None))
            except (BrokenPipeError,EOFError): pass
        for process in self.processes:
            process.join(timeout=10)
            if process.is_alive(): process.terminate(); process.join(timeout=5)
        for c in self.connections: c.close()


def generalized_advantage(rewards,values,next_values,done,terminated,gamma=.99,lam=.95):
    advantage=np.zeros_like(rewards,dtype=np.float32)
    running=np.zeros(rewards.shape[1],np.float32)
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*next_values[t]*(1-terminated[t])-values[t]
        running=delta+gamma*lam*(1-done[t])*running
        advantage[t]=running
    return advantage,advantage+values


def residual_decoder(latent,home,lower,upper,scale):
    midpoint=(np.asarray(lower)+np.asarray(upper))/2
    half_range=(np.asarray(upper)-np.asarray(lower))/2
    return np.clip((np.asarray(home)-midpoint+scale*np.tanh(latent))/half_range,-1.,1.)


def export_actor(params,path,home,lower,upper,scale):
    import onnx
    from onnx import helper,numpy_helper,TensorProto
    nodes=[]; initializers=[]; previous='obs'
    for i,name in enumerate(('hidden0','hidden1','mean')):
        dense=params[name]
        weight=f'w{i}'; bias=f'b{i}'; linear=f'linear{i}'; added=f'added{i}'
        initializers.extend([numpy_helper.from_array(np.asarray(dense['kernel'],np.float32),weight),
                             numpy_helper.from_array(np.asarray(dense['bias'],np.float32),bias)])
        nodes.extend([helper.make_node('MatMul',[previous,weight],[linear]),
                      helper.make_node('Add',[linear,bias],[added])])
        previous='residual' if name=='mean' else f'activation{i}'
        nodes.append(helper.make_node('Tanh',[added],[previous]))
    half_range=(np.asarray(upper)-np.asarray(lower))/2
    offset=(np.asarray(home)-(np.asarray(upper)+np.asarray(lower))/2)/half_range
    initializers.extend([numpy_helper.from_array(np.asarray(scale/half_range,np.float32),'residual_gain'),
                         numpy_helper.from_array(np.asarray(offset,np.float32),'home_normalized'),
                         numpy_helper.from_array(np.array(-1.,np.float32),'minimum'),
                         numpy_helper.from_array(np.array(1.,np.float32),'maximum')])
    nodes.extend([helper.make_node('Mul',['residual','residual_gain'],['scaled_residual']),
                  helper.make_node('Add',['scaled_residual','home_normalized'],['unclamped_target']),
                  helper.make_node('Clip',['unclamped_target','minimum','maximum'],['normalized_targets'])])
    graph=helper.make_graph(nodes,'independent_getup_stage1',
        [helper.make_tensor_value_info('obs',TensorProto.FLOAT,[None,50])],
        [helper.make_tensor_value_info('normalized_targets',TensorProto.FLOAT,[None,14])],initializers)
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)],ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model,str(path))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--precheck',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=64)
    p.add_argument('--envs',type=int,default=8)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--horizon',type=int,default=128)
    p.add_argument('--tilt-max',type=float,default=.25)
    p.add_argument('--seed',type=int,default=418)
    p.add_argument('--repeat',type=int,default=5)
    p.add_argument('--initial-std',type=float,default=.30)
    p.add_argument('--residual-scale',type=float,default=.15)
    p.add_argument('--reward-scale',type=float,default=.01)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    args=p.parse_args()
    if not (0.0184 <= args.initial_std <= .74 and 0 < args.reward_scale <= 1 and 0 < args.learning_rate <= .001):
        raise ValueError('unsafe or invalid optimizer/exploration settings')
    if not (1<=args.repeat<=10): raise ValueError('invalid repeat')
    if not (0<args.residual_scale<=.3):raise ValueError('invalid residual bound')
    args.output.mkdir(parents=True,exist_ok=False)
    qualification=json.loads(args.precheck.read_text())
    qualified=qualification['qualified_low_starts']
    if not qualified:
        raise RuntimeError('no qualified loaded-foot low starts')
    # Restrict to the six lowest qualified postures for a clear stage boundary.
    starts=[r['target_rad'] for r in qualified[:6]]
    scene=Path(qualification['scene_path'])
    stand=args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    from diagnostics.getup_native_curriculum import CurriculumSim
    sim=CurriculumSim(scene,stand)
    contract=dict(stage='loaded low posture / mild tilt -> home standing; not full fallen recovery',
                  observation_size=50,action_size=14,
                  observation_layout=['gyro:3','upvector:3','joint_delta_home:14','joint_velocity_x0.05:14',
                                      'previous_motor_target_delta_home:14','foot_contact_flags:2'],
                  action_decoder='home + residual_scale_rad*tanh(latent), normalized to original joint limits; physical clamp and 5.24rad/s slew',
                  residual_scale_rad=args.residual_scale,zero_latent_equals_home=True,
                  lower_rad=sim.lower.tolist(),upper_rad=sim.upper.tolist(),home_rad=sim.home.tolist(),
                  starts_rad=starts,tilt_max_rad=args.tilt_max,scene_path=str(scene),
                  simulation_only=True,hardware_readiness=False,seed=args.seed,
                  iterations=args.iterations,num_envs=args.envs,horizon=args.horizon,
                  policy_action_repeat=args.repeat,policy_period_s=.02*args.repeat,motor_target_period_s=.02,
                  gamma_per_macro=.99,gamma_per_control_step=.99**(1/args.repeat),gae_lambda_per_macro=.95,
                  initial_std=args.initial_std,reward_scale=args.reward_scale,learning_rate=args.learning_rate,
                  independent_actor_critic_gradient_clipping=True,optimizer_state_resumed=False,
                  resume_params=None,resume_sha256=None,
                  hashes={str(f):digest(f) for f in (Path(__file__),Path(__file__).with_name('getup_native_curriculum.py'),args.precheck,scene,stand,Path(__file__).with_name('getup_macro_curriculum_r15.py'))})
    (args.output/'controller_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    del sim
    import jax
    import jax.numpy as jp
    import flax.linen as nn
    from flax import serialization
    import optax
    print('POLICY DEVICES:',jax.devices(),flush=True)
    class Actor(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.tanh(nn.Dense(128,name='hidden0')(x))
            x=nn.tanh(nn.Dense(64,name='hidden1')(x))
            mean=nn.Dense(14,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros,name='mean')(x)
            std=self.param('log_std',nn.initializers.constant(np.log(args.initial_std)),(14,))
            return mean,jp.clip(std,-4.,-.3)
    class Critic(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.tanh(nn.Dense(128)(x)); x=nn.tanh(nn.Dense(64)(x))
            return nn.Dense(1)(x)[...,0]
    actor,critic=Actor(),Critic()
    key=jax.random.PRNGKey(args.seed); key,a,b=jax.random.split(key,3)
    params=dict(actor=actor.init(a,jp.zeros((1,50)))['params'],critic=critic.init(b,jp.zeros((1,50)))['params'])
    home=np.array(contract['home_rad']); low=np.array(contract['lower_rad']); high=np.array(contract['upper_rad'])
    # A fresh zero-mean residual is exactly the home baseline, not an R15 resume.
    params['actor']['mean']['bias']=jp.zeros((14,),dtype=jp.float32)
    home_action=jp.asarray((home-(low+high)/2)/((high-low)/2),dtype=jp.float32)
    residual_gain=jp.asarray(args.residual_scale/((high-low)/2),dtype=jp.float32)
    labels={name:jax.tree.map(lambda _:name,subtree) for name,subtree in params.items()}
    transforms={name:optax.chain(optax.clip_by_global_norm(1.),optax.adam(args.learning_rate))
                for name in ('actor','critic')}
    optimizer=optax.multi_transform(transforms,labels)
    optstate=optimizer.init(params)
    export_actor(params['actor'],args.output/'initial.onnx',home,low,high,args.residual_scale)
    (args.output/'initial.msgpack').write_bytes(serialization.to_bytes(params))
    def log_prob(z,mean,logstd):
        return jp.sum(-.5*((z-mean)/jp.exp(logstd))**2-logstd-.5*np.log(2*np.pi),axis=-1)
    @jax.jit
    def sample(par,obs,rng):
        mean,logstd=actor.apply({'params':par['actor']},obs)
        z=mean+jp.exp(logstd)*jax.random.normal(rng,mean.shape)
        action=jp.clip(home_action+residual_gain*jp.tanh(z),-1.,1.)
        return action,z,log_prob(z,mean,logstd),critic.apply({'params':par['critic']},obs)
    @jax.jit
    def values(par,obs):
        return critic.apply({'params':par['critic']},obs)
    @jax.jit
    def update(par,state,obs,z,oldlog,adv,returns):
        def loss(candidate):
            mean,logstd=actor.apply({'params':candidate['actor']},obs)
            logp=log_prob(z,mean,logstd)
            ratio=jp.exp(logp-oldlog)
            policy=-jp.mean(jp.minimum(ratio*adv,jp.clip(ratio,.8,1.2)*adv))
            value=jp.mean((critic.apply({'params':candidate['critic']},obs)-returns)**2)
            entropy=jp.sum(logstd+.5*np.log(2*np.pi*np.e))
            total=policy+.5*value-.001*entropy
            return total,jp.stack([policy,value,entropy,jp.mean(oldlog-logp)])
        (_,stats),grads=jax.value_and_grad(loss,has_aux=True)(par)
        updates,state=optimizer.update(grads,state,par)
        return optax.apply_updates(par,updates),state,stats
    vector=None; history=[]; started=time.time(); rng=np.random.default_rng(args.seed); actual_steps=0; decisions=0
    try:
        vector=NativeVector(scene,stand,starts,args.workers,args.envs,args.seed,args.tilt_max,args.repeat)
        for iteration in range(args.iterations):
            observations=[]; latents=[]; oldlogs=[]; oldvalues=[]; rewards=[]; dones=[]; terminated=[]; bootstrap=[]; episodes=[]; counts=[]
            for _ in range(args.horizon):
                obs=vector.obs.copy(); key,sub=jax.random.split(key)
                action,z,logp,v=sample(params,jp.asarray(obs),sub)
                nextobs,r,done,term,boot,info,n=vector.step(np.asarray(action))
                counts.append(n); actual_steps+=int(n.sum()); decisions+=len(n)
                observations.append(obs); latents.append(np.asarray(z)); oldlogs.append(np.asarray(logp)); oldvalues.append(np.asarray(v))
                rewards.append(r); dones.append(done); terminated.append(term); bootstrap.append(boot); episodes.extend(info)
            next_v=np.asarray(values(params,jp.asarray(np.asarray(bootstrap).reshape(-1,50)))).reshape(args.horizon,args.envs)
            adv,ret=macro_advantage(np.array(rewards)*args.reward_scale,np.array(oldvalues),next_v,np.array(dones),np.array(terminated),np.array(counts),args.repeat)
            flatadv=adv.reshape(-1); flatadv=(flatadv-flatadv.mean())/(flatadv.std()+1e-8)
            arrays=[np.asarray(observations).reshape(-1,50),np.asarray(latents).reshape(-1,14),
                    np.asarray(oldlogs).reshape(-1),flatadv,ret.reshape(-1)]
            stats=[]
            for epoch in range(4):
                order=rng.permutation(len(flatadv))
                for offset in range(0,len(order),256):
                    ix=order[offset:offset+256]
                    params,optstate,s=update(params,optstate,*[jp.asarray(x[ix]) for x in arrays])
                    stats.append(np.asarray(s))
            s=np.mean(stats,axis=0)
            if not np.isfinite(s).all() or not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(params)):
                raise RuntimeError('non-finite PPO update; refusing checkpoint')
            row=dict(iteration=iteration+1,environment_steps=actual_steps,policy_decisions=decisions,
                     completed_episodes=len(episodes),successful_episodes=sum(e['success'] for e in episodes),
                     mean_episode_return=float(np.mean([e['return_sum'] for e in episodes])) if episodes else None,
                     policy_loss=float(s[0]),value_loss=float(s[1]),gaussian_entropy=float(s[2]),approx_kl=float(s[3]),
                     wall_seconds=time.time()-started)
            history.append(row)
            (args.output/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
            if (iteration+1)%8==0 or iteration+1==args.iterations:
                (args.output/f'checkpoint_{iteration+1:04d}.msgpack').write_bytes(serialization.to_bytes(params))
        export_actor(params['actor'],args.output/'final.onnx',home,low,high,args.residual_scale)
        (args.output/'final.msgpack').write_bytes(serialization.to_bytes(params))
        summary=dict(completed_iterations=args.iterations,environment_steps=actual_steps,policy_decisions=decisions,policy_action_repeat=args.repeat,
                     wall_seconds=time.time()-started,simulation_only=True,hardware_readiness=False,
                     stage=contract['stage'],onnx_sha256=digest(args.output/'final.onnx'))
        (args.output/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('PPO COMPLETED:',json.dumps(summary),flush=True)
    finally:
        if vector is not None: vector.close()


if __name__=='__main__':
    main()
    # All artifacts are saved and own workers closed by main. Avoid a GPU-runtime
    # destructor hang after successful completion; OS teardown releases the context.
    import os,sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
