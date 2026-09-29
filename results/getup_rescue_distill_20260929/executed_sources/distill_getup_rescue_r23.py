"""BC of verified R22 early-rescue trajectories plus successful home starts.

Small, state-conditioned, 50 Hz research actor. Whole-episode holdout is used
for offline checkpoint selection; only separate native closed-loop evaluation
can establish effectiveness. No walking-controller replacement.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from diagnostics import search_getup_rescue_r22 as rescue
from diagnostics.getup_independent_native import digest
from diagnostics.train_getup_residual_r18 import export_actor


def split_episodes(seeds):
    unique = np.unique(seeds)
    if len(unique) < 5:
        raise RuntimeError('too few independent demonstration episodes')
    order = np.random.default_rng(423).permutation(unique)
    valid = order[:max(1, len(order)//5)]
    mask = np.isin(seeds, valid)
    if set(seeds[mask]) & set(seeds[~mask]):
        raise RuntimeError('episode leakage')
    return ~mask, mask


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--search',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=200)
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    c=json.loads(args.contract.read_text())
    rescue.init_worker(str(args.contract),.55)
    obs=[]; targets=[]; seeds=[]; elapsed=[]; provenance=[]; controls=0
    for folder in args.search:
        report=json.loads((folder/'results.json').read_text())
        for r in report['results']:
            if not(r['selected']['success'] and r['selected_long']['success']):
                continue
            a=np.load(folder/f"seed_{r['seed']}_replay.npz",allow_pickle=False)
            obs.append(a['obs']);targets.append(a['targets']);elapsed.append(a['elapsed'])
            seeds.append(np.full(len(a['obs']),r['seed']))
            provenance.append(dict(seed=r['seed'],source='searched rescue',folder=str(folder)))
        for r in json.loads((folder/'baseline_scan.json').read_text()):
            if not r['success']:continue
            checked,a=rescue.replay(r['seed'],np.zeros((2,8)),record=True)
            controls+=checked['steps']
            if not checked['success'] or checked['initial_hash']!=r['initial_hash']:
                raise RuntimeError('home demonstration replay mismatch')
            o,t,_,_=map(np.asarray,zip(*a))
            obs.append(o);targets.append(t);elapsed.append(np.arange(len(o))*.02)
            seeds.append(np.full(len(o),r['seed']))
            provenance.append(dict(seed=r['seed'],source='successful home',folder=str(folder)))
    if not any(r['source']=='searched rescue' for r in provenance):
        raise RuntimeError('no verified rescue teacher; refusing distillation')
    x=np.concatenate(obs).astype(np.float32);y=np.concatenate(targets).astype(np.float32)
    seed=np.concatenate(seeds);t=np.concatenate(elapsed)
    train,valid=split_episodes(seed)
    home=np.asarray(c['home_rad'],np.float32);lower=np.asarray(c['lower_rad'],np.float32);upper=np.asarray(c['upper_rad'],np.float32)
    scale=1.0
    if np.max(np.abs(y-home))>scale+1e-6 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError('teacher not representable by declared actor')
    weights=np.where(t<1.2,4.,1.).astype(np.float32)
    np.savez_compressed(args.output/'demonstrations.npz',obs=x,targets=y,episode_seed=seed,
        elapsed=t,training_mask=train,validation_mask=valid,weight=weights)
    c.update(stage='R23 near-standing teacher BC; NOT full fallen get-up',
        seed=423,policy_action_repeat=1,policy_period_s=.02,residual_scale_rad=scale,
        action_decoder='home + 1.0*tanh(latent), original joint clamp and original slew',
        training_method='behavior cloning of original-physics executable teacher trajectories',
        independent_training_seed_list=np.unique(seed[train]).tolist(),
        offline_validation_seed_list=np.unique(seed[valid]).tolist(),
        default_controller_replaced=False,epochs=args.epochs,teacher_provenance=provenance)
    (args.output/'controller_contract.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    import jax
    import jax.numpy as jp
    import flax.linen as nn
    from flax import serialization
    import optax
    class Actor(nn.Module):
        @nn.compact
        def __call__(self,z):
            z=nn.tanh(nn.Dense(128,name='hidden0')(z))
            z=nn.tanh(nn.Dense(64,name='hidden1')(z))
            z=nn.Dense(14,kernel_init=nn.initializers.zeros,bias_init=nn.initializers.zeros,name='mean')(z)
            return jp.clip(jp.asarray(home)+scale*jp.tanh(z),jp.asarray(lower),jp.asarray(upper))
    actor=Actor();params=actor.init(jax.random.PRNGKey(423),jp.zeros((1,50)))['params']
    export_actor(params,args.output/'initial.onnx',home,lower,upper,scale)
    (args.output/'initial.msgpack').write_bytes(serialization.to_bytes(dict(actor=params)))
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(3e-4));state=opt.init(params)
    @jax.jit
    def loss(par,o,label,w):
        err=jp.mean((actor.apply({'params':par},o)-label)**2,axis=-1)
        return jp.sum(err*w)/jp.sum(w)
    @jax.jit
    def update(par,st,o,label,w):
        val,grad=jax.value_and_grad(loss)(par,o,label,w)
        updates,st=opt.update(grad,st,par)
        return optax.apply_updates(par,updates),st,val
    rng=np.random.default_rng(423);history=[];best=params;bestval=float('inf');bestepoch=0;updates=0
    started=time.time();index=np.flatnonzero(train)
    for epoch in range(args.epochs):
        order=rng.permutation(index)
        for start in range(0,len(order),256):
            ix=order[start:start+256]
            params,state,_=update(params,state,jp.asarray(x[ix]),jp.asarray(y[ix]),jp.asarray(weights[ix]));updates+=1
        if (epoch+1)%10==0 or epoch==0:
            tr=float(loss(params,x[train],y[train],weights[train]))
            va=float(loss(params,x[valid],y[valid],weights[valid]))
            if not np.isfinite([tr,va]).all():raise RuntimeError('nonfinite BC loss')
            if va<bestval:best=params;bestval=va;bestepoch=epoch+1
            row=dict(epoch=epoch+1,training_weighted_joint_mse_rad2=tr,
                validation_weighted_joint_mse_rad2=va,updates=updates,elapsed_s=time.time()-started)
            history.append(row);print('BC:',json.dumps(row),flush=True)
            (args.output/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    export_actor(best,args.output/'final.onnx',home,lower,upper,scale)
    (args.output/'final.msgpack').write_bytes(serialization.to_bytes(dict(actor=best)))
    summary=dict(epochs_completed=args.epochs,optimizer_updates=updates,selected_epoch=bestepoch,
        selection='whole-episode offline validation MSE only; NOT closed-loop recovery rate',
        verified_rescue_episodes=sum(r['source']=='searched rescue' for r in provenance),
        home_episodes=sum(r['source']=='successful home' for r in provenance),
        training_episodes=len(np.unique(seed[train])),validation_episodes=len(np.unique(seed[valid])),
        demonstration_samples=len(x),additional_native_demo_control_steps=controls,
        offline_validation_mse_rad2=bestval,training_environment_steps=0,
        simulation_only=True,hardware_readiness=False,default_controller_replaced=False,
        onnx_sha256=digest(args.output/'final.onnx'),
        source_hashes={str(f):digest(f) for f in (Path(__file__),args.contract,args.output/'demonstrations.npz')})
    (args.output/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('BC COMPLETE:',json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
    import os,sys
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
