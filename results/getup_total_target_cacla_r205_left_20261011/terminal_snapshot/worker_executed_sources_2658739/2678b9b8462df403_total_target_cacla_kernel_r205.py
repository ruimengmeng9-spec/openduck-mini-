"""Finite CACLA variant: pure float64 math; no simulator, Q model, or policy gradient."""
import copy
import json
from pathlib import Path
import numpy as np

SEED=305
GAMMA=0.99**0.2
NOISE=1e-4
LR=1e-4
BATCH=512
STEPS=64
NAMES=('w1','b1','w2','b2','w3','b3')

def initial_weights(seed=SEED):
    rng=np.random.default_rng(seed)
    out={}
    for group,last in (('actor',10),('value',1)):
        dims=(101,32,16,last)
        for n in range(1,4):
            shape=(dims[n-1],dims[n])
            out[group+'_w'+str(n)]=(np.zeros(shape,dtype=np.float64) if group=='actor' and n==3
                else rng.normal(size=shape)/np.sqrt(float(dims[n-1])))
            out[group+'_b'+str(n)]=np.zeros(dims[n],dtype=np.float64)
    assert sum(a.size for a in out.values())==7771
    return out

def validate_weights(weights):
    for group,last in (('actor',10),('value',1)):
        dims=(101,32,16,last)
        for n in range(1,4):
            for suffix,shape in (('w',(dims[n-1],dims[n])),('b',(dims[n],))):
                a=weights[group+'_'+suffix+str(n)]
                assert a.dtype==np.float64 and a.shape==shape and np.isfinite(a).all()

def features(current,initial,k):
    current=np.asarray(current,dtype=np.float64);initial=np.asarray(initial,dtype=np.float64)
    assert current.shape==initial.shape==(50,) and np.isfinite(current).all() and np.isfinite(initial).all()
    assert isinstance(k,(int,np.integer)) and k>=0
    return np.concatenate((np.tanh(current),np.tanh(initial),np.array([min(k/528.0,1.0)])))

def forward(weights,x,group):
    x=np.asarray(x,dtype=np.float64)
    assert x.ndim==2 and x.shape[1]==101 and np.isfinite(x).all()
    h1=np.tanh(x@weights[group+'_w1']+weights[group+'_b1'])
    h2=np.tanh(h1@weights[group+'_w2']+weights[group+'_b2'])
    raw=h2@weights[group+'_w3']+weights[group+'_b3']
    out=np.tanh(raw) if group=='actor' else raw
    return out,(x,h1,h2,raw)

def request(weights,phi,rng=None):
    phi=np.asarray(phi,dtype=np.float64);assert phi.shape==(101,)
    mean,cache=forward(weights,phi[None,:],'actor')
    before=None if rng is None else copy.deepcopy(rng.bit_generator.state)
    noise=np.zeros(10,dtype=np.float64) if rng is None else NOISE*rng.normal(size=10)
    proposal=mean[0]+noise;box=np.clip(proposal,-1.0,1.0)
    return dict(rawmean=cache[3][0].copy(),mean=mean[0].copy(),noise=noise,proposal=proposal,boxrequest=box,
        rng_before=before,rng_after=None if rng is None else copy.deepcopy(rng.bit_generator.state))

def backward(weights,cache,derivative,group):
    x,h1,h2,raw=cache
    dz=derivative*(1.0-np.tanh(raw)**2) if group=='actor' else derivative
    g={group+'_w3':h2.T@dz,group+'_b3':dz.sum(axis=0)}
    d2=(dz@weights[group+'_w3'].T)*(1.0-h2*h2)
    g.update({group+'_w2':h1.T@d2,group+'_b2':d2.sum(axis=0)})
    d1=(d2@weights[group+'_w2'].T)*(1.0-h1*h1)
    g.update({group+'_w1':x.T@d1,group+'_b1':d1.sum(axis=0)})
    return g

def gradients(weights,phi,nextphi,actions,rewards,bootstrap,home_tail):
    n=len(phi);assert n>0 and actions.shape==(n,10)
    pred,vcache=forward(weights,phi,'value')
    nextvalue,_=forward(weights,nextphi,'value')
    y=rewards+GAMMA*(bootstrap*nextvalue[:,0]+home_tail)
    delta=y-pred[:,0];positive=delta>0.0
    mean,acache=forward(weights,phi,'actor')
    error=mean-actions
    vg=backward(weights,vcache,(-delta/n)[:,None],'value')
    ag=backward(weights,acache,error*positive[:,None]/n,'actor')
    return dict(y=y,pred=pred[:,0],nextvalue=nextvalue[:,0],delta=delta,positive=positive,
        mean=mean,rawmean=acache[3],action_error=error,value_loss=float(0.5*np.mean(delta*delta)),
        actor_loss=float(0.5*np.mean(np.sum(error*error*positive[:,None],axis=1))),
        gradients={**ag,**vg})

def clip_gradients(grads):
    raw_norm=float(np.sqrt(sum(float(np.sum(a*a)) for a in grads.values())))
    assert np.isfinite(raw_norm) and all(np.isfinite(a).all() for a in grads.values())
    scale=min(1.0,1.0/raw_norm) if raw_norm>0.0 else 1.0
    return {k:v*scale for k,v in grads.items()},raw_norm,scale

def episode_transitions(phi,actions,full_rewards,invalid):
    phi=np.asarray(phi,dtype=np.float64);actions=np.asarray(actions,dtype=np.float64)
    rewards=np.asarray(full_rewards,dtype=np.float64);n=len(rewards)
    assert phi.shape==(n,101) and actions.shape==(n,10) and n>=2
    assert np.isfinite(phi).all() and np.isfinite(actions).all() and np.isfinite(rewards).all()
    assert np.max(np.abs(actions))<=1.0
    assert invalid or n==2279
    controls=np.arange(1,min(529,n),dtype=np.int64)
    x=phi[controls].copy();nx=np.zeros_like(x)
    bootstrap=np.zeros(len(controls),dtype=np.float64);tail=np.zeros(len(controls),dtype=np.float64)
    for i,k in enumerate(controls):
        if k<528 and k+1<n:
            nx[i]=phi[k+1];bootstrap[i]=1.0
        if k==528 and n>529:
            tail[i]=np.sum(rewards[529:]*GAMMA**np.arange(n-529,dtype=np.float64))
    return dict(phi=x,nextphi=nx,actions=actions[controls].copy(),rewards=rewards[controls].copy(),
        bootstrap=bootstrap,home_tail=tail,controls=controls,
        terminal=(bootstrap==0.0),full_rewards=rewards.copy(),invalid=bool(invalid))

def merge_episodes(episodes):
    assert episodes
    names=('phi','nextphi','actions','rewards','bootstrap','home_tail','controls','terminal')
    out={k:np.concatenate([e[k] for e in episodes]) for k in names}
    out['episode_offsets']=np.cumsum([0]+[len(e['controls']) for e in episodes],dtype=np.int64)
    out['full_reward_offsets']=np.cumsum([0]+[len(e['full_rewards']) for e in episodes],dtype=np.int64)
    out['full_rewards']=np.concatenate([e['full_rewards'] for e in episodes])
    out['invalid']=np.array([e['invalid'] for e in episodes],dtype=bool)
    return out

def json_new(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def npz_new(path,arrays):
    path=Path(path);assert not path.exists()
    with path.open('xb') as f:np.savez_compressed(f,**arrays)

class Learner:
    def __init__(self,weights=None,learning_seed=305001):
        self.weights={k:v.copy() for k,v in (initial_weights() if weights is None else weights).items()}
        validate_weights(self.weights)
        self.actor_count=0;self.value_count=0;self.round_count=0
        self.rng=np.random.default_rng(learning_seed)
    def state(self):
        return dict(actor_count=self.actor_count,value_count=self.value_count,round_count=self.round_count,
            learning_rng=copy.deepcopy(self.rng.bit_generator.state),adam_or_target_network=False)
    def checkpoint(self,stem):
        stem=Path(stem);npz_new(str(stem)+'.npz',self.weights);json_new(str(stem)+'.json',self.state())
    def update(self,replay):
        assert self.round_count<2 and self.value_count<128 and len(replay['phi'])>0
        before={k:v.copy() for k,v in self.weights.items()};state_before=self.state()
        indices=self.rng.integers(0,len(replay['phi']),size=BATCH,dtype=np.int64)
        batch={k:replay[k][indices] for k in ('phi','nextphi','actions','rewards','bootstrap','home_tail')}
        trace=gradients(self.weights,**batch)
        raw=trace.pop('gradients');clipped={};norm={};scale={}
        for group in ('actor','value'):
            cg,n,s=clip_gradients({k:v for k,v in raw.items() if k.startswith(group+'_')})
            clipped.update(cg);norm[group]=n;scale[group]=s
        actor_updated=bool(np.any(trace['positive']))
        for name in self.weights:
            if name.startswith('value_') or actor_updated:self.weights[name]=self.weights[name]-LR*clipped[name]
        self.value_count+=1;self.actor_count+=int(actor_updated)
        validate_weights(self.weights)
        return dict(**trace,indices=indices,raw_gradients=raw,clipped_gradients=clipped,norm=norm,scale=scale,
            before=before,after={k:v.copy() for k,v in self.weights.items()},state_before=state_before,
            state_after=self.state(),actor_updated=actor_updated)
    def train_round(self,replay,dest=None):
        assert self.round_count<2 and self.value_count==self.round_count*STEPS
        if dest is not None:
            dest=Path(dest);assert not dest.exists();dest.mkdir()
            npz_new(dest/'replay.npz',replay)
            self.checkpoint(dest/'before')
        traces=[]
        for step in range(STEPS):
            t=self.update(replay);traces.append(t)
            if dest is not None:
                arrays={k:t[k] for k in ('indices','y','pred','nextvalue','delta','positive','mean','rawmean','action_error')}
                for prefix in ('before','after','raw_gradients','clipped_gradients'):
                    arrays.update({prefix+'__'+k:v for k,v in t[prefix].items()})
                npz_new(dest/('update_%03d.npz'%step),arrays)
                json_new(dest/('update_%03d.json'%step),{k:t[k] for k in
                    ('value_loss','actor_loss','norm','scale','state_before','state_after','actor_updated')})
        self.round_count+=1
        if dest is not None:self.checkpoint(dest/'after')
        return traces

