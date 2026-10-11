"""Fresh finite model-free NAF Q kernel; causal scalar inputs, no simulator."""
import copy
import numpy as np
SEED=304
INPUTS=101
ACTIONS=10
OUTPUTS=66
BOUND=.18
GAMMA=.99**.2
HORIZON=529
SIGMA=1e-4
RATE=1e-4
UPDATES=64
BATCH=512
TAU=.005
DIAG_FLOOR=1e-3
ROWS,COLS=np.tril_indices(ACTIONS)
DIAG=ROWS==COLS
BASIS=np.zeros((55,10,10))
BASIS[np.arange(55),ROWS,COLS]=1.

def finite(value,shape):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite declared causal array required')
    return a

def features(current,initial,control):
    if not isinstance(control,(int,np.integer)) or not 0<=control<=2279:raise ValueError('Original control clock required')
    return np.r_[np.tanh(finite(current,(50,))),np.tanh(finite(initial,(50,))),min(control/528.,1.)]

def initial_weights():
    rng=np.random.default_rng(SEED);weights={}
    for i,(a,b) in enumerate([(INPUTS,32),(32,16),(16,OUTPUTS)]):
        weights[f'q_w{i}']=rng.normal(0,1/np.sqrt(a),(a,b))
        weights[f'q_b{i}']=np.zeros(b)
    weights['q_w2'][:,1:]=0.
    weights['q_b2'][11:]=np.where(DIAG,np.log(np.expm1(1.-DIAG_FLOOR)),0.)
    return weights

def network(weights,x,xp=np):
    for i in range(3):
        x=x@weights[f'q_w{i}']+weights[f'q_b{i}']
        if i<2:x=xp.tanh(x)
    return x

def heads(weights,x,xp=np):
    out=network(weights,x,xp);raw=out[...,1:11]
    triangle=out[...,11:]
    values=xp.where(xp.asarray(DIAG),xp.logaddexp(0.,triangle)+DIAG_FLOOR,triangle)
    lower=xp.einsum('...k,kij->...ij',values,xp.asarray(BASIS))
    return out[...,0],xp.tanh(raw),lower,raw

def q_value(weights,x,action,xp=np):
    value,mean,lower,_=heads(weights,x,xp)
    residual=xp.einsum('...i,...ij->...j',action-mean,lower)
    return value-.5*xp.sum(residual**2,axis=-1)

def request(weights,x,rng=None):
    value,mean,lower,raw=heads(weights,finite(x,(INPUTS,)))
    noise=np.zeros(ACTIONS) if rng is None else rng.normal(size=ACTIONS)
    proposal=mean+SIGMA*noise
    action=np.clip(proposal,-1.,1.)
    if not all(np.isfinite(v).all() for v in (value,mean,lower,raw,noise,proposal,action)):raise ValueError('Nonfinite NAF request')
    return dict(value=np.asarray(value),mean=mean,lower=lower,raw_mean=raw,noise=noise,proposal=proposal,action=action)

def replace_legs(fixed,reference,action,legs,lower,upper,active):
    if not active:return fixed
    a=finite(action,(ACTIONS,))
    if np.abs(a).max()>1.:raise ValueError('Normalized box request required')
    out=np.asarray(fixed,dtype=float).copy()
    out[legs]=np.clip(np.asarray(reference)[legs]+BOUND*a,np.asarray(lower)[legs],np.asarray(upper)[legs])
    return out

def complete_returns(rewards):
    r=np.asarray(rewards,dtype=np.float64)
    if r.ndim!=1 or not len(r) or not np.isfinite(r).all():raise ValueError('Complete finite original episode required')
    out=np.empty_like(r);total=0.
    for k in range(len(r)-1,-1,-1):total=r[k]+GAMMA*total;out[k]=total
    return out

def episode_replay(observations,actions,rewards,active):
    obs=np.asarray(observations,dtype=float);a=np.asarray(actions,dtype=float);r=np.asarray(rewards,dtype=float)
    mask=np.asarray(active,dtype=bool);n=len(r)
    if obs.shape!=(n,50) or a.shape!=(n,10) or mask.shape!=(n,):raise ValueError('Complete aligned original episode required')
    if not all(np.isfinite(v).all() for v in (obs,a,r)) or np.abs(a).max()>1.:raise ValueError('Finite normalized requests required')
    expected=(np.arange(n)>0)&(np.arange(n)<HORIZON)
    if not np.array_equal(mask,expected):raise ValueError('Recovery-only transitions required')
    indices=np.flatnonzero(mask)
    if not len(indices):raise ValueError('No active transition')
    returns=complete_returns(r)
    x=np.stack([features(obs[k],obs[0],int(k)) for k in indices]);nx=[];boot=[];tail=[];terminal=[]
    for k in indices:
        end=k==n-1
        nx.append(features(obs[k] if end else obs[k+1],obs[0],int(k+1)))
        boot.append(float(not end and k+1<HORIZON))
        tail.append(float(returns[k+1]) if not end and k+1==HORIZON else 0.)
        terminal.append(end)
    return dict(features=x,action=a[indices],reward=r[indices],next_features=np.asarray(nx),bootstrap=np.asarray(boot),home_tail=np.asarray(tail),terminal=np.asarray(terminal),control=indices)

def concatenate_replays(episodes):
    if not episodes:raise ValueError('Only nonempty current-round episodes')
    arrays={k:np.concatenate([e[k] for e in episodes],axis=0) for k in episodes[0]}
    arrays['episode_offsets']=np.r_[0,np.cumsum([len(e['reward']) for e in episodes])]
    return arrays

def bellman(reward,bootstrap,home_tail,next_value):
    return reward+GAMMA*(bootstrap*next_value+home_tail)

class Learner:
    def __init__(self,weights):
        import jax
        jax.config.update('jax_enable_x64',True)
        import jax.numpy as jp
        import optax
        from flax import serialization
        self.jax,self.jp,self.optax,self.serialization=jax,jp,optax,serialization
        self.params={k:jp.asarray(v) for k,v in weights.items()}
        self.target=jax.tree.map(lambda v:v.copy(),self.params)
        self.optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(RATE))
        self.state=self.optimizer.init(self.params);self.count=0
        def objective(params,x,a,y):
            prediction=q_value(params,x,a,jp);error=prediction-y
            return .5*jp.mean(error**2),(prediction,error)
        self.objective=objective
        self.gradient=jax.jit(jax.value_and_grad(objective,has_aux=True))
        self.target_values=jax.jit(lambda t,x:heads(t,x,jp)[0])

    def numpy_weights(self):
        return {k:np.asarray(v).copy() for k,v in self.params.items()}

    def numpy_target(self):
        return {k:np.asarray(v).copy() for k,v in self.target.items()}

    def state_bytes(self):
        return self.serialization.to_bytes(dict(params=self.params,target=self.target,optimizer_state=self.state,count=self.count))

    def update(self,replay,rng):
        n=len(replay['reward']);assert n>0
        for k in ('features','action','next_features','bootstrap','home_tail','terminal','control'):assert len(replay[k])==n
        offsets=np.asarray(replay['episode_offsets'])
        assert offsets.ndim==1 and offsets[0]==0 and offsets[-1]==n and np.all(np.diff(offsets)>0)
        before=self.numpy_weights();target_before=self.numpy_target();state_before=self.state_bytes()
        rng_before=copy.deepcopy(rng.bit_generator.state)
        indices=[];targets=[];predictions=[];errors=[];stats=[];gradients={k:[] for k in self.params}
        for _ in range(UPDATES):
            ix=rng.integers(0,n,size=BATCH);indices.append(ix)
            x,a,r,nx,boot,tail=[self.jp.asarray(replay[k][ix]) for k in ('features','action','reward','next_features','bootstrap','home_tail')]
            next_value=self.target_values(self.target,nx)
            y=self.jax.lax.stop_gradient(bellman(r,boot,tail,next_value))
            (loss,(prediction,error)),gradient=self.gradient(self.params,x,a,y)
            if not all(np.isfinite(np.asarray(v)).all() for v in self.jax.tree.leaves((y,prediction,error,loss,gradient))):raise ValueError('Nonfinite original NAF TD update')
            step,self.state=self.optimizer.update(gradient,self.state,self.params)
            self.params=self.optax.apply_updates(self.params,step)
            self.target=self.jax.tree.map(lambda t,v:(1-TAU)*t+TAU*v,self.target,self.params)
            self.count+=1
            if not all(np.isfinite(np.asarray(v)).all() for v in self.jax.tree.leaves((self.params,self.target,self.state))):raise ValueError('Nonfinite original NAF state')
            targets.append(np.asarray(y));predictions.append(np.asarray(prediction));errors.append(np.asarray(error))
            stats.append([float(loss),float(self.optax.global_norm(gradient)),float(np.mean(np.asarray(y))),float(np.max(np.abs(np.asarray(y)))),self.count])
            for k,v in gradient.items():gradients[k].append(np.asarray(v))
        after=self.numpy_weights()
        return dict(updates=UPDATES,joint_Q_updates=UPDATES,separate_actor_updates=0,
            replay_indices=np.asarray(indices),targets=np.asarray(targets),predictions=np.asarray(predictions),td_errors=np.asarray(errors),
            statistics=np.asarray(stats),all_gradients={k:np.asarray(v) for k,v in gradients.items()},
            before=before,after=after,target_before=target_before,target_after=self.numpy_target(),
            state_before=state_before,state_after=self.state_bytes(),rng_before=rng_before,rng_after=copy.deepcopy(rng.bit_generator.state),
            changes={k:float(np.linalg.norm(after[k]-before[k])) for k in before})

    def checkpoint(self,path):
        np.savez_compressed(path.with_suffix('.npz'),**self.numpy_weights())
        path.with_suffix('.learner.msgpack').write_bytes(self.state_bytes())

