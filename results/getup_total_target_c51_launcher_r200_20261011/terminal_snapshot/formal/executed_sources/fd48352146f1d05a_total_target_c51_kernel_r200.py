"""Fresh categorical value policy. No actor, simulator or old learner state."""
import numpy as np

SEED=300
BOUND=.18
GAMMA=.99**.2
HORIZON=529
INPUTS=101
ACTIONS=21
ATOMS=51
VMIN=-128.
VMAX=128.
SUPPORT=np.linspace(VMIN,VMAX,ATOMS,dtype=np.float64)
CODE_MAGNITUDE=1e-4
EPSILON=.1
RATE=1e-4
UPDATES=64
BATCH=512

def finite(value,shape):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite declared causal array required')
    return a

def features(current,initial,control):
    if not isinstance(control,(int,np.integer)) or not 0<=control<=2279:raise ValueError('Original control clock required')
    return np.r_[np.tanh(finite(current,(50,))),np.tanh(finite(initial,(50,))),min(control/528.,1.)]

def codebook():
    out=np.zeros((ACTIONS,10))
    for j in range(10):
        out[1+2*j,j]=CODE_MAGNITUDE
        out[2+2*j,j]=-CODE_MAGNITUDE
    return out

def initial_weights():
    rng=np.random.default_rng(SEED);weights={}
    for i,(a,b) in enumerate([(101,32),(32,16),(16,ACTIONS*ATOMS)]):
        weights[f'value_w{i}']=np.zeros((a,b)) if i==2 else rng.normal(0,1/np.sqrt(a),(a,b))
        weights[f'value_b{i}']=np.zeros(b)
    return weights

def network(weights,x,xp=np):
    for i in range(3):
        x=x@weights[f'value_w{i}']+weights[f'value_b{i}']
        if i<2:x=xp.tanh(x)
    return x.reshape(x.shape[:-1]+(ACTIONS,ATOMS))

def probabilities(logits,xp=np):
    shifted=logits-xp.max(logits,axis=-1,keepdims=True)
    p=xp.exp(shifted)
    return p/xp.sum(p,axis=-1,keepdims=True)

def values(weights,x,xp=np):
    p=probabilities(network(weights,x,xp),xp)
    return xp.sum(p*xp.asarray(SUPPORT),axis=-1),p

def select(weights,x,rng,sample):
    q,p=values(weights,finite(x,(101,)))
    finite(q,(ACTIONS,));finite(p,(ACTIONS,ATOMS))
    greedy=int(np.argmax(q)) # First-index exact tie: initial uniform => zero code.
    gate=float(rng.random()) if sample else 0.
    draw=int(rng.integers(ACTIONS)) if sample else 0
    chosen=draw if sample and gate<EPSILON else greedy
    return dict(action=codebook()[chosen].copy(),chosen=chosen,greedy=greedy,
                expected_values=q,selected_probabilities=p[chosen],
                exploration_gate=gate,exploration_code=draw)

def replace_legs(fixed,reference,action,legs,lower,upper,active):
    if not active:return fixed
    a=finite(action,(10,))
    if not any(np.array_equal(a,c) for c in codebook()):raise ValueError('Exact fixed action code required')
    out=np.asarray(fixed,dtype=float).copy()
    out[legs]=np.clip(np.asarray(reference)[legs]+BOUND*a,np.asarray(lower)[legs],np.asarray(upper)[legs])
    return out

def complete_returns(rewards):
    r=np.asarray(rewards,dtype=np.float64)
    if r.ndim!=1 or not len(r) or not np.isfinite(r).all():raise ValueError('Complete finite episode required')
    out=np.empty_like(r);total=0.
    for k in range(len(r)-1,-1,-1):total=r[k]+GAMMA*total;out[k]=total
    return out

def episode_replay(observations,codes,rewards,active):
    obs=np.asarray(observations,dtype=float);c=np.asarray(codes);r=np.asarray(rewards,dtype=float)
    mask=np.asarray(active,dtype=bool);n=len(r)
    if obs.shape!=(n,50) or c.shape!=(n,) or mask.shape!=(n,):raise ValueError('Complete aligned episode required')
    if c.dtype.kind not in 'iu' or np.any(c<0) or np.any(c>=ACTIONS):raise ValueError('Integer action codes required')
    if not all(np.isfinite(v).all() for v in (obs,r)):raise ValueError('Nonfinite episode')
    expected=(np.arange(n)>0)&(np.arange(n)<HORIZON)
    if not np.array_equal(mask,expected):raise ValueError('Recovery-only replay mask required')
    ret=complete_returns(r);indices=np.flatnonzero(mask)
    if not len(indices):raise ValueError('No recovery transition')
    x=np.stack([features(obs[k],obs[0],int(k)) for k in indices]);nextx=[];boot=[];boundary=[];terminal=[]
    for k in indices:
        end=k==n-1
        nextx.append(features(obs[k+1] if not end else obs[k],obs[0],int(k+1)))
        boot.append(float(not end and k+1<HORIZON))
        boundary.append(float(ret[k+1]) if not end and k+1==HORIZON else 0.)
        terminal.append(end)
    return dict(features=x,action_code=c[indices],reward=r[indices],next_features=np.asarray(nextx),
                bootstrap=np.asarray(boot),home_tail=np.asarray(boundary),terminal=np.asarray(terminal),control=indices)

def project(reward,bootstrap,home_tail,next_probabilities,xp=np):
    # Fixed-support estimator approximation only: never change original rewards or labels.
    atoms=xp.asarray(SUPPORT)
    raw=reward[:,None]+GAMMA*(home_tail[:,None]+bootstrap[:,None]*atoms[None,:])
    clipped=xp.clip(raw,VMIN,VMAX)
    location=(clipped-VMIN)/((VMAX-VMIN)/(ATOMS-1))
    triangular=xp.clip(1-xp.abs(location[:,:,None]-xp.arange(ATOMS)[None,None,:]),0.,1.)
    mass=xp.sum(next_probabilities[:,:,None]*triangular,axis=1)
    low=xp.sum(next_probabilities*(raw<VMIN),axis=1)
    high=xp.sum(next_probabilities*(raw>VMAX),axis=1)
    return mass,raw,low,high

class Learner:
    def __init__(self,weights):
        import jax
        jax.config.update('jax_enable_x64',True)
        import jax.numpy as jp
        import optax
        from flax import serialization
        self.jax,self.jp,self.optax,self.serialization=jax,jp,optax,serialization
        self.online={k:jp.asarray(v) for k,v in weights.items()}
        self.target=jax.tree.map(lambda v:v.copy(),self.online)
        self.optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(RATE))
        self.state=self.optimizer.init(self.online);self.count=0
        def target_value(t,nx,r,boot,tail):
            q,p=values(t,nx,jp);greedy=jp.argmax(q,axis=1)
            selected=p[jp.arange(len(nx)),greedy]
            mass,raw,low,high=project(r,boot,tail,selected,jp)
            return jax.tree.map(jax.lax.stop_gradient,(mass,raw,low,high,greedy,selected))
        def objective(w,x,c,m):
            logits=network(w,x,jp)[jp.arange(len(x)),c]
            logp=jax.nn.log_softmax(logits,axis=1)
            loss=-jp.mean(jp.sum(jax.lax.stop_gradient(m)*logp,axis=1))
            return loss
        self.target_value=jax.jit(target_value);self.objective=objective
        self.gradient=jax.jit(jax.value_and_grad(objective))

    def numpy_weights(self):return {k:np.asarray(v).copy() for k,v in self.online.items()}

    def update(self,replay,rng):
        n=len(replay['reward']);assert n>0 and all(len(v)==n for v in replay.values())
        before=self.numpy_weights();target_before={k:np.asarray(v).copy() for k,v in self.target.items()}
        stats=[];norms=[];indices=[];masses=[];raws=[];greedies=[];selecteds=[]
        for _ in range(UPDATES):
            ix=rng.integers(0,n,size=BATCH);indices.append(ix)
            x,c,r,nx,boot,tail=[self.jp.asarray(replay[k][ix]) for k in ('features','action_code','reward','next_features','bootstrap','home_tail')]
            mass,raw,low,high,greedy,selected=self.target_value(self.target,nx,r,boot,tail)
            np.testing.assert_allclose(np.asarray(mass).sum(axis=1),1.,atol=2e-14,rtol=0)
            loss,g=self.gradient(self.online,x,c,mass)
            change,self.state=self.optimizer.update(g,self.state,self.online)
            self.online=self.optax.apply_updates(self.online,change);self.count+=1
            assert all(np.isfinite(np.asarray(v)).all() for v in self.jax.tree.leaves((g,self.online)))
            stats.append([float(loss),float(self.jp.mean(low)),float(self.jp.mean(high)),
                          float(self.jp.min(raw)),float(self.jp.max(raw)),self.count])
            norms.append(float(self.optax.global_norm(g)));masses.append(np.asarray(mass));raws.append(np.asarray(raw))
            greedies.append(np.asarray(greedy));selecteds.append(np.asarray(selected))
        for k,v in self.target.items():np.testing.assert_array_equal(np.asarray(v),target_before[k])
        # Exactly one hard copy after this round's 64 updates; no Polyak/delayed actor.
        self.target=self.jax.tree.map(lambda v:v.copy(),self.online)
        after=self.numpy_weights()
        return dict(updates=UPDATES,statistics=np.asarray(stats),gradient_norms=np.asarray(norms),
                    replay_indices=np.asarray(indices),projected_masses=np.asarray(masses),
                    unbounded_target_atoms=np.asarray(raws),next_greedy_codes=np.asarray(greedies),
                    next_selected_probabilities=np.asarray(selecteds),
                    last_gradient={k:np.asarray(v) for k,v in g.items()},
                    changes={k:float(np.linalg.norm(after[k]-before[k])) for k in before},
                    hard_target_copies=1)

    def checkpoint(self,path):
        np.savez_compressed(path.with_suffix('.npz'),**self.numpy_weights())
        data=dict(online=self.online,target=self.target,state=self.state,count=self.count)
        path.with_suffix('.learner.msgpack').write_bytes(self.serialization.to_bytes(data))

