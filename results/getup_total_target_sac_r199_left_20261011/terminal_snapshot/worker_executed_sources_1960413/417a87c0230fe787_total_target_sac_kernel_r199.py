"""Fresh entropy-regularized stochastic learner; causal sensors only, no simulator."""
import numpy as np

SEED=299
BOUND=.18
GAMMA=.99**.2
HORIZON=529
INPUTS=101
INITIAL_LOG_STD=float(np.log(1e-4))
LOG_STD_MIN=-12.
LOG_STD_MAX=-4.
INITIAL_ALPHA=1e-4
TARGET_ENTROPY=-75.
CRITIC_RATE=1e-4
ACTOR_RATE=1e-6
ALPHA_RATE=1e-4
UPDATES=64
BATCH=512
TAU=.005

def finite(value,shape):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite declared causal array required')
    return a

def features(current,initial,control):
    if not isinstance(control,(int,np.integer)) or not 0<=control<=2279:raise ValueError('Original control clock required')
    return np.r_[np.tanh(finite(current,(50,))),np.tanh(finite(initial,(50,))),min(control/528.,1.)]

def initial_weights():
    rng=np.random.default_rng(SEED);weights={}
    for name,n,last in [('actor',101,20),('q1',111,1),('q2',111,1)]:
        for i,(a,b) in enumerate([(n,32),(32,16),(16,last)]):
            weights[f'{name}_w{i}']=np.zeros((a,b)) if name=='actor' and i==2 else rng.normal(0,1/np.sqrt(a),(a,b))
            weights[f'{name}_b{i}']=np.zeros(b)
    return weights

def network(weights,x,name,xp=np):
    for i in range(3):
        x=x@weights[f'{name}_w{i}']+weights[f'{name}_b{i}']
        if i<2:x=xp.tanh(x)
    return x

def distribution(weights,x,xp=np):
    out=network(weights,x,'actor',xp)
    return out[...,:10],xp.clip(INITIAL_LOG_STD+out[...,10:],LOG_STD_MIN,LOG_STD_MAX)

def squashed(mu,logstd,noise,xp=np):
    z=mu+xp.exp(logstd)*noise
    a=xp.tanh(z)
    # Exact normalized requested-density Jacobian, stable even at saturated tanh.
    jac=2*(np.log(2.)-z-xp.logaddexp(0.,-2*z))
    logp=xp.sum(-.5*(noise**2+np.log(2*np.pi))-logstd-jac,axis=-1)
    return a,z,logp

def replace_legs(fixed,reference,action,legs,lower,upper,active):
    if not active:return fixed
    a=finite(action,(10,))
    if np.abs(a).max()>1.:raise ValueError('Normalized requested action must be bounded')
    out=np.asarray(fixed,dtype=float).copy()
    out[legs]=np.clip(np.asarray(reference)[legs]+BOUND*a,np.asarray(lower)[legs],np.asarray(upper)[legs])
    return out

def complete_returns(rewards):
    r=np.asarray(rewards,dtype=np.float64)
    if r.ndim!=1 or not len(r) or not np.isfinite(r).all():raise ValueError('Complete finite episode required')
    out=np.empty_like(r);total=0.
    for k in range(len(r)-1,-1,-1):total=r[k]+GAMMA*total;out[k]=total
    return out

def episode_replay(observations,actions,rewards,active):
    obs=np.asarray(observations,dtype=float);a=np.asarray(actions,dtype=float);r=np.asarray(rewards,dtype=float)
    mask=np.asarray(active,dtype=bool);n=len(r)
    if obs.shape!=(n,50) or a.shape!=(n,10) or mask.shape!=(n,):raise ValueError('Complete aligned episode required')
    if not all(np.isfinite(v).all() for v in (obs,a,r)):raise ValueError('Nonfinite episode')
    expected=(np.arange(n)>0)&(np.arange(n)<HORIZON)
    if not np.array_equal(mask,expected):raise ValueError('Recovery-only replay mask required')
    if np.abs(a).max()>1.:raise ValueError('Recorded normalized request required')
    ret=complete_returns(r);indices=np.flatnonzero(mask)
    if not len(indices):raise ValueError('No recovery transition')
    x=np.stack([features(obs[k],obs[0],int(k)) for k in indices]);nextx=[];boot=[];boundary=[];terminal=[]
    for k in indices:
        end=k==n-1
        nextx.append(features(obs[k+1] if not end else obs[k],obs[0],int(k+1)))
        boot.append(float(not end and k+1<HORIZON))
        boundary.append(float(ret[k+1]) if not end and k+1==HORIZON else 0.)
        terminal.append(end)
    return dict(features=x,action=a[indices],reward=r[indices],next_features=np.asarray(nextx),bootstrap=np.asarray(boot),home_tail=np.asarray(boundary),terminal=np.asarray(terminal),control=indices)

def soft_bellman(reward,bootstrap,home_tail,q1,q2,logp,alpha,xp=np):
    # Entropy is offline recovery-policy objective only; original home and rewards unchanged.
    return reward+GAMMA*(bootstrap*(xp.minimum(q1,q2)-alpha*logp)+home_tail)

class Learner:
    def __init__(self,weights):
        import jax
        jax.config.update('jax_enable_x64',True)
        import jax.numpy as jp
        import optax
        from flax import serialization
        self.jax,self.jp,self.optax,self.serialization=jax,jp,optax,serialization
        self.actor={k:jp.asarray(v) for k,v in weights.items() if k.startswith('actor_')}
        self.critics={k:jp.asarray(v) for k,v in weights.items() if not k.startswith('actor_')}
        self.target_critics=jax.tree.map(lambda v:v.copy(),self.critics)
        self.log_alpha=jp.asarray(np.log(INITIAL_ALPHA))
        self.actor_optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(ACTOR_RATE))
        self.critic_optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(CRITIC_RATE))
        self.alpha_optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(ALPHA_RATE))
        self.actor_state=self.actor_optimizer.init(self.actor);self.critic_state=self.critic_optimizer.init(self.critics);self.alpha_state=self.alpha_optimizer.init(self.log_alpha)
        self.count=0
        def target_value(p,tq,la,nx,noise,r,boot,tail):
            mu,ls=distribution(p,nx,jp);a,_,lp=squashed(mu,ls,noise,jp)
            inp=jp.concatenate([nx,a],axis=1)
            y=soft_bellman(r,boot,tail,network(tq,inp,'q1',jp)[:,0],network(tq,inp,'q2',jp)[:,0],lp,jp.exp(la),jp)
            return jax.lax.stop_gradient(y)
        def critic_objective(q,x,a,y):
            inp=jp.concatenate([x,a],axis=1)
            e1=network(q,inp,'q1',jp)[:,0]-y;e2=network(q,inp,'q2',jp)[:,0]-y
            loss=.5*jp.mean(e1**2+e2**2)
            return loss,jp.array([loss,jp.mean(y),jp.max(jp.abs(y))])
        def actor_objective(p,q,la,x,noise):
            mu,ls=distribution(p,x,jp);a,_,lp=squashed(mu,ls,noise,jp)
            inp=jp.concatenate([x,a],axis=1)
            value=jp.minimum(network(q,inp,'q1',jp)[:,0],network(q,inp,'q2',jp)[:,0])
            loss=jp.mean(jax.lax.stop_gradient(jp.exp(la))*lp-value)
            return loss,jp.array([loss,jp.mean(lp),jp.mean(value)])
        def alpha_objective(la,logp):
            # Dual alpha loss expressed in log-alpha coordinates, not log-alpha surrogate.
            return -jp.exp(la)*jp.mean(jax.lax.stop_gradient(logp+TARGET_ENTROPY))
        self.target_value=jax.jit(target_value)
        self.critic_objective=critic_objective;self.actor_objective=actor_objective;self.alpha_objective=alpha_objective
        self.critic_gradient=jax.jit(jax.value_and_grad(critic_objective,has_aux=True))
        self.actor_gradient=jax.jit(jax.value_and_grad(actor_objective,has_aux=True))
        self.alpha_gradient=jax.jit(jax.value_and_grad(alpha_objective))

    def numpy_weights(self):return {k:np.asarray(v).copy() for k,v in {**self.actor,**self.critics}.items()}

    def update(self,replay,rng):
        n=len(replay['reward']);assert n>0 and all(len(v)==n for v in replay.values())
        before=self.numpy_weights();before_alpha=float(self.log_alpha);stats=[];norms=[];indices=[];next_noises=[];policy_noises=[]
        for _ in range(UPDATES):
            ix=rng.integers(0,n,size=BATCH);nn=rng.normal(size=(BATCH,10));pn=rng.normal(size=(BATCH,10))
            indices.append(ix);next_noises.append(nn);policy_noises.append(pn)
            x,a,r,nx,boot,tail=[self.jp.asarray(replay[k][ix]) for k in ('features','action','reward','next_features','bootstrap','home_tail')]
            y=self.target_value(self.actor,self.target_critics,self.log_alpha,nx,self.jp.asarray(nn),r,boot,tail)
            (_,qs),qg=self.critic_gradient(self.critics,x,a,y)
            change,self.critic_state=self.critic_optimizer.update(qg,self.critic_state,self.critics);self.critics=self.optax.apply_updates(self.critics,change)
            (_,ps),pg=self.actor_gradient(self.actor,self.critics,self.log_alpha,x,self.jp.asarray(pn))
            change,self.actor_state=self.actor_optimizer.update(pg,self.actor_state,self.actor);self.actor=self.optax.apply_updates(self.actor,change)
            mu,ls=distribution(self.actor,x,self.jp);_,_,lp=squashed(mu,ls,self.jp.asarray(pn),self.jp)
            aloss,ag=self.alpha_gradient(self.log_alpha,lp)
            change,self.alpha_state=self.alpha_optimizer.update(ag,self.alpha_state,self.log_alpha);self.log_alpha=self.optax.apply_updates(self.log_alpha,change)
            self.target_critics=self.jax.tree.map(lambda t,v:(1-TAU)*t+TAU*v,self.target_critics,self.critics)
            self.count+=1
            assert all(np.isfinite(np.asarray(v)).all() for v in self.jax.tree.leaves((qg,pg,ag,self.actor,self.critics,self.target_critics,self.log_alpha)))
            stats.append(np.r_[np.asarray(qs),np.asarray(ps),float(aloss),float(self.log_alpha),float(self.jp.exp(self.log_alpha)),self.count])
            norms.append([float(self.optax.global_norm(qg)),float(self.optax.global_norm(pg)),float(np.abs(ag))])
        after=self.numpy_weights()
        return dict(updates=UPDATES,actor_updates=UPDATES,alpha_updates=UPDATES,statistics=np.asarray(stats),gradient_norms=np.asarray(norms),replay_indices=np.asarray(indices),next_policy_noise=np.asarray(next_noises),actor_policy_noise=np.asarray(policy_noises),last_gradient={k:np.asarray(v) for k,v in {**qg,**pg}.items()},last_alpha_gradient=np.asarray(ag),log_alpha_change=float(self.log_alpha)-before_alpha,changes={k:float(np.linalg.norm(after[k]-before[k])) for k in before})

    def checkpoint(self,path):
        np.savez_compressed(path.with_suffix('.npz'),**self.numpy_weights())
        data=dict(actor=self.actor,critics=self.critics,target_critics=self.target_critics,log_alpha=self.log_alpha,actor_state=self.actor_state,critic_state=self.critic_state,alpha_state=self.alpha_state,count=self.count)
        path.with_suffix('.learner.msgpack').write_bytes(self.serialization.to_bytes(data))
