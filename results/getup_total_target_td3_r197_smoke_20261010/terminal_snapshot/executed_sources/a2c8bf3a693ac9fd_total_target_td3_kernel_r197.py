"""Fresh twin-Q replay learner for bounded total recovery leg targets."""
import numpy as np

SEED=297
BOUND=.18
SIGMA=1e-4
GAMMA=.99**.2
HORIZON=529
INPUTS=101
CRITIC_RATE=1e-4
ACTOR_RATE=1e-6
UPDATES=64
BATCH=512
DELAY=2
TAU=.005
TARGET_SIGMA=1e-4
TARGET_CLIP=2e-4

def finite(value,shape):
    a=np.asarray(value,dtype=np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite declared causal array required')
    return a

def features(current,initial,control):
    if not isinstance(control,(int,np.integer)) or not 0<=control<=2279:raise ValueError('Original control clock required')
    return np.r_[np.tanh(finite(current,(50,))),np.tanh(finite(initial,(50,))),min(control/528.,1.)]

def initial_weights():
    rng=np.random.default_rng(SEED);weights={}
    for name,n,last in [('actor',101,10),('q1',111,1),('q2',111,1)]:
        for i,(a,b) in enumerate([(n,32),(32,16),(16,last)]):
            weights[f'{name}_w{i}']=np.zeros((a,b)) if name=='actor' and i==2 else rng.normal(0,1/np.sqrt(a),(a,b))
            weights[f'{name}_b{i}']=np.zeros(b)
    return weights

def network(weights,x,name,xp=np):
    for i in range(3):
        x=x@weights[f'{name}_w{i}']+weights[f'{name}_b{i}']
        if i<2 or name=='actor':x=xp.tanh(x)
    return x

def sample_action(mean,noise):
    return np.clip(finite(mean,(10,))+SIGMA*finite(noise,(10,)),-1.,1.)

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
    for k in range(len(r)-1,-1,-1):
        total=r[k]+GAMMA*total;out[k]=total
    return out

def episode_replay(observations,actions,rewards,active):
    """Only newly recorded causal transitions; original home tail is fixed offline backup."""
    obs=np.asarray(observations,dtype=float);a=np.asarray(actions,dtype=float);r=np.asarray(rewards,dtype=float)
    mask=np.asarray(active,dtype=bool);n=len(r)
    if obs.shape!=(n,50) or a.shape!=(n,10) or mask.shape!=(n,):raise ValueError('Complete aligned episode required')
    if not all(np.isfinite(v).all() for v in (obs,a,r)):raise ValueError('Nonfinite episode')
    expected=(np.arange(n)>0)&(np.arange(n)<HORIZON)
    if not np.array_equal(mask,expected):raise ValueError('Recovery-only replay mask required')
    ret=complete_returns(r);indices=np.flatnonzero(mask)
    if not len(indices):raise ValueError('No recovery transition')
    x=np.stack([features(obs[k],obs[0],int(k)) for k in indices])
    nextx=[];boot=[];boundary=[];terminal=[]
    for k in indices:
        end=k==n-1
        nxt=obs[k+1] if not end else obs[k]
        nextx.append(features(nxt,obs[0],int(k+1)))
        boot.append(float(not end and k+1<HORIZON))
        boundary.append(float(ret[k+1]) if not end and k+1==HORIZON else 0.)
        terminal.append(end)
    return dict(features=x,action=a[indices],reward=r[indices],next_features=np.asarray(nextx),
                bootstrap=np.asarray(boot),home_tail=np.asarray(boundary),
                terminal=np.asarray(terminal),control=indices)

def bellman(reward,bootstrap,home_tail,q1,q2,xp=np):
    return reward+GAMMA*(bootstrap*xp.minimum(q1,q2)+home_tail)

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
        self.target_actor=jax.tree.map(lambda v:v.copy(),self.actor)
        self.target_critics=jax.tree.map(lambda v:v.copy(),self.critics)
        self.actor_optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(ACTOR_RATE))
        self.critic_optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(CRITIC_RATE))
        self.actor_state=self.actor_optimizer.init(self.actor);self.critic_state=self.critic_optimizer.init(self.critics)
        self.count=0
        def target_value(ta,tq,nx,noise,r,boot,tail):
            na=jp.clip(network(ta,nx,'actor',jp)+jp.clip(TARGET_SIGMA*noise,-TARGET_CLIP,TARGET_CLIP),-1.,1.)
            qa=jp.concatenate([nx,na],axis=1)
            y=bellman(r,boot,tail,network(tq,qa,'q1',jp)[:,0],network(tq,qa,'q2',jp)[:,0],jp)
            return jax.lax.stop_gradient(y)
        def critic_objective(q,x,a,y):
            inp=jp.concatenate([x,a],axis=1)
            e1=network(q,inp,'q1',jp)[:,0]-y;e2=network(q,inp,'q2',jp)[:,0]-y
            loss=jp.mean(e1**2+e2**2)
            return loss,jp.array([loss,jp.mean(y),jp.max(jp.abs(y))])
        def actor_objective(p,q,x):
            a=network(p,x,'actor',jp)
            return -jp.mean(network(q,jp.concatenate([x,a],axis=1),'q1',jp)[:,0])
        self.target_value=jax.jit(target_value)
        self.critic_objective=critic_objective;self.actor_objective=actor_objective
        self.critic_gradient=jax.jit(jax.value_and_grad(critic_objective,has_aux=True))
        self.actor_gradient=jax.jit(jax.value_and_grad(actor_objective))

    def numpy_weights(self):
        return {k:np.asarray(v).copy() for k,v in {**self.actor,**self.critics}.items()}

    def update(self,replay,rng):
        n=len(replay['reward'])
        assert n>0 and all(len(v)==n for v in replay.values())
        before=self.numpy_weights();stats=[];norms=[];indices=[];noises=[];lastq=lasta=None;actor_updates=0
        for _ in range(UPDATES):
            ix=rng.integers(0,n,size=BATCH);noise=rng.normal(size=(BATCH,10))
            indices.append(ix);noises.append(noise)
            x,a,r,nx,boot,tail=[self.jp.asarray(replay[k][ix]) for k in ('features','action','reward','next_features','bootstrap','home_tail')]
            y=self.target_value(self.target_actor,self.target_critics,nx,self.jp.asarray(noise),r,boot,tail)
            (_,s),grad=self.critic_gradient(self.critics,x,a,y)
            assert all(np.isfinite(np.asarray(v)).all() for v in self.jax.tree.leaves(grad))
            qnorm=float(self.optax.global_norm(grad));lastq=grad
            change,self.critic_state=self.critic_optimizer.update(grad,self.critic_state,self.critics)
            self.critics=self.optax.apply_updates(self.critics,change);self.count+=1
            aloss=0.;anorm=0.;updated=self.count%DELAY==0
            if updated:
                loss,ag=self.actor_gradient(self.actor,self.critics,x)
                assert all(np.isfinite(np.asarray(v)).all() for v in self.jax.tree.leaves(ag))
                aloss=float(loss);anorm=float(self.optax.global_norm(ag));lasta=ag
                change,self.actor_state=self.actor_optimizer.update(ag,self.actor_state,self.actor)
                self.actor=self.optax.apply_updates(self.actor,change);actor_updates+=1
                self.target_actor=self.jax.tree.map(lambda t,v:(1-TAU)*t+TAU*v,self.target_actor,self.actor)
                self.target_critics=self.jax.tree.map(lambda t,v:(1-TAU)*t+TAU*v,self.target_critics,self.critics)
            stats.append(np.r_[np.asarray(s),aloss,float(updated),self.count]);norms.append([qnorm,anorm])
        after=self.numpy_weights();assert all(np.isfinite(v).all() for v in after.values())
        return dict(updates=UPDATES,actor_updates=actor_updates,statistics=np.asarray(stats),gradient_norms=np.asarray(norms),
                    replay_indices=np.asarray(indices),target_noise=np.asarray(noises),
                    last_gradient={k:np.asarray(v) for k,v in {**lastq,**lasta}.items()},
                    changes={k:float(np.linalg.norm(after[k]-before[k])) for k in before})

    def checkpoint(self,path):
        np.savez_compressed(path.with_suffix('.npz'),**self.numpy_weights())
        data=dict(actor=self.actor,critics=self.critics,target_actor=self.target_actor,target_critics=self.target_critics,
                  actor_state=self.actor_state,critic_state=self.critic_state,count=self.count)
        path.with_suffix('.learner.msgpack').write_bytes(self.serialization.to_bytes(data))

