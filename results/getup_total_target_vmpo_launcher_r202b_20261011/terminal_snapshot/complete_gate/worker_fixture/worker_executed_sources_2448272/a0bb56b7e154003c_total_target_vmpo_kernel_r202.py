"""Fresh finite EM-style nonparametric-target policy fitting; no physical-safety claim."""
import numpy as np

SEED = 302
BOUND = .18
SIGMA = 1e-4
GAMMA = .99 ** .2
MAX_KL = .001
E_KL = .1
ACTOR_STEPS = 64
ACTOR_LR = 1e-7
DUAL_LR = .01
ETA_STEPS = 80
VALUE_UPDATES = 64
VALUE_LR = 1e-4


def finite(value, shape):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Finite declared causal shape required')
    return a


def features(current, initial, control):
    if not isinstance(control, (int, np.integer)) or not 0 <= control <= 2279:
        raise ValueError('Original control clock required')
    a = finite(current, (50,)); b = finite(initial, (50,))
    return np.r_[np.tanh(a), np.tanh(b), min(control / 528., 1.)]


def initial_weights():
    rng = np.random.default_rng(SEED)
    weights = {}
    for name, last in [('actor', 10), ('value', 1)]:
        for i, (n, m) in enumerate([(101, 32), (32, 16), (16, last)]):
            weights[f'{name}_w{i}'] = (np.zeros((n, m)) if name == 'actor' and i == 2
                                       else rng.normal(0, 1 / np.sqrt(n), (n, m)))
            weights[f'{name}_b{i}'] = np.zeros(m)
    return weights


def network(weights, x, name, xp=np):
    for i in range(3):
        x = x @ weights[f'{name}_w{i}'] + weights[f'{name}_b{i}']
        if i < 2:
            x = xp.tanh(x)
    return x


def log_probability(z, mean, xp=np):
    # Latent Gaussian, not post-joint/slew executed-action density.
    return -.5 * xp.sum(((z - mean) / SIGMA) ** 2 +
                       2 * np.log(SIGMA) + np.log(2 * np.pi), axis=-1)


def exact_mean_kl(old_mean, new_mean, xp=np):
    return xp.mean(xp.sum(((old_mean - new_mean) / SIGMA) ** 2, axis=-1) / 2)


def replace_legs(fixed, reference, latent, legs, lower, upper, active):
    if not active:
        return fixed
    adjusted = np.asarray(fixed, dtype=float).copy()
    adjusted[legs] = np.clip(np.asarray(reference)[legs] +
                             BOUND * np.tanh(finite(latent, (10,))),
                             np.asarray(lower)[legs], np.asarray(upper)[legs])
    return adjusted


def complete_returns(rewards):
    r = np.asarray(rewards, dtype=np.float64)
    if r.ndim != 1 or not len(r) or not np.isfinite(r).all():
        raise ValueError('One complete finite episode required')
    out = np.empty_like(r); total = 0.
    for k in range(len(r) - 1, -1, -1):
        total = r[k] + GAMMA * total; out[k] = total
    return out


def e_step(advantage):
    a=np.asarray(advantage,dtype=np.float64)
    if a.ndim!=1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Finite complete active advantages required')
    # Stable descending order: original sample order breaks exact ties.
    indices=np.argsort(-a,kind='stable')[:(len(a)+1)//2]
    top=a[indices]
    def at(eta):
        u=(top-top.max())/eta
        q=np.exp(u);q/=q.sum()
        positive=q>0
        kl=float(np.sum(q[positive]*np.log(q[positive]*len(q))))
        logmean=float(top.max()/eta+np.log(np.exp(u).mean()))
        return q,kl,float(eta*(E_KL+logmean))
    lo,hi=1e-6,1e6;trace=[]
    for _ in range(ETA_STEPS):
        eta=np.sqrt(lo*hi);q,kl,dual=at(eta)
        trace.append((lo,hi,eta,kl,dual))
        if kl>E_KL:lo=eta
        else:hi=eta
    q,kl,dual=at(hi)
    if kl>E_KL+1e-12:raise ArithmeticError('Finite E-step constraint not attained')
    weights=np.zeros(len(a));weights[indices]=q
    return dict(indices=indices,weights=weights,eta=float(hi),top_kl=kl,
                dual_value=dual,trace=np.asarray(trace),epsilon=E_KL,
                bottom_half_not_discarded_from_original_evidence=True)


class Learner:
    def __init__(self, weights):
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jp
        import optax
        from jax.flatten_util import ravel_pytree
        from flax import serialization
        self.jax,self.jp,self.optax=jax,jp,optax
        self.serialization=serialization
        self.actor={k:jp.asarray(v) for k,v in weights.items() if k.startswith('actor_')}
        self.value={k:jp.asarray(v) for k,v in weights.items() if k.startswith('value_')}
        self.unravel=ravel_pytree(self.actor)[1]
        self.actor_optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(ACTOR_LR))
        self.value_optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adam(VALUE_LR))
        self.actor_state=self.actor_optimizer.init(self.actor)
        self.value_state=self.value_optimizer.init(self.value)
        self.actor_fit_attempts=0;self.actor_batches=0;self.value_updates=0
        self.alpha=1.
        def actor_loss(p,x,z,q,oldmean,alpha):
            mean=network(p,x,'actor',jp)
            mle=-jp.sum(jax.lax.stop_gradient(q)*log_probability(z,mean,jp))
            kl=exact_mean_kl(jax.lax.stop_gradient(oldmean),mean,jp)
            return mle+jax.lax.stop_gradient(alpha)*kl,(mle,kl)
        self.actor_gradient=jax.jit(jax.value_and_grad(actor_loss,has_aux=True))
        self.materialized_mean=jax.jit(lambda p,x:network(p,x,'actor',jp))
        self.value_gradient=jax.jit(jax.value_and_grad(lambda p,x,r:
            jp.mean((network(p,x,'value',jp)[:,0]-r)**2)))

    def actor_update(self,x,z,advantage):
        x=np.asarray(x,dtype=np.float64);z=np.asarray(z,dtype=np.float64)
        if x.ndim!=2 or not len(x):raise ValueError('Complete nonempty active batch required')
        finite(x,(len(x),101));finite(z,(len(x),10));finite(advantage,(len(x),))
        e=e_step(advantage)
        xp,zp,qp=map(self.jp.asarray,(x,z,e['weights']))
        before={k:np.asarray(v).copy() for k,v in self.actor.items()}
        before_state=self.actor_state;before_alpha=self.alpha
        oldmean=np.asarray(self.materialized_mean(self.actor,xp)).copy()
        baseline=-float(np.sum(e['weights']*log_probability(z,oldmean)))
        oldp=self.jp.asarray(oldmean)
        from jax.flatten_util import ravel_pytree
        theta0=np.asarray(ravel_pytree(self.actor)[0]).copy()
        parameters=[];gradients=[];scalars=[];states=[]
        for i in range(ACTOR_STEPS):
            (loss,(mle,kl)),g=self.actor_gradient(self.actor,xp,zp,qp,oldp,self.jp.asarray(self.alpha))
            flat=np.asarray(ravel_pytree(g)[0]);finite(flat,theta0.shape)
            if not np.isfinite([float(loss),float(mle),float(kl)]).all():
                raise ArithmeticError('Nonfinite M-step')
            norm=float(self.optax.global_norm(g))
            update,self.actor_state=self.actor_optimizer.update(g,self.actor_state,self.actor)
            self.actor=self.optax.apply_updates(self.actor,update)
            newmean=np.asarray(self.materialized_mean(self.actor,xp))
            newkl=float(exact_mean_kl(oldmean,newmean))
            newmle=-float(np.sum(e['weights']*log_probability(z,newmean)))
            if not np.isfinite([newkl,newmle]).all():raise ArithmeticError('Nonfinite proposed M-step')
            previous_alpha=self.alpha
            self.alpha=max(1e-8,self.alpha+DUAL_LR*(newkl-MAX_KL))
            gradients.append(flat.copy())
            parameters.append(np.asarray(ravel_pytree(self.actor)[0]).copy())
            scalars.append([float(loss),float(mle),float(kl),norm,newmle,newkl,previous_alpha,self.alpha])
            states.append(self.serialization.to_bytes(self.actor_state))
            self.actor_fit_attempts+=1
        accepted=bool(0<=newkl<=MAX_KL and newmle<baseline)
        proposal={k:np.asarray(v).copy() for k,v in self.actor.items()}
        proposal_state=self.serialization.to_bytes(self.actor_state)
        proposal_alpha=self.alpha
        if not accepted:
            self.actor={k:self.jp.asarray(v) for k,v in before.items()}
            self.actor_state=before_state;self.alpha=before_alpha
        self.actor_batches+=1
        chosenmean=np.asarray(self.materialized_mean(self.actor,xp))
        chosenkl=float(exact_mean_kl(oldmean,chosenmean))
        if not accepted:
            np.testing.assert_array_equal(chosenmean,oldmean)
            assert chosenkl==0.
        return dict(e_step=e,old_theta=theta0,old_mean=oldmean,
            gradients=np.stack(gradients),parameters=np.stack(parameters),scalars=np.asarray(scalars),
            actor_optimizer_step_bytes=states,proposal_optimizer_bytes=proposal_state,
            proposal_alpha=proposal_alpha,before_alpha=before_alpha,
            baseline_mle=baseline,proposal_mle=newmle,proposal_mean_kl=newkl,
            accepted=accepted,chosen_mean_kl=chosenkl,
            proposal_weights=proposal,actor_fit_attempts=self.actor_fit_attempts,
            actor_batches=self.actor_batches,
            actor_changes={k:float(np.linalg.norm(np.asarray(v)-before[k])) for k,v in self.actor.items()})

    def value_update(self,x,returns,rng):
        x=np.asarray(x,dtype=np.float64);returns=np.asarray(returns,dtype=np.float64)
        if x.ndim!=2 or not len(x):raise ValueError('Nonempty complete episode batch required')
        finite(x,(len(x),101));finite(returns,(len(x),))
        indices=[];losses=[];norms=[];gradients=[];states=[]
        before={k:np.asarray(v).copy() for k,v in self.value.items()}
        from jax.flatten_util import ravel_pytree
        for _ in range(VALUE_UPDATES):
            ix=rng.integers(0,len(x),size=512)
            loss,g=self.value_gradient(self.value,self.jp.asarray(x[ix]),self.jp.asarray(returns[ix]))
            flat=np.asarray(ravel_pytree(g)[0])
            if not np.isfinite(float(loss)) or not np.isfinite(flat).all():
                raise ArithmeticError('Nonfinite value update')
            indices.append(ix);losses.append(float(loss));norms.append(float(self.optax.global_norm(g)))
            gradients.append(flat.copy())
            update,self.value_state=self.value_optimizer.update(g,self.value_state,self.value)
            self.value=self.optax.apply_updates(self.value,update);self.value_updates+=1
            states.append(self.serialization.to_bytes(self.value_state))
        return dict(indices=np.asarray(indices),losses=np.asarray(losses),norms=np.asarray(norms),
            gradients=np.stack(gradients),optimizer_step_bytes=states,
            last_gradient={k:np.asarray(v) for k,v in g.items()},
            value_update_count=self.value_updates,
            value_changes={k:float(np.linalg.norm(np.asarray(v)-before[k])) for k,v in self.value.items()})

    def numpy_weights(self):
        return {k:np.asarray(v).copy() for k,v in {**self.actor,**self.value}.items()}

    def state_bytes(self):
        return self.serialization.to_bytes(dict(actor=self.actor,value=self.value,
            actor_state=self.actor_state,value_state=self.value_state,alpha=self.alpha,
            actor_fit_attempts=self.actor_fit_attempts,actor_batches=self.actor_batches,
            value_updates=self.value_updates))

    def checkpoint(self,path):
        np.savez_compressed(path.with_suffix('.npz'),**self.numpy_weights())
        path.with_suffix('.learner.msgpack').write_bytes(self.state_bytes())

    @staticmethod
    def save_actor_evidence(path,e):
        np.savez_compressed(path,old_theta=e['old_theta'],old_mean=e['old_mean'],
            top_indices=e['e_step']['indices'],weights=e['e_step']['weights'],
            eta_trace=e['e_step']['trace'],gradients=e['gradients'],
            proposed_parameters=e['parameters'],scalars=e['scalars'])
