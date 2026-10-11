"""Fresh finite natural-gradient/mean-KL actor update; no physical-safety claim."""
import numpy as np

SEED = 301
BOUND = .18
SIGMA = 1e-4
GAMMA = .99 ** .2
MAX_KL = .001
DAMPING = .01
CG_CAP = 10
CG_STOP = 1e-20
BACKTRACK_CAP = 10
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


def conjugate_gradient(operator, gradient):
    """Bounded deterministic solve with complete decision trace."""
    g = np.asarray(gradient, dtype=np.float64)
    if g.ndim != 1 or not np.isfinite(g).all():
        raise ValueError('Finite gradient vector required')
    direction = np.zeros_like(g); residual = g.copy(); search = g.copy()
    rr = float(residual @ residual); trace = []
    for iteration in range(CG_CAP):
        if rr <= CG_STOP:
            break
        product = finite(operator(search), g.shape)
        curvature = float(search @ product)
        if curvature <= 0:
            raise ArithmeticError('Nonpositive damped Fisher curvature')
        alpha = rr / curvature
        previous = residual.copy()
        direction = direction + alpha * search
        residual = residual - alpha * product
        next_rr = float(residual @ residual)
        beta = next_rr / rr
        trace.append(dict(iteration=iteration,search=search.copy(),
                          product=product.copy(),previous_residual=previous,
                          residual=residual.copy(),direction=direction.copy(),
                          rr=rr,curvature=curvature,alpha=alpha,
                          next_rr=next_rr,beta=beta))
        search = residual + beta * search; rr = next_rr
    return direction, trace


def choose_backtrack(theta, fullstep, gradient, evaluate):
    baseline_surrogate, baseline_kl = evaluate(theta)
    if not np.isfinite(baseline_surrogate) or baseline_kl != 0:
        raise ArithmeticError('Exact unchanged policy baseline required')
    predicted = float(gradient @ fullstep)
    trials = []; accepted = None; chosen = theta.copy()
    if predicted > 0 and np.isfinite(predicted):
        for j in range(BACKTRACK_CAP):
            fraction = .5 ** j; candidate = theta + fraction * fullstep
            surrogate, kl = evaluate(candidate)
            actual = surrogate - baseline_surrogate
            good = bool(np.isfinite(surrogate) and np.isfinite(kl) and
                        0 <= kl <= MAX_KL and actual > 0 and
                        actual >= .1 * fraction * predicted)
            trials.append(dict(index=j,fraction=fraction,parameters=candidate.copy(),
                               surrogate=float(surrogate),mean_kl=float(kl),
                               actual_gain=float(actual),
                               predicted_gain=fraction*predicted,accepted=good))
            if good:
                accepted = j; chosen = candidate; break
    return chosen, dict(baseline_surrogate=float(baseline_surrogate),
                        baseline_mean_kl=float(baseline_kl),
                        predicted_gain=predicted,accepted_index=accepted,trials=trials)


class Learner:
    def __init__(self, weights):
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jp
        import optax
        from jax.flatten_util import ravel_pytree
        from flax import serialization
        self.jax, self.jp, self.optax = jax, jp, optax
        self.serialization = serialization
        self.actor = {k: jp.asarray(v) for k,v in weights.items() if k.startswith('actor_')}
        self.value = {k: jp.asarray(v) for k,v in weights.items() if k.startswith('value_')}
        self.theta, self.unravel = ravel_pytree(self.actor)
        self.actor_updates = 0; self.value_updates = 0
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.),optax.adam(VALUE_LR))
        self.state = self.optimizer.init(self.value)
        self.flat_mean = jax.jit(lambda t,x: network(self.unravel(t),x,'actor',jp))

        def surrogate(t,x,z,oldlog,advantage):
            logp = log_probability(z,self.flat_mean(t,x),jp)
            # No clipping of ratio or rewards. Nonfinite trials are rejected.
            return jp.mean(jp.exp(logp-oldlog)*advantage)
        self.surrogate = jax.jit(surrogate)
        self.policy_gradient = jax.jit(jax.grad(surrogate))
        def kl(t,x,oldmean):
            return exact_mean_kl(oldmean,self.flat_mean(t,x),jp)
        self.kl = jax.jit(kl)
        kl_gradient = jax.grad(kl)
        self.fisher = jax.jit(lambda t,v,x,oldmean:
            jax.jvp(lambda q:kl_gradient(q,x,oldmean),(t,),(v,))[1])
        def value_loss(p,x,returns):
            return jp.mean((network(p,x,'value',jp)[:,0]-returns)**2)
        self.value_gradient = jax.jit(jax.value_and_grad(value_loss))

    def actor_update(self, x, z, advantage):
        x = np.asarray(x,dtype=np.float64); z = np.asarray(z,dtype=np.float64)
        advantage = np.asarray(advantage,dtype=np.float64)
        if x.ndim!=2 or not len(x):
            raise ValueError('Nonempty active-only batch required')
        finite(x,(len(x),101));finite(z,(len(x),10));finite(advantage,(len(x),))
        xp,zp,ap = map(self.jp.asarray,(x,z,advantage))
        theta = np.asarray(self.theta).copy()
        oldmean = self.flat_mean(self.theta,xp)
        oldlog = log_probability(zp,oldmean,self.jp)
        g = finite(self.policy_gradient(self.theta,xp,zp,oldlog,ap),theta.shape)
        def product(v):
            return np.asarray(self.fisher(self.jp.asarray(theta),self.jp.asarray(v),
                                          xp,oldmean)) + DAMPING*v
        d, trace = conjugate_gradient(product,g)
        damped_product = product(d); curvature = float(d @ damped_product)
        if curvature < 0 or not np.isfinite(curvature):
            raise ArithmeticError('Invalid final damped curvature')
        fullstep = d*np.sqrt(2*MAX_KL/curvature) if curvature>0 else np.zeros_like(d)
        def evaluate(t):
            t = self.jp.asarray(t)
            return (float(self.surrogate(t,xp,zp,oldlog,ap)),
                    float(self.kl(t,xp,oldmean)))
        chosen, line = choose_backtrack(theta,fullstep,g,evaluate)
        self.theta = self.jp.asarray(chosen); self.actor = self.unravel(self.theta)
        self.actor_updates += 1
        return dict(old_theta=theta,old_mean=np.asarray(oldmean),
                    old_log_probability=np.asarray(oldlog),gradient=g,
                    cg_direction=d,cg_trace=trace,damped_product=damped_product,
                    damped_curvature=curvature,full_step=fullstep,
                    chosen_theta=chosen,line_search=line,
                    actor_update_count=self.actor_updates,
                    actor_changes={k:float(np.linalg.norm(np.asarray(self.actor[k])-
                                      np.asarray(self.unravel(self.jp.asarray(theta))[k])))
                                   for k in self.actor})

    def value_update(self, x, returns, rng):
        x=np.asarray(x,dtype=np.float64);returns=np.asarray(returns,dtype=np.float64)
        if x.ndim!=2 or not len(x):raise ValueError('Nonempty complete episode batch required')
        finite(x,(len(x),101));finite(returns,(len(x),))
        indices=[];losses=[];norms=[];before={k:np.asarray(v).copy() for k,v in self.value.items()}
        for _ in range(VALUE_UPDATES):
            ix=rng.integers(0,len(x),size=512)
            loss,grad=self.value_gradient(self.value,self.jp.asarray(x[ix]),
                                          self.jp.asarray(returns[ix]))
            if not np.isfinite(float(loss)) or not all(np.isfinite(np.asarray(v)).all()
                    for v in self.jax.tree.leaves(grad)):
                raise ArithmeticError('Nonfinite value update')
            indices.append(ix);losses.append(float(loss));norms.append(float(self.optax.global_norm(grad)))
            updates,self.state=self.optimizer.update(grad,self.state,self.value)
            self.value=self.optax.apply_updates(self.value,updates);self.value_updates+=1
        return dict(indices=np.asarray(indices),losses=np.asarray(losses),norms=np.asarray(norms),
                    last_gradient={k:np.asarray(v) for k,v in grad.items()},
                    value_update_count=self.value_updates,
                    value_changes={k:float(np.linalg.norm(np.asarray(v)-before[k])) for k,v in self.value.items()})

    def numpy_weights(self):
        return {k:np.asarray(v).copy() for k,v in {**self.actor,**self.value}.items()}

    def state_bytes(self):
        return self.serialization.to_bytes(dict(actor=self.actor,value=self.value,
                  state=self.state,actor_updates=self.actor_updates,value_updates=self.value_updates))

    def checkpoint(self,path):
        np.savez_compressed(path.with_suffix('.npz'),**self.numpy_weights())
        path.with_suffix('.learner.msgpack').write_bytes(self.state_bytes())

    @staticmethod
    def save_actor_evidence(path,evidence):
        arrays={k:evidence[k] for k in ('old_theta','old_mean','old_log_probability',
                'gradient','cg_direction','damped_product','full_step','chosen_theta')}
        trace=evidence['cg_trace']
        n=len(evidence['old_theta'])
        for key in ('search','product','previous_residual','residual','direction'):
            arrays['cg_'+key]=np.stack([t[key] for t in trace]) if trace else np.empty((0,n))
        arrays['cg_scalars']=np.asarray([[t[k] for k in ('rr','curvature','alpha','next_rr','beta')]
                                        for t in trace]).reshape(-1,5)
        trials=evidence['line_search']['trials']
        arrays['trial_parameters']=np.stack([t['parameters'] for t in trials]) if trials else np.empty((0,n))
        arrays['trial_scalars']=np.asarray([[t[k] for k in ('fraction','surrogate','mean_kl',
                      'actual_gain','predicted_gain','accepted')] for t in trials]).reshape(-1,6)
        np.savez_compressed(path,**arrays)
