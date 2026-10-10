"""Fresh total-leg-target actor/critic, not an additive R157 residual."""
import numpy as np

SEED = 296
BOUND = .18
SIGMA = 1e-4
GAMMA = .99 ** .2
HORIZON = 529
INPUTS = 101
LEARNING_RATE = 1e-6
EPOCHS = 4


def finite(value, shape):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError('Finite causal arrays with the declared shape required')
    return a


def features(current, initial, control):
    if not isinstance(control, (int, np.integer)) or not 0 <= control <= 2279:
        raise ValueError('Original control clock required')
    a = finite(current, (50,)); b = finite(initial, (50,))
    return np.r_[np.tanh(a), np.tanh(b), min(control / 528., 1.)]


def initial_weights():
    rng = np.random.default_rng(SEED)
    weights = {}
    for name, last in [('actor', 10), ('critic', 1)]:
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
    return -.5 * xp.sum(((z - mean) / SIGMA) ** 2 + 2 * np.log(SIGMA)
                       + np.log(2 * np.pi), axis=-1)


def replace_legs(fixed, reference, latent, legs, lower, upper, active):
    """REPLACE the ten leg corrections; do not add to old feedback."""
    if not active:
        return fixed
    z = finite(latent, (10,))
    adjusted = np.asarray(fixed, dtype=float).copy()
    adjusted[legs] = np.clip(np.asarray(reference)[legs] + BOUND * np.tanh(z),
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


class Learner:
    def __init__(self, weights):
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jp
        import optax
        from flax import serialization
        self.jax, self.jp, self.optax = jax, jp, optax
        self.serialization = serialization
        self.weights = {k: jp.asarray(v) for k, v in weights.items()}
        self.optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(LEARNING_RATE))
        self.state = self.optimizer.init(self.weights)

        def objective(p, x, z, oldlog, advantage, returns, mask):
            mean = network(p, x, 'actor', jp)
            logp = log_probability(z, mean, jp)
            ratio = jp.exp(jp.clip(logp - oldlog, -40., 40.))
            clipped = jp.clip(ratio, .8, 1.2)
            policy = -jp.sum(mask * jp.minimum(ratio * advantage, clipped * advantage)) / jp.maximum(jp.sum(mask), 1.)
            value = jp.mean((network(p, x, 'critic', jp)[:, 0] - returns) ** 2)
            kl = jp.sum(mask * (oldlog - logp)) / jp.maximum(jp.sum(mask), 1.)
            return policy + .5 * value, jp.array([policy, value, kl])
        self.objective = objective
        self.gradient = jax.jit(jax.value_and_grad(objective, has_aux=True))

    def update(self, arrays, rng):
        stats = []; norms = []; epochs = 0
        before = {k: np.asarray(v).copy() for k, v in self.weights.items()}
        last_grad = None
        for epoch in range(EPOCHS):
            order = rng.permutation(len(arrays[0]))
            for start in range(0, len(order), 512):
                ix = order[start:start + 512]
                (_, s), grad = self.gradient(self.weights, *[self.jp.asarray(a[ix]) for a in arrays])
                assert all(np.isfinite(np.asarray(v)).all() for v in self.jax.tree.leaves(grad))
                norms.append(float(self.optax.global_norm(grad))); last_grad = grad
                updates, self.state = self.optimizer.update(grad, self.state, self.weights)
                self.weights = self.optax.apply_updates(self.weights, updates)
                stats.append(np.asarray(s))
            epochs += 1
            # Full sampled batch KL, not a biased unweighted minibatch average.
            _, full = self.objective(self.weights, *[self.jp.asarray(a) for a in arrays])
            if float(full[2]) > .02:
                break
        after = self.numpy_weights()
        assert all(np.isfinite(v).all() for v in after.values())
        return dict(epochs=epochs,updates=len(stats),statistics=np.asarray(stats),
                    gradient_norms=np.asarray(norms),
                    last_gradient={k:np.asarray(v) for k,v in last_grad.items()},
                    changes={k:float(np.linalg.norm(after[k]-before[k])) for k in before})

    def numpy_weights(self):
        return {k:np.asarray(v).copy() for k,v in self.weights.items()}

    def checkpoint(self, path):
        np.savez_compressed(path.with_suffix('.npz'), **self.numpy_weights())
        path.with_suffix('.learner.msgpack').write_bytes(self.serialization.to_bytes(dict(weights=self.weights,state=self.state)))
