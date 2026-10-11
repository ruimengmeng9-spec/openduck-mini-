"""Read saved arrays and evaluate unchanged-policy arithmetic only; no updates or dynamics."""
import os
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:6])
import hashlib
import json
from pathlib import Path
import numpy as np
from diagnostics import total_target_trpo_kernel_r201 as kernel

root = Path('/data/shijinsheng/open_duck/outputs/getup_total_target_trpo_r201_left_20261011')
paths = [root/'checkpoint_0001.npz', root/'checkpoint_0001.learner.msgpack',
         root/'training_0002/complete_original_MC_batch.npz', Path(kernel.__file__)]
def hashes():
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
before = hashes()
with np.load(paths[0], allow_pickle=False) as saved:
    weights = {k: saved[k].copy() for k in saved.files}
with np.load(paths[2], allow_pickle=False) as saved:
    batch = {k: saved[k].copy() for k in saved.files}
learner = kernel.Learner(weights)
jax, jp = learner.jax, learner.jp
active = batch['active']
x, z, adv = (jp.asarray(batch[k][active]) for k in ('features', 'latents', 'normalized_advantage'))
theta = jp.asarray(np.asarray(learner.theta).copy())
oldmean = learner.flat_mean(learner.theta, x)
oldlog = kernel.log_probability(z, oldmean, jp)
surrogate = float(learner.surrogate(theta, x, z, oldlog, adv))
kl = float(learner.kl(theta, x, oldmean))
recomputed_mean = learner.flat_mean(theta, x)
materialized_kl = float(kernel.exact_mean_kl(oldmean, recomputed_mean, jp))
# Independent implementation of the same actor, without calling kernel.network.
def independent_actor(p, features):
    h = jp.tanh(features @ p['actor_w0'] + p['actor_b0'])
    h = jp.tanh(h @ p['actor_w1'] + p['actor_b1'])
    return h @ p['actor_w2'] + p['actor_b2']
independent_mean = jax.jit(independent_actor)(learner.actor, x)
def independent_nested_kl(p, features, previous_mean):
    mean = independent_actor(p, features)
    return jp.mean(jp.sum(((previous_mean-mean)/1e-4)**2, axis=-1)/2)
independent_kl = float(jax.jit(independent_nested_kl)(learner.actor, x, independent_mean))
state = learner.serialization.msgpack_restore(paths[1].read_bytes())
result = dict(new_dynamic_attempts=0, actor_update_called=False, value_update_called=False,
    numeric_validator_rerun=False, environment_or_mujoco_used=False,
    phase='post-failure saved-data arithmetic inspection, not original runtime trace',
    complete_samples=len(batch['returns']), active_samples=int(active.sum()),
    batch_finite=all(np.isfinite(v).all() for v in batch.values()),
    checkpoint_actor_updates=int(state['actor_updates']), checkpoint_value_updates=int(state['value_updates']),
    theta_copy_exact=bool(np.array_equal(np.asarray(theta), np.asarray(learner.theta))),
    baseline_surrogate=surrogate, baseline_surrogate_finite=bool(np.isfinite(surrogate)),
    unchanged_nested_jit_mean_kl=kl, unchanged_materialized_mean_kl=materialized_kl,
    materialized_mean_max_difference=float(np.max(np.abs(np.asarray(oldmean)-np.asarray(recomputed_mean)))),
    independent_nested_jit_mean_kl=independent_kl,
    independent_mean_max_difference=float(np.max(np.abs(np.asarray(oldmean)-np.asarray(independent_mean)))),
    original_baseline_guard_would_raise=bool(not np.isfinite(surrogate) or kl != 0),
    production_source_modified=False, original_second_round_CG_trace_not_recovered=True,
    input_hashes_before=before, input_hashes_after=hashes())
assert result['input_hashes_before'] == result['input_hashes_after']
assert learner.actor_updates == 0 and learner.value_updates == 0
print(json.dumps(result), flush=True)
