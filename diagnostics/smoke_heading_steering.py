"""Parity, zero initialization and one compiled environment step."""
import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as ort
from playground.open_duck_mini_v2.heading_steering import FrozenOnnxActor, HeadingSteering, zero_steering_networks
from playground.open_duck_mini_v2.focused_skill import focused_config

path = '/data/shijinsheng/open_duck/training/backward_reference_residual_r2/final.onnx'
base = FrozenOnnxActor(path)
samples = np.random.default_rng(1).normal(size=(20,101)).astype(np.float32)
observations = np.asarray(base.mean)[None,:] + samples / np.maximum(np.asarray(base.inv_std),1e-5)[None,:]
session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
error = max(float(np.max(np.abs(np.asarray(base.infer(jp.asarray(obs)))-session.run(None, {'obs':obs[None,:]})[0][0]))) for obs in observations)
assert error < 1e-4, error
print('BASELINE ONNX/JAX MAX ERROR:',error,flush=True)
net = zero_steering_networks({'state':107,'privileged_state':218}, 2, policy_hidden_layer_sizes=(64,64), value_hidden_layer_sizes=(128,128),policy_obs_key='state',value_obs_key='privileged_state')
params = net.policy_network.init(jax.random.PRNGKey(1))
assert np.all(np.asarray(params['params']['hidden_2']['kernel']) == 0)
assert np.all(np.asarray(params['params']['hidden_2']['bias'][:2]) == 0)
print('ZERO CORRECTION INITIALIZATION: PASSED',flush=True)
cfg = focused_config('backward')
cfg.lin_vel_x = [-.074,-.074]
env = HeadingSteering(baseline_path=path,residual_gain=.12,ramp_s=1.,task='flat_terrain',config=cfg)
print('COMPILING RESET/STEP',flush=True)
state = jax.jit(env.reset)(jax.random.PRNGKey(86))
state = jax.jit(env.step)(state,jp.zeros(2))
state.data.qpos.block_until_ready()
assert env.action_size == 2
assert state.obs['state'].shape == (107,)
assert state.obs['privileged_state'].shape == (218,)
assert np.isfinite(np.asarray(state.obs['state'])).all()
np.testing.assert_allclose(np.asarray(state.obs['state'][41:55]),np.asarray(state.info['last_act']))
print('HEADING STEERING SMOKE: PASSED',float(state.reward),float(state.done),flush=True)
