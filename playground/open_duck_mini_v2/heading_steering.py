"""Simulation-only, frozen R2 gait plus a bounded two-axis steering actor.

Contract: action is chosen before physics; observations contain the most recent
three motor actions. Only this opt-in environment uses the new timing contract.
"""

import hashlib
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco.mjx as mjx
import numpy as np
import onnx
from onnx import numpy_helper

from .focused_skill import FocusedSkill
from .reference_residual import ReferenceResidualBackward


def yaw_and_features(q, target_yaw, time_s, ramp_s, xp=np):
    w, x, y, z = q[3:7]
    yaw = xp.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    error = yaw - target_yaw
    gravity = xp.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)])
    ramp = xp.clip(time_s / ramp_s, 0., 1.) if ramp_s > 0 else xp.asarray(1.)
    return xp.concatenate([xp.stack([xp.sin(error), xp.cos(error)]), gravity, xp.reshape(ramp, (1,))])


def steering_delta(action, yaw_gain, roll_gain, xp=np):
    # Both hip-yaw axes have the same sign in this model. Both hip-roll axes
    # also share parity under the validated left/right reflection convention.
    a = xp.clip(action, -1., 1.)
    return xp.stack([yaw_gain*a[0], roll_gain*a[1], 0.*a[0], 0.*a[0], 0.*a[0],
                     0.*a[0], 0.*a[0], 0.*a[0], 0.*a[0], yaw_gain*a[0],
                     roll_gain*a[1], 0.*a[0], 0.*a[0], 0.*a[0]])


class FrozenOnnxActor:
    """Exact exported SiLU/tanh actor in JAX; weights never enter PPO params."""
    def __init__(self, path):
        self.sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        model = onnx.load(path)
        arrays = {x.name: numpy_helper.to_array(x).astype(np.float32) for x in model.graph.initializer}
        sub = next(x for x in model.graph.node if x.op_type == "Sub")
        mul = next(x for x in model.graph.node if x.op_type == "Mul" and x.input[0] == sub.output[0])
        self.mean = jp.asarray(arrays[sub.input[1]])
        self.inv_std = jp.asarray(arrays[mul.input[1]])
        self.layers = [(jp.asarray(arrays[x.input[1]]), jp.asarray(arrays[x.input[2]]))
                       for x in model.graph.node if x.op_type == "Gemm"]
        if [x.shape for x, _ in self.layers] != [(101, 512), (512, 256), (256, 128), (128, 28)]:
            raise ValueError("frozen baseline must be the verified 101-dimensional R2 actor")

    def infer(self, obs):
        value = (obs - self.mean) * self.inv_std
        for i, (kernel, bias) in enumerate(self.layers):
            value = value @ kernel + bias
            if i < len(self.layers) - 1:
                value = jax.nn.silu(value)
        return jp.tanh(value[:14])


class HeadingSteering(ReferenceResidualBackward):
    def __init__(self, *, baseline_path, yaw_gain=.05, roll_gain=.025,
                 initial_error=.15, **kwargs):
        if not 0 < yaw_gain <= .08 or not 0 <= roll_gain <= .04 or not 0 <= initial_error <= .5:
            raise ValueError("steering bounds exceed this simulation experiment's limits")
        self.baseline = FrozenOnnxActor(baseline_path)
        self.steering_yaw_gain = yaw_gain
        self.steering_roll_gain = roll_gain
        self.initial_error = initial_error
        super().__init__(**kwargs)

    @property
    def action_size(self):
        return 2

    def augment(self, state):
        info = state.info
        base_obs = state.obs["state"][:101]
        # Parent generates observations BEFORE shifting history. Rewrite ONLY
        # these blocks to expose the same fresh history as the native decoder.
        history = jp.concatenate([info["last_act"], info["last_last_act"], info["last_last_last_act"]])
        base_obs = base_obs.at[41:83].set(history)
        features = yaw_and_features(self.get_floating_base_qpos(state.data.qpos),
                                    info["initial_yaw"], state.data.time,
                                    self.reference_ramp_s, xp=jp)
        obs = dict(state.obs)
        obs["state"] = jp.concatenate([base_obs, features])
        privileged = obs["privileged_state"]
        # Only the parent 212 channels, not any previously appended features.
        obs["privileged_state"] = jp.concatenate([privileged[:212].at[:101].set(base_obs), features])
        return state.replace(obs=obs)

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info)
        info["rng"], key = jax.random.split(info["rng"])
        # Positive and negative relative goal offsets, on unchanged flat ground.
        info["initial_yaw"] += jax.random.uniform(key, (), minval=-self.initial_error, maxval=self.initial_error)
        info["imitation_i"] = jp.asarray(0)
        info["imitation_phase"] = jp.array([1., 0.])
        data = mjx.forward(self.mjx_model, state.data)
        contact = state.obs["state"][97:99].astype(bool)
        obs = self._get_obs(data, info, contact)
        return self.augment(state.replace(data=data, info=info, obs=obs))

    def step(self, state, correction):
        base_action = self.baseline.infer(state.obs["state"][:101])
        reference = self.reference_target(state.info["command"], state.info["imitation_i"], state.data.time)
        baseline_target = jp.clip(reference + self.residual_gain*base_action, self.target_lower, self.target_upper)
        target = jp.clip(baseline_target + steering_delta(correction, self.steering_yaw_gain,
                                                        self.steering_roll_gain, xp=jp),
                         self.target_lower, self.target_upper)
        action = (target-self._default_actuator) / self._config.action_scale
        result = FocusedSkill.step(self, state, action)
        return self.augment(result)


def zero_steering_networks(*args, **kwargs):
    """Start deterministic correction at zero, with small exploration noise."""
    from brax.training import networks
    from brax.training.agents.ppo import networks as ppo_networks
    from flax.core import freeze, unfreeze
    result = ppo_networks.make_ppo_networks(*args, **kwargs)
    original_init = result.policy_network.init

    def init(key):
        params = original_init(key)
        tree = unfreeze(params)
        layers = tree["params"]
        last = layers[f"hidden_{len(kwargs.get('policy_hidden_layer_sizes', (64,64)))}"]
        last["kernel"] = jp.zeros_like(last["kernel"])
        last["bias"] = jp.concatenate([jp.zeros(2), jp.full((2,), -2.)])
        return freeze(tree) if type(params).__name__ == "FrozenDict" else tree

    return result.replace(policy_network=networks.FeedForwardNetwork(init=init, apply=result.policy_network.apply))
