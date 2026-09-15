"""Roll out one PPO checkpoint inside its native JAX/MJX training environment."""

from __future__ import annotations

import argparse
import functools
import json

import jax
import jax.numpy as jp
import numpy as np
from brax.training import checkpoint
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import locomotion_params

from playground.open_duck_mini_v2.focused_skill import FocusedSkill, focused_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--speed", type=float, default=-0.06)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = FocusedSkill("backward", config=focused_config("backward"))
    ppo = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
    network_options = dict(ppo.network_factory)
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **network_options)
    networks = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    params = checkpoint.load(args.checkpoint)
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)

    rng = jax.random.PRNGKey(args.seed)
    state = env.reset(rng)
    state.info["command"] = jp.array([args.speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    start = np.asarray(env.get_floating_base_qpos(state.data.qpos)[:3], dtype=float)
    velocities = []
    actions = []
    scalar_rewards = []
    metric_history: dict[str, list[float]] = {}

    @jax.jit
    def one_step(current_state, key):
        action, _ = policy(current_state.obs, key)
        return env.step(current_state, action), action

    for index in range(args.steps):
        rng, action_rng = jax.random.split(rng)
        state, action = one_step(state, action_rng)
        state.info["command"] = jp.array(
            [args.speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        )
        velocities.append(np.asarray(env.get_local_linvel(state.data), dtype=float))
        actions.append(np.asarray(action, dtype=float))
        scalar_rewards.append(float(state.reward))
        for key, value in state.metrics.items():
            array = np.asarray(value)
            if array.ndim == 0:
                metric_history.setdefault(key, []).append(float(array))
        if float(state.done):
            break

    end = np.asarray(env.get_floating_base_qpos(state.data.qpos)[:3], dtype=float)
    velocity_array = np.asarray(velocities)
    action_array = np.asarray(actions)
    result = {
        "checkpoint": args.checkpoint,
        "command_mps": args.speed,
        "completed_steps": len(velocities),
        "simulation_seconds": float(state.data.time),
        "done": float(state.done),
        "world_displacement_xyz_m": (end - start).tolist(),
        "mean_local_velocity_mps": velocity_array.mean(axis=0).tolist(),
        "tail_mean_local_velocity_mps": velocity_array[-100:].mean(axis=0).tolist(),
        "action_min": float(action_array.min()),
        "action_max": float(action_array.max()),
        "final_gravity_z": float(env.get_gravity(state.data)[-1]),
        "mean_step_reward": float(np.mean(scalar_rewards)),
        "mean_metrics": {
            key: float(np.mean(values)) for key, values in sorted(metric_history.items())
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
