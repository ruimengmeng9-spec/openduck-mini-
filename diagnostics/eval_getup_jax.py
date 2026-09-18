"""Roll out a getup checkpoint in its own JAX/MJX env; report recovery metrics."""

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
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    env = FocusedSkill("getup", task="flat_terrain_getup", config=focused_config("getup"))
    ppo = locomotion_params.brax_ppo_config("BerkeleyHumanoidJoystickFlatTerrain")
    networks = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **dict(ppo.network_factory),
    )
    params = checkpoint.load(args.checkpoint)
    policy = ppo_networks.make_inference_fn(networks)(params, deterministic=True)

    @jax.jit
    def step(state, key):
        action, _ = policy(state.obs, key)
        return env.step(state, action), action

    rows = []
    for seed in args.seeds:
        rng = jax.random.PRNGKey(seed)
        state = env.reset(rng)
        up_z0 = float(env.get_gravity(state.data)[-1])
        h0 = float(env.get_floating_base_qpos(state.data.qpos)[2])
        best_up = up_z0
        best_h = h0
        stood = None
        for i in range(args.steps):
            rng, k = jax.random.split(rng)
            state, _ = step(state, k)
            gz = float(env.get_gravity(state.data)[-1])
            h = float(env.get_floating_base_qpos(state.data.qpos)[2])
            best_up = max(best_up, gz)
            best_h = max(best_h, h)
            if stood is None and gz > 0.85 and h > 0.12:
                stood = float(state.data.time)
        rows.append({
            "seed": seed,
            "start_up_z": round(up_z0, 4),
            "start_height_m": round(h0, 4),
            "peak_up_z": round(best_up, 4),
            "peak_height_m": round(best_h, 4),
            "final_up_z": round(float(env.get_gravity(state.data)[-1]), 4),
            "final_height_m": round(float(env.get_floating_base_qpos(state.data.qpos)[2]), 4),
            "stood_up": stood is not None,
            "time_to_stand_s": round(stood, 3) if stood else None,
        })
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    ok = sum(1 for r in rows if r["stood_up"])
    print(f"SUCCESS {ok}/{len(rows)}")


if __name__ == "__main__":
    main()
