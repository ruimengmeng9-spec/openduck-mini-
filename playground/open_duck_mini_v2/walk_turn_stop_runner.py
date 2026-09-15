"""PPO runner for the focused Open Duck stand/walk/turn/stop task."""

from __future__ import annotations

import argparse
import functools
import os
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp

from playground.common.export_onnx import export_onnx
from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import walk_turn_stop


def install_single_device_jax_compatibility() -> None:
    """Restore the removed helper Brax 0.14 still calls on JAX 0.10.

    Training intentionally exposes exactly one GPU, so adding a leading replica
    axis and placing it on that device is the old helper's required behavior.
    """
    try:
        getattr(jax, "device_put_replicated")
        return
    except AttributeError:
        pass

    def device_put_replicated(value, devices):
        if len(devices) != 1:
            raise RuntimeError(
                "This compatibility path requires exactly one visible device"
            )
        device = devices[0]
        return jax.tree_util.tree_map(
            lambda leaf: jax.device_put(jp.expand_dims(leaf, axis=0), device),
            value,
        )

    setattr(jax, "device_put_replicated", device_put_replicated)


class WalkTurnStopRunner(BaseRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        config = walk_turn_stop.default_config()
        self.env_config = config
        self.env = walk_turn_stop.WalkTurnStop(task=args.task, config=config)
        self.eval_env = walk_turn_stop.WalkTurnStop(
            task=args.task,
            config=walk_turn_stop.default_config(),
        )
        self.randomizer = None if args.no_domain_randomization else randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(self.env.observation_size["state"][0])
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")

    def policy_params_fn(self, current_step, make_policy, params) -> None:
        """Save restartable parameters without repeatedly invoking TensorFlow."""
        del make_policy
        checkpointer = ocp.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(params)
        stamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        path = self.output_dir / f"{stamp}_{current_step}"
        print(f"Saving checkpoint (step: {current_step}): {path}")
        checkpointer.save(str(path), params, force=True, save_args=save_args)

    def train(self) -> None:
        install_single_device_jax_compatibility()
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )
        self.ppo_params.num_timesteps = self.args.num_timesteps
        self.ppo_params.num_envs = self.args.num_envs
        self.ppo_params.num_evals = self.args.num_evals
        self.ppo_params.num_eval_envs = self.args.num_eval_envs
        self.ppo_params.seed = self.args.seed
        self.ppo_training_params = dict(self.ppo_params)

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks,
                **self.ppo_params.network_factory,
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks

        print(f"PPO params: {self.ppo_training_params}")
        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )
        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )
        if not self.args.skip_onnx_export:
            output_path = self.output_dir / "final.onnx"
            print(f"Exporting final ONNX once: {output_path}")
            export_onnx(
                params,
                self.action_size,
                self.ppo_params,
                self.obs_size,
                output_path=str(output_path),
            )


def build_parser() -> argparse.ArgumentParser:
    root = Path(os.environ.get("OPEN_DUCK_ROOT", Path.home() / "open_duck"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default=str(root / "training/walk_turn_stop_v1"),
    )
    parser.add_argument("--num_timesteps", type=int, default=150_000_000)
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--num_evals", type=int, default=10)
    parser.add_argument("--num_eval_envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="flat_terrain")
    parser.add_argument("--restore_checkpoint_path", default=None)
    parser.add_argument("--no_domain_randomization", action="store_true")
    parser.add_argument("--skip_onnx_export", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    WalkTurnStopRunner(args).train()


if __name__ == "__main__":
    main()
