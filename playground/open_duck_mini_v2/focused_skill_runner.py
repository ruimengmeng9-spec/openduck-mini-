"""PPO runner for focused Open Duck Mini locomotion curricula."""

from __future__ import annotations

import argparse
import functools
from datetime import datetime
from pathlib import Path

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from flax.training import orbax_utils
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp

from playground.common import randomize
from playground.common.export_onnx import export_onnx
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import focused_skill
from playground.open_duck_mini_v2.walk_turn_stop_runner import (
    install_single_device_jax_compatibility,
)


class FocusedSkillRunner(BaseRunner):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.env_config = focused_skill.focused_config(args.skill)
        # Allow retargeting the command curriculum from the command line so a
        # sweep does not require editing (and re-reviewing) the skill definition.
        if args.vx_min is not None or args.vx_max is not None:
            lo = args.vx_min if args.vx_min is not None else self.env_config.lin_vel_x[0]
            hi = args.vx_max if args.vx_max is not None else self.env_config.lin_vel_x[1]
            self.env_config.lin_vel_x = [lo, hi]
            print(f"Velocity command overridden to [{lo}, {hi}]")
        self.env = focused_skill.FocusedSkill(
            skill=args.skill, task=args.task, config=self.env_config
        )
        eval_config = focused_skill.focused_config(args.skill)
        eval_config.lin_vel_x = list(self.env_config.lin_vel_x)
        self.eval_env = focused_skill.FocusedSkill(
            skill=args.skill,
            task=args.task,
            config=eval_config,
        )
        self.randomizer = (
            randomize.domain_randomize if args.domain_randomization else None
        )
        self.action_size = self.env.action_size
        self.obs_size = int(self.env.observation_size["state"][0])
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Skill: {args.skill}")
        print(f"Observation size: {self.obs_size}")

    def policy_params_fn(self, current_step, make_policy, params) -> None:
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
        self.ppo_params.learning_rate = self.args.learning_rate
        if self.args.skill == "backward":
            # A reverse lunge can remain upright for several seconds before
            # falling.  The default 0.97 discount makes that delayed failure
            # nearly irrelevant, so use a longer effective planning horizon.
            self.ppo_params.discounting = 0.995
            self.ppo_params.entropy_cost = 0.002
        elif self.args.skill == "getup":
            # Standing up is a multi-second manoeuvre: with the default 0.97
            # discount (~33 step horizon) the posture that finally pays off is
            # invisible from the floor, and the policy settles for lying still.
            self.ppo_params.discounting = 0.993
            self.ppo_params.entropy_cost = 0.003
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
    root = Path("/data/shijinsheng/open_duck")
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", choices=focused_skill.SUPPORTED_SKILLS, required=True)
    parser.add_argument("--output_dir", default=str(root / "training/focused_skill"))
    parser.add_argument("--num_timesteps", type=int, default=50_000_000)
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--num_evals", type=int, default=8)
    parser.add_argument("--num_eval_envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", default="flat_terrain")
    parser.add_argument("--restore_checkpoint_path", default=None)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--vx-min",
        type=float,
        default=None,
        help="Override the lower bound of the lin_vel_x command curriculum.",
    )
    parser.add_argument(
        "--vx-max",
        type=float,
        default=None,
        help="Override the upper bound of the lin_vel_x command curriculum.",
    )
    parser.add_argument("--domain_randomization", action="store_true")
    parser.add_argument("--skip_onnx_export", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    FocusedSkillRunner(args).train()


if __name__ == "__main__":
    main()
