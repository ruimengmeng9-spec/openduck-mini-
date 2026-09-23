"""Measure how often turn errors are hidden by the task's reward floor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
import onnxruntime as ort
from mujoco_playground._src.collision import geoms_colliding

from playground.open_duck_mini_v2 import walk_turn_stop


ROOT = Path("/data/shijinsheng/open_duck")
DEFAULT_MODEL = ROOT / "training/official_seed_turn_balance_v2/final.onnx"
DEFAULT_OUTPUT = ROOT / "outputs/turn_reward_clipping_20260923/results.json"


def force_command(env, state, command: list[float]):
    info = dict(state.info)
    info["command"] = jp.asarray(command)
    contact = jp.asarray(
        [
            geoms_colliding(state.data, geom_id, env._floor_geom_id)
            for geom_id in env._feet_geom_id
        ]
    )
    obs = env._get_obs(state.data, info, contact)
    return state.replace(info=info, obs=obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    # Reconstruct the reward configuration used by turn-balance v2, but remove
    # observation delays/noise so positive and negative commands are directly
    # comparable. A negative floor exposes the pre-clipping total reward.
    os.environ["TRACKING_ANG_VEL_SCALE"] = "12"
    os.environ["TURN_PROGRESS_SCALE"] = "20"
    os.environ["TURN_SHORTFALL_SCALE"] = "-20"
    os.environ["TURN_WRONG_WAY_SCALE"] = "-20"
    config = walk_turn_stop.default_config()
    config.noise_config.level = 0.0
    config.noise_config.action_min_delay = 0
    config.noise_config.action_max_delay = 1
    config.noise_config.imu_min_delay = 0
    config.noise_config.imu_max_delay = 1
    config.push_config.enable = False
    config.reward_config.reward_floor = -1.0e6
    env = walk_turn_stop.WalkTurnStop(config=config)
    step = jax.jit(env.step)
    reset = jax.jit(env.reset)

    session = ort.InferenceSession(
        str(args.model.resolve()), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    rows = []
    for yaw_command in (0.15, -0.15):
        state = reset(jax.random.PRNGKey(0))
        state = force_command(env, state, [0.0] * 7)
        for _ in range(150):
            observation = np.asarray(state.obs["state"], dtype=np.float32)[None]
            action = session.run(None, {input_name: observation})[0][0]
            state = step(state, jp.asarray(action))

        command = [0.0, 0.0, yaw_command, 0.0, 0.0, 0.0, 0.0]
        state = force_command(env, state, command)
        samples = []
        for index in range(250):
            observation = np.asarray(state.obs["state"], dtype=np.float32)[None]
            action = session.run(None, {input_name: observation})[0][0]
            state = step(state, jp.asarray(action))
            reward = float(state.reward)
            gyro_z = float(env.get_gyro(state.data)[2])
            phase = float(
                state.info["imitation_i"] / env.PRM.nb_steps_in_period
            )
            samples.append(
                {
                    "time_s": (index + 1) * env.dt,
                    "phase": phase,
                    "gyro_z": gyro_z,
                    "raw_reward": reward,
                    "wrong_way": gyro_z * yaw_command < 0.0,
                }
            )

        late = [sample for sample in samples if sample["time_s"] >= 1.0]
        wrong = [sample for sample in late if sample["wrong_way"]]
        burst = [
            sample for sample in late if 0.4 <= sample["phase"] < 0.7
        ]

        def negative_fraction(items: list[dict]) -> float:
            return float(np.mean([item["raw_reward"] < 0.0 for item in items]))

        def mean_reward(items: list[dict]) -> float:
            return float(np.mean([item["raw_reward"] for item in items]))

        lost_negative = sum(max(-item["raw_reward"], 0.0) for item in late)
        row = {
            "yaw_command_rad_s": yaw_command,
            "frames_after_1s": len(late),
            "wrong_way_frames": len(wrong),
            "wrong_way_fraction": round(len(wrong) / len(late), 6),
            "negative_raw_reward_fraction": round(negative_fraction(late), 6),
            "wrong_way_negative_reward_fraction": round(
                negative_fraction(wrong) if wrong else 0.0, 6
            ),
            "mean_raw_reward_correct_way": round(
                mean_reward([item for item in late if not item["wrong_way"]]), 6
            ),
            "mean_raw_reward_wrong_way": round(
                mean_reward(wrong) if wrong else 0.0, 6
            ),
            "phase_0p4_to_0p7_negative_reward_fraction": round(
                negative_fraction(burst), 6
            ),
            "negative_reward_removed_by_floor_sum": round(lost_negative, 6),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    payload = {
        "model": str(args.model),
        "reward_floor_in_training": 0.0,
        "diagnostic_reward_floor": -1.0e6,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"SAVED {args.output}")


if __name__ == "__main__":
    main()
