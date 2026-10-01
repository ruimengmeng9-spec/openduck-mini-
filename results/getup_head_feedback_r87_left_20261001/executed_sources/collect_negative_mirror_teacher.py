"""Collect simulated teacher actions for negative turn commands.

The teacher is a diagnostic blend of the original v5 actor and its reflected
positive-turn response. Data is simulation-only and not a hardware policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from diagnostics.test_turn_mirror_policy import mirror_joint, mirror_obs
from diagnostics.negative_turn_residual import ResidualNegativePolicy
from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
OFFICIAL = ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
DEFAULT_MODEL = ROOT / "training/official_seed_turn_balance_v5_yaw_error/final.onnx"


class RecordingTeacher:
    def __init__(
        self, model: Path, default: np.ndarray, blend: float,
        student_residual: Path | None, student_band: str,
        negative_speed_min: float, negative_speed_max: float,
    ):
        self.policy = OnnxInfer(str(model), awd=True)
        self.student = (
            ResidualNegativePolicy(
                self.policy, student_residual, command_band=student_band
            )
            if student_residual else None
        )
        self.default = np.asarray(default, dtype=np.float32)
        self.blend = blend
        self.negative_speed_min = negative_speed_min
        self.negative_speed_max = negative_speed_max
        self.observations: list[np.ndarray] = []
        self.base_actions: list[np.ndarray] = []
        self.teacher_actions: list[np.ndarray] = []

    def infer(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        base = np.asarray(self.policy.infer(obs), dtype=np.float32)
        speed = -float(obs[8])
        if not self.negative_speed_min - 1.0e-5 <= speed <= self.negative_speed_max + 1.0e-5:
            return base
        reflected = mirror_joint(
            np.asarray(
                self.policy.infer(mirror_obs(obs, self.default, flip_phase=False)),
                dtype=np.float32,
            )
        )
        teacher = np.clip(
            (1.0 - self.blend) * base + self.blend * reflected, -1.0, 1.0
        ).astype(np.float32)
        self.observations.append(obs.copy())
        self.base_actions.append(base.copy())
        self.teacher_actions.append(teacher.copy())
        return self.student.infer(obs) if self.student else teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=24)
    parser.add_argument("--blend", type=float, default=0.87)
    parser.add_argument("--qvel-noise", type=float, default=0.02)
    parser.add_argument("--student-residual", type=Path)
    parser.add_argument("--student-band", choices=("narrow", "wide", "extended"), default="narrow")
    parser.add_argument("--negative-speed-min", type=float, default=0.15)
    parser.add_argument("--negative-speed-max", type=float, default=0.15)
    parser.add_argument("--mass-scale-min", type=float, default=1.0)
    parser.add_argument("--mass-scale-max", type=float, default=1.0)
    args = parser.parse_args()
    if not 0.0 <= args.blend <= 1.0 or args.seeds < 1:
        parser.error("blend must be in [0,1] and seeds must be positive")
    if not 0.10 <= args.negative_speed_min <= args.negative_speed_max <= 0.18:
        parser.error("negative speeds must satisfy 0.10 <= min <= max <= 0.18")
    if not 0.5 <= args.mass_scale_min <= args.mass_scale_max <= 1.5:
        parser.error("mass scales must satisfy 0.5 <= min <= max <= 1.5")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_obs = []
    all_base = []
    all_teacher = []
    all_seed = []
    summaries = []
    zero = [0.0] * 7
    walk = [0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    positive = [0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0]
    prefixes = (
        [("stand", zero, 3.0)],
        [("stand", zero, 3.0), ("walk", walk, 5.0), ("stop", zero, 3.0)],
        [("stand", zero, 3.0), ("positive", positive, 5.0), ("stop", zero, 3.0)],
        [
            ("stand", zero, 3.0), ("walk", walk, 5.0),
            ("stop", zero, 3.0), ("positive", positive, 5.0),
            ("stop", zero, 3.0),
        ],
    )
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        rng = np.random.default_rng(seed)
        mass_scale = (
            args.mass_scale_min
            if args.mass_scale_min == args.mass_scale_max
            else float(rng.uniform(args.mass_scale_min, args.mass_scale_max))
        )
        simulation = DuckSimulation(
            REPO, OFFICIAL, output_root=args.output.parent,
            warmup_s=0.0 if mass_scale != 1.0 else 3.0,
        )
        sim = simulation.sim
        if mass_scale != 1.0:
            sim.model.body_mass[1:] *= mass_scale
            sim.model.body_inertia[1:] *= mass_scale
            mujoco.mj_setConst(sim.model, sim.data)
            simulation.reset()
            simulation.stand(3.0, source="mass_scaled_warmup")
        teacher = RecordingTeacher(
            args.model, sim.default_actuator, args.blend,
            args.student_residual, args.student_band,
            args.negative_speed_min, args.negative_speed_max,
        )
        sim.policy = teacher
        speed = rng.uniform(args.negative_speed_min, args.negative_speed_max)
        negative = [0.0, 0.0, -float(speed), 0.0, 0.0, 0.0, 0.0]
        sim.data.qvel[:] += rng.uniform(
            -args.qvel_noise, args.qvel_noise, size=sim.model.nv
        )
        mujoco.mj_forward(sim.model, sim.data)
        prefix = prefixes[seed % len(prefixes)]
        failed_prefix = False
        for name, command, duration in prefix:
            result = simulation._run_command(name, command, duration_s=duration, source="mirror_dataset")
            if result.get("fallen"):
                failed_prefix = True
                break
        if failed_prefix:
            summaries.append({"seed": seed, "fallen_before_negative": True})
            continue
        result = simulation._run_command(
            "turn", negative, duration_s=5.0, source="mirror_dataset"
        )
        count = len(teacher.observations)
        if count:
            all_obs.extend(teacher.observations)
            all_base.extend(teacher.base_actions)
            all_teacher.extend(teacher.teacher_actions)
            all_seed.extend([seed] * count)
        summaries.append({
            "seed": seed,
            "negative_command_rad_s": negative[2],
            "mass_scale": mass_scale,
            "prefix": [name for name, _, _ in prefix],
            "negative_steps": count,
            "fallen": bool(result.get("fallen")),
        })
        print(json.dumps(summaries[-1]), flush=True)

    if not all_obs:
        raise RuntimeError("No teacher actions collected")
    np.savez_compressed(
        args.output,
        obs=np.stack(all_obs).astype(np.float32),
        base=np.stack(all_base).astype(np.float32),
        teacher=np.stack(all_teacher).astype(np.float32),
        seed=np.asarray(all_seed, dtype=np.int32),
    )
    args.output.with_suffix(".json").write_text(
        json.dumps({
            "model": str(args.model), "blend": args.blend,
            "student_residual": str(args.student_residual) if args.student_residual else None,
            "student_band": args.student_band,
            "negative_speed_range": [args.negative_speed_min, args.negative_speed_max],
            "mass_scale_range": [args.mass_scale_min, args.mass_scale_max],
            "qvel_noise": args.qvel_noise, "samples": len(all_obs),
            "summaries": summaries,
        }, indent=2), encoding="utf-8"
    )
    print("SAVED", args.output, len(all_obs), flush=True)


if __name__ == "__main__":
    main()
