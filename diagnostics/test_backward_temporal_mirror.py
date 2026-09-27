"""Test whether a forward policy can provide a safe backward teacher.

This is a diagnostic only. Time reversal of a walking gait is not a physical
symmetry: joint limits, feet, impacts and gravity are not reversed. Every
candidate is validated in closed-loop native MuJoCo before it can be used as
a distillation target.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
OFFICIAL = ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
FORWARD = ROOT / "training/official_seed_turn_balance_v5_yaw_error/final.onnx"
SAGITTAL = np.array([2, 3, 4, 11, 12, 13])
ACTION_FLIPS = {
    "flip_hip": np.array([2, 11]),
    "flip_knee": np.array([3, 12]),
    "flip_ankle": np.array([4, 13]),
    "flip_hip_knee": np.array([2, 3, 11, 12]),
    "flip_hip_ankle": np.array([2, 4, 11, 13]),
    "flip_knee_ankle": np.array([3, 4, 12, 13]),
    "flip_all": SAGITTAL,
}


class TemporalMirrorPolicy:
    def __init__(self, base: OnnxInfer, mode: str, flip_strength: float = 1.0,
                 pitch_degrees=None, guard_start_deg: float | None = None,
                 guard_full_deg: float = -15.0,
                 guard_strength: float = 0.3):
        self.base = base
        self.mode = mode
        self.flip_strength = flip_strength
        self.pitch_degrees = pitch_degrees
        self.guard_start_deg = guard_start_deg
        self.guard_full_deg = guard_full_deg
        self.guard_strength = guard_strength

    def infer(self, obs: np.ndarray) -> np.ndarray:
        if float(obs[6]) >= -0.001 or self.mode == "baseline":
            return np.asarray(self.base.infer(obs), dtype=np.float32)
        transformed = np.asarray(obs, dtype=np.float32).copy()
        if "command_flip" in self.mode:
            transformed[6] *= -1
        if "phase_reverse" in self.mode:
            transformed[100] *= -1
        action = np.asarray(self.base.infer(transformed), dtype=np.float32).copy()
        strength = self.flip_strength
        if self.guard_start_deg is not None:
            pitch = self.pitch_degrees()
            weight = np.clip(
                (pitch - self.guard_full_deg) /
                (self.guard_start_deg - self.guard_full_deg), 0.0, 1.0
            )
            strength = self.guard_strength + weight * (
                self.flip_strength - self.guard_strength
            )
        for marker, indices in ACTION_FLIPS.items():
            if self.mode.endswith(marker):
                action[indices] *= 1.0 - 2.0 * strength
                break
        if "sagittal_flip" in self.mode:
            action[SAGITTAL] *= 1.0 - 2.0 * strength
        return action


def yaw(qpos: np.ndarray) -> float:
    w, x, y, z = (float(v) for v in qpos[3:7])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_case(model: Path, output: Path, mode: str, speed: float,
             duration: float, seed: int, flip_strength: float,
             guard_start_deg: float | None, guard_full_deg: float,
             guard_strength: float) -> dict:
    simulation = DuckSimulation(REPO, OFFICIAL, output_root=output, warmup_s=3.0)
    sim = simulation.sim
    def current_pitch_degrees() -> float:
        q = sim.get_floating_base_qpos(sim.data.qpos)
        w, x, y, z = (float(v) for v in q[3:7])
        return math.degrees(math.asin(float(np.clip(
            2 * (w * y - z * x), -1.0, 1.0
        ))))

    sim.policy = TemporalMirrorPolicy(
        OnnxInfer(str(model), awd=True), mode, flip_strength,
        current_pitch_degrees, guard_start_deg, guard_full_deg, guard_strength,
    )
    rng = np.random.default_rng(seed)
    sim.data.qvel[:] += rng.uniform(-0.02, 0.02, size=sim.model.nv)
    mujoco.mj_forward(sim.model, sim.data)
    start = sim.get_floating_base_qpos(sim.data.qpos).copy()
    heading = yaw(start)
    elapsed = 0.0
    fallen = False
    up_min = 1.0
    while elapsed < duration - 1e-9:
        chunk = min(1.0, duration - elapsed)
        result = simulation._run_command(
            "backward", [speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            duration_s=chunk, source="temporal_mirror_diagnostic",
        )
        elapsed += float(result["executed_duration_s"])
        up_min = min(up_min, float(result["up_vector_z"]))
        fallen = bool(result["fallen"])
        if fallen:
            break
    end = sim.get_floating_base_qpos(sim.data.qpos)
    dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
    forward = math.cos(heading) * dx + math.sin(heading) * dy
    return {
        "mode": mode, "seed": seed, "command_mps": speed,
        "flip_strength": flip_strength,
        "guard_start_deg": guard_start_deg,
        "guard_full_deg": guard_full_deg if guard_start_deg is not None else None,
        "guard_strength": guard_strength if guard_start_deg is not None else None,
        "duration_s": round(elapsed, 4), "fallen": fallen,
        "forward_mps": round(forward / max(elapsed, 1e-6), 5),
        "forward_m": round(forward, 5), "min_up_z": round(up_min, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=FORWARD)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--speed", type=float, default=-0.074)
    parser.add_argument("--flip-strength", type=float, default=1.0)
    parser.add_argument("--guard-start-deg", type=float)
    parser.add_argument("--guard-full-deg", type=float, default=-15.0)
    parser.add_argument("--guard-strength", type=float, default=0.3)
    parser.add_argument("--modes", nargs="+", default=[
        "baseline", "command_flip", "phase_reverse",
        "command_flip_phase_reverse", "command_flip_sagittal_flip",
        "command_flip_phase_reverse_sagittal_flip",
    ])
    args = parser.parse_args()
    if not 0.0 <= args.flip_strength <= 1.0:
        parser.error("--flip-strength must be between 0 and 1")
    if args.guard_start_deg is not None and (
        args.guard_start_deg <= args.guard_full_deg or
        not 0.0 <= args.guard_strength <= args.flip_strength
    ):
        parser.error("invalid pitch guard configuration")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in args.modes:
        for seed in range(args.seeds):
            row = run_case(args.model, args.output_dir, mode, args.speed,
                           args.duration_s, seed, args.flip_strength,
                           args.guard_start_deg, args.guard_full_deg,
                           args.guard_strength)
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for mode in args.modes:
        selected = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "safe": sum(not r["fallen"] for r in selected),
            "total": len(selected),
            "mean_forward_mps": round(float(np.mean(
                [r["forward_mps"] for r in selected]
            )), 5),
        }
    result = {"model": str(args.model), "duration_s": args.duration_s,
              "flip_strength": args.flip_strength,
              "guard_start_deg": args.guard_start_deg,
              "guard_full_deg": args.guard_full_deg,
              "guard_strength": args.guard_strength,
              "command_mps": args.speed, "summary": summary, "rows": rows}
    (args.output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
