"""Long, perturbed native-MuJoCo validation for a backward ONNX policy."""

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
DEFAULT_MODEL = ROOT / "training/backward_residual_v22/final.onnx"
DEFAULT_OUTPUT = ROOT / "outputs/backward_v22_sustained_20260921"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=-0.074)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--qvel-noise", type=float, default=0.02)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--reset-phase-on-start", action="store_true")
    parser.add_argument("--no-motor-slew-limit", action="store_true")
    parser.add_argument("--base-model", type=Path, help="Optional balanced model for reverse blending")
    parser.add_argument("--reverse-weight", type=float, default=1.0)
    parser.add_argument("--pitch-guard-full-reverse-deg", type=float)
    parser.add_argument("--pitch-guard-full-guard-deg", type=float, default=-18.0)
    parser.add_argument(
        "--heading-kp",
        type=float,
        default=0.0,
        help="Optional proportional heading hold gain (yaw command / yaw error).",
    )
    parser.add_argument("--max-yaw-command", type=float, default=0.3)
    parser.add_argument("--yaw-command", type=float, default=0.0)
    parser.add_argument("--control-period-s", type=float, default=0.2)
    args = parser.parse_args()
    if not 0.0 <= args.reverse_weight <= 1.0:
        parser.error("--reverse-weight must be in [0, 1]")
    if args.reverse_weight != 1.0 and not args.base_model:
        parser.error("--base-model is required when --reverse-weight is not 1")
    if args.pitch_guard_full_reverse_deg is not None and not args.base_model:
        parser.error("--base-model is required for pitch guarding")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        simulation = DuckSimulation(
            REPO, OFFICIAL, output_root=args.output_dir, warmup_s=args.warmup_s
        )
        simulation.sim.policy = OnnxInfer(str(args.model), awd=True)
        if args.base_model:
            from diagnostics.backward_blend_policy import (
                BlendedBackwardPolicy, PitchGuardBackwardPolicy,
            )

            base = OnnxInfer(str(args.base_model), awd=True)
            if args.pitch_guard_full_reverse_deg is None:
                simulation.sim.policy = BlendedBackwardPolicy(
                    base, simulation.sim.policy, args.reverse_weight,
                )
            else:
                def current_pitch_degrees() -> float:
                    q = simulation.sim.get_floating_base_qpos(simulation.sim.data.qpos)
                    w, x, y, z = (float(v) for v in q[3:7])
                    return math.degrees(math.asin(float(np.clip(
                        2 * (w * y - z * x), -1.0, 1.0
                    ))))

                simulation.sim.policy = PitchGuardBackwardPolicy(
                    base, simulation.sim.policy, current_pitch_degrees,
                    args.pitch_guard_full_reverse_deg,
                    args.pitch_guard_full_guard_deg,
                    args.reverse_weight,
                )
        simulation._active_policy_name = args.model.parent.name
        if args.reset_phase_on_start:
            simulation.sim.imitation_i = 0
            simulation.sim.imitation_phase = np.array([1.0, 0.0], dtype=np.float32)
        if args.no_motor_slew_limit:
            simulation.sim.max_motor_velocity = float("inf")
        simulation.sim.data.qvel[:] += rng.uniform(
            -args.qvel_noise, args.qvel_noise, size=simulation.sim.model.nv
        )
        mujoco.mj_forward(simulation.sim.model, simulation.sim.data)
        start = simulation.sim.get_floating_base_qpos(
            simulation.sim.data.qpos
        ).copy()
        qw, qx, qy, qz = (float(x) for x in start[3:7])
        yaw = math.atan2(
            2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)
        )
        elapsed = 0.0
        fallen = False
        final_result = None
        while elapsed < args.duration_s - 1e-9:
            chunk = min(
                args.control_period_s if args.heading_kp else 5.0,
                args.duration_s - elapsed,
            )
            current = simulation.sim.get_floating_base_qpos(
                simulation.sim.data.qpos
            )
            cw, cx, cy, cz = (float(x) for x in current[3:7])
            current_yaw = math.atan2(
                2 * (cw * cz + cx * cy), 1 - 2 * (cy * cy + cz * cz)
            )
            yaw_error = math.atan2(
                math.sin(current_yaw - yaw), math.cos(current_yaw - yaw)
            )
            yaw_command = args.yaw_command
            if args.heading_kp:
                yaw_command = float(
                    np.clip(
                        -args.heading_kp * yaw_error,
                        -args.max_yaw_command,
                        args.max_yaw_command,
                    )
                )
            final_result = simulation._run_command(
                "backward",
                [args.speed, 0.0, yaw_command, 0.0, 0.0, 0.0, 0.0],
                duration_s=chunk,
                source="sustained_native_validation",
            )
            elapsed += float(final_result["executed_duration_s"])
            fallen = bool(final_result.get("fallen"))
            if fallen or float(final_result["executed_duration_s"]) < chunk - 1e-3:
                break
        assert final_result is not None
        end = simulation.sim.get_floating_base_qpos(simulation.sim.data.qpos)
        ew, ex, ey, ez = (float(x) for x in end[3:7])
        end_yaw = math.atan2(
            2 * (ew * ez + ex * ey), 1 - 2 * (ey * ey + ez * ez)
        )
        heading_change = math.atan2(
            math.sin(end_yaw - yaw), math.cos(end_yaw - yaw)
        )
        dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        row = {
            "seed": seed,
            "model": str(args.model),
            "base_model": str(args.base_model) if args.base_model else None,
            "reverse_weight": args.reverse_weight,
            "pitch_guard_full_reverse_deg": args.pitch_guard_full_reverse_deg,
            "pitch_guard_full_guard_deg": args.pitch_guard_full_guard_deg,
            "command_mps": args.speed,
            "qvel_noise": args.qvel_noise,
            "warmup_s": args.warmup_s,
            "reset_phase_on_start": args.reset_phase_on_start,
            "motor_slew_limit_enabled": not args.no_motor_slew_limit,
            "heading_kp": args.heading_kp,
            "executed_duration_s": round(elapsed, 6),
            "fallen": fallen,
            "forward_displacement_m": round(forward, 6),
            "lateral_displacement_m": round(lateral, 6),
            "mean_forward_speed_mps": round(forward / max(elapsed, 1e-6), 6),
            "heading_change_deg": round(math.degrees(heading_change), 3),
            "up_vector_z": final_result.get("up_vector_z"),
        }
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    summary = {
        "completed": sum(not row["fallen"] for row in rows),
        "total": len(rows),
        "mean_forward_speed_mps": round(
            float(np.mean([row["mean_forward_speed_mps"] for row in rows])), 6
        ),
        "mean_abs_lateral_displacement_m": round(
            float(np.mean([abs(row["lateral_displacement_m"]) for row in rows])), 6
        ),
        "rows": rows,
    }
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
