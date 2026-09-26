"""Native-MuJoCo acceptance test for a hardware-oriented locomotion policy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer
from playground.open_duck_mini_v2 import constants


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
OFFICIAL = ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
DEFAULT_MODEL = ROOT / "training/walk_turn_stop_sim2real_v2/final.onnx"
DEFAULT_OUTPUT = ROOT / "outputs/walk_turn_stop_sim2real_v2_validation"

PHASES = (
    ("stand", 3.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    # Matches the first recommended hardware test: 20% of the 0.15 m/s stick.
    ("slow_walk", 5.0, [0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("stop_after_slow", 3.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("walk", 5.0, [0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("stop_after_walk", 3.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("turn_positive", 5.0, [0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0]),
    ("stop_after_positive", 3.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ("turn_negative", 5.0, [0.0, 0.0, -0.15, 0.0, 0.0, 0.0, 0.0]),
    ("final_stop", 3.0, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
)


def yaw_of(qpos: np.ndarray) -> float:
    qw, qx, qy, qz = (float(x) for x in qpos[3:7])
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--qvel-noise", type=float, default=0.02)
    parser.add_argument("--contact-friction", type=float)
    parser.add_argument(
        "--mirror-negative-phase", choices=("same", "flip"),
        help="Diagnostic only: mirror the policy for negative yaw commands",
    )
    parser.add_argument(
        "--mirror-negative-blend", type=float, default=1.0,
        help="Diagnostic only: weight of mirrored action versus original action",
    )
    parser.add_argument(
        "--negative-residual", type=Path,
        help="Diagnostic only: learned ONNX action residual for -0.15 rad/s",
    )
    parser.add_argument(
        "--negative-residual-band", choices=("narrow", "wide"), default="narrow",
    )
    parser.add_argument("--negative-yaw", type=float, default=-0.15)
    args = parser.parse_args()
    if args.contact_friction is not None and args.contact_friction <= 0:
        parser.error("--contact-friction must be positive")
    if not 0.0 <= args.mirror_negative_blend <= 1.0:
        parser.error("--mirror-negative-blend must be in [0, 1]")
    if args.negative_residual and args.mirror_negative_phase:
        parser.error("Choose either --negative-residual or --mirror-negative-phase")
    if not -0.3 <= args.negative_yaw < 0.0:
        parser.error("--negative-yaw must be in [-0.3, 0)")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        simulation = DuckSimulation(
            REPO, OFFICIAL, output_root=args.output_dir, warmup_s=3.0
        )
        simulation.sim.policy = OnnxInfer(str(args.model), awd=True)
        if args.negative_residual:
            from diagnostics.negative_turn_residual import ResidualNegativePolicy

            simulation.sim.policy = ResidualNegativePolicy(
                simulation.sim.policy, args.negative_residual,
                command_band=args.negative_residual_band,
            )
        if args.mirror_negative_phase:
            from diagnostics.test_turn_mirror_policy import MirrorNegativePolicy

            simulation.sim.policy = MirrorNegativePolicy(
                simulation.sim.policy,
                simulation.sim.default_actuator,
                flip_phase=args.mirror_negative_phase == "flip",
                blend=args.mirror_negative_blend,
            )
        simulation._active_policy_name = args.model.parent.name
        if args.contact_friction is not None:
            model = simulation.sim.model
            contact_geoms = (
                model.geom("floor").id,
                model.geom(constants.LEFT_FEET_GEOMS[0]).id,
                model.geom(constants.RIGHT_FEET_GEOMS[0]).id,
            )
            model.geom_friction[list(contact_geoms), 0] = args.contact_friction
        rng = np.random.default_rng(seed)
        simulation.sim.data.qvel[:] += rng.uniform(
            -args.qvel_noise, args.qvel_noise, size=simulation.sim.model.nv
        )
        mujoco.mj_forward(simulation.sim.model, simulation.sim.data)

        for phase, duration, command in PHASES:
            if phase == "turn_negative":
                command = [0.0, 0.0, args.negative_yaw, 0.0, 0.0, 0.0, 0.0]
            start = simulation.sim.get_floating_base_qpos(
                simulation.sim.data.qpos
            ).copy()
            start_yaw = yaw_of(start)
            action = (
                "turn"
                if "turn" in phase
                else ("walk" if "walk" in phase else "stop")
            )
            result = simulation._run_command(
                action,
                command,
                duration_s=duration,
                source="sim2real_acceptance",
            )
            end = simulation.sim.get_floating_base_qpos(
                simulation.sim.data.qpos
            ).copy()
            end_yaw = yaw_of(end)
            dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
            forward = math.cos(start_yaw) * dx + math.sin(start_yaw) * dy
            heading = math.atan2(
                math.sin(end_yaw - start_yaw), math.cos(end_yaw - start_yaw)
            )
            executed = max(float(result["executed_duration_s"]), 1.0e-6)
            row = {
                "seed": seed,
                "phase": phase,
                "command": command,
                "executed_duration_s": round(executed, 6),
                "fallen": bool(result.get("fallen")),
                "forward_displacement_m": round(forward, 6),
                "mean_forward_speed_mps": round(forward / executed, 6),
                "heading_change_deg": round(math.degrees(heading), 3),
                "up_vector_z": result.get("up_vector_z"),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if row["fallen"] or executed < duration - 1.0e-3:
                break

    expected_phases = args.seeds * len(PHASES)
    summary = {
        "model": str(args.model),
        "seed_start": args.seed_start,
        "contact_friction": args.contact_friction,
        "mirror_negative_phase": args.mirror_negative_phase,
        "mirror_negative_blend": args.mirror_negative_blend,
        "negative_residual": str(args.negative_residual) if args.negative_residual else None,
        "negative_residual_band": args.negative_residual_band,
        "negative_yaw": args.negative_yaw,
        "completed_all_phases": len(rows) == expected_phases
        and not any(row["fallen"] for row in rows),
        "completed_phases": len(rows),
        "expected_phases": expected_phases,
        "falls": sum(row["fallen"] for row in rows),
        "rows": rows,
    }
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
