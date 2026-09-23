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
    parser.add_argument("--qvel-noise", type=float, default=0.02)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in range(args.seeds):
        simulation = DuckSimulation(
            REPO, OFFICIAL, output_root=args.output_dir, warmup_s=3.0
        )
        simulation.sim.policy = OnnxInfer(str(args.model), awd=True)
        simulation._active_policy_name = args.model.parent.name
        rng = np.random.default_rng(seed)
        simulation.sim.data.qvel[:] += rng.uniform(
            -args.qvel_noise, args.qvel_noise, size=simulation.sim.model.nv
        )
        mujoco.mj_forward(simulation.sim.model, simulation.sim.data)

        for phase, duration, command in PHASES:
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
