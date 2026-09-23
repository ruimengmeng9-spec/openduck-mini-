"""Validate the 100M-step backward policies in native MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
OFFICIAL = ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
DEFAULT_OUTPUT = ROOT / "outputs/backward_long_validation_20260920"

CANDIDATES = {
    "long_imit8_100m": ROOT / "training/backward_long_imit8/final.onnx",
    "long_imit20_100m": ROOT / "training/backward_long_imit20/final.onnx",
    "teacher_v20_20m": ROOT / "training/backward_teacher_v20/final.onnx",
    "reference_bc_v21": ROOT / "training/backward_reference_bc_v21/final.onnx",
    "residual_v22_30m": ROOT / "training/backward_residual_v22/final.onnx",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--speeds", type=float, nargs="+", default=[-0.074])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, model in CANDIDATES.items():
        if not model.is_file():
            print(f"SKIP {name}: {model} missing", flush=True)
            continue
        for speed in args.speeds:
            simulation = DuckSimulation(
                REPO, OFFICIAL, output_root=args.output_dir, warmup_s=3.0
            )
            simulation.sim.policy = OnnxInfer(str(model), awd=True)
            simulation._active_policy_name = name
            start = simulation.sim.get_floating_base_qpos(
                simulation.sim.data.qpos
            ).copy()
            qw, qx, qy, qz = (float(x) for x in start[3:7])
            yaw = math.atan2(
                2 * (qw * qz + qx * qy),
                1 - 2 * (qy * qy + qz * qz),
            )
            result = simulation._run_command(
                "backward",
                [speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                duration_s=args.duration_s,
                source="long_checkpoint_validation",
            )
            end = simulation.sim.get_floating_base_qpos(simulation.sim.data.qpos)
            dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
            forward = math.cos(yaw) * dx + math.sin(yaw) * dy
            executed = max(float(result["executed_duration_s"]), 1e-6)
            row = {
                "candidate": name,
                "model": str(model),
                "command_mps": speed,
                "achieved_forward_speed_mps": round(forward / executed, 5),
                "forward_displacement_m": round(forward, 5),
                "fallen": result.get("fallen"),
                "executed_duration_s": result["executed_duration_s"],
                "up_vector_z": result.get("up_vector_z"),
                "base_position_m": result.get("base_position_m"),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    result_path = args.output_dir / "results.json"
    result_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SAVED", result_path, flush=True)


if __name__ == "__main__":
    main()
