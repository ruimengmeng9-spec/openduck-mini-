"""Compare backward candidates in the native MuJoCo runtime.

V13 collapsed to a near-stationary policy, so this script reports the achieved
local forward speed per candidate and command.  A negative, command-proportional
speed means the reverse gait survived; a value near zero means it degraded.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer

REPO = Path("/data/shijinsheng/open_duck/projects/Open_Duck_Playground")
OFFICIAL = Path("/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx")
DEFAULT_OUTPUT = Path("/data/shijinsheng/open_duck/outputs/backward_v15_validation_20260918")

CANDIDATES = {
    "v12_best_reverse": Path(
        "/data/shijinsheng/open_duck/training/backward_v12_ratio_curriculum_20260915/final.onnx"
    ),
    "v13_degenerate": Path(
        "/data/shijinsheng/open_duck/training/backward_v13_stabilize_ratio_20260915/final.onnx"
    ),
    "v14_progress_balance": Path(
        "/data/shijinsheng/open_duck/training/backward_v14_progress_balance_20260918/final.onnx"
    ),
    "v15_fixed": Path(
        "/data/shijinsheng/open_duck/training/backward_v15_from_v12/final.onnx"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument("--speeds", type=float, nargs="+", default=[-0.04, -0.06])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, model in CANDIDATES.items():
        if not model.is_file():
            print(f"SKIP {name}: {model} missing", flush=True)
            continue
        for speed in args.speeds:
            simulation = DuckSimulation(REPO, OFFICIAL, output_root=args.output_dir, warmup_s=3.0)
            simulation.sim.policy = OnnxInfer(str(model), awd=True)
            simulation._active_policy_name = name
            start = simulation.sim.get_floating_base_qpos(simulation.sim.data.qpos).copy()
            qw, qx, qy, qz = (float(x) for x in start[3:7])
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            result = simulation._run_command(
                "backward", [speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                duration_s=args.duration_s, source="v14_validation",
            )
            end = simulation.sim.get_floating_base_qpos(simulation.sim.data.qpos)
            dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
            # Only the component along the robot's own heading counts as progress.
            forward = math.cos(yaw) * dx + math.sin(yaw) * dy
            executed = max(float(result["executed_duration_s"]), 1e-6)
            row = {
                "candidate": name,
                "command_mps": speed,
                "achieved_forward_speed_mps": round(forward / executed, 5),
                "forward_displacement_m": round(forward, 5),
                "fallen": result.get("fallen"),
                "executed_duration_s": result["executed_duration_s"],
                "up_vector_z": result.get("up_vector_z"),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    (args.output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SAVED", args.output_dir / "results.json", flush=True)


if __name__ == "__main__":
    main()
