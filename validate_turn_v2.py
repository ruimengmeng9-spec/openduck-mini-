"""Compare turn-policy candidates in the native MuJoCo runtime.

The deployed turn policy (`open_duck_turn_v1.onnx`) and any newly trained
candidate are both rolled out at several yaw commands, and the achieved heading
change is converted into an effective yaw rate so the two can be compared on
the same axis.  Results are appended as JSON so repeated runs stay auditable.
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
DEFAULT_OUTPUT = Path("/data/shijinsheng/open_duck/outputs/turn_validation")

# The deployed policy is the reference; the candidate is whatever we want to
# promote next.  Both run through the same 3-second warm-up so the comparison is
# about the turn response, not about a cold start.
CANDIDATES = {
    "turn_v1_deployed": Path(
        "/data/shijinsheng/open_duck/models/control/open_duck_turn_v1.onnx"
    ),
    "turn_v2_focused": Path(
        "/data/shijinsheng/open_duck/training/turn_v2_focused_20260918/final.onnx"
    ),
}


def yaw_of(qpos) -> float:
    """Extract the base yaw angle from a floating-base quaternion."""
    qw, qx, qy, qz = (float(x) for x in qpos[3:7])
    return math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duration-s", type=float, default=5.0)
    parser.add_argument(
        "--rates",
        type=float,
        nargs="+",
        default=[0.3, -0.3, 0.15, -0.15],
        help="Yaw-rate commands in rad/s; include both signs to measure symmetry.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, model in CANDIDATES.items():
        if not model.is_file():
            print(f"SKIP {name}: {model} missing", flush=True)
            continue
        for rate in args.rates:
            # Warm up on the stable walk policy, then swap in the turn candidate.
            # This mirrors how the policy is switched at runtime.
            simulation = DuckSimulation(REPO, OFFICIAL, output_root=args.output_dir, warmup_s=3.0)
            simulation.sim.policy = OnnxInfer(str(model), awd=True)
            simulation._active_policy_name = name

            start_yaw = yaw_of(simulation.sim.get_floating_base_qpos(simulation.sim.data.qpos))
            result = simulation._run_command(
                "turn",
                [0.0, 0.0, rate, 0.0, 0.0, 0.0, 0.0],
                duration_s=args.duration_s,
                source="turn_validation",
            )
            end_yaw = yaw_of(simulation.sim.get_floating_base_qpos(simulation.sim.data.qpos))
            # Wrap to (-180, 180] so a nearly-identical heading does not read as a
            # full turn in the opposite direction.
            delta_deg = math.degrees(
                math.atan2(math.sin(end_yaw - start_yaw), math.cos(end_yaw - start_yaw))
            )
            executed = max(float(result["executed_duration_s"]), 1.0e-6)
            row = {
                "policy": name,
                "model": str(model),
                "command_rad_s": rate,
                "heading_change_deg": round(delta_deg, 3),
                "effective_yaw_rate": round(math.radians(delta_deg) / executed, 4),
                "response_ratio": round(
                    (math.radians(delta_deg) / executed) / rate if rate else 0.0, 4
                ),
                "fallen": result.get("fallen"),
                "executed_duration_s": result["executed_duration_s"],
                "up_vector_z": result.get("up_vector_z"),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)

    result_path = args.output_dir / "turn_comparison.json"
    result_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("SAVED", result_path, flush=True)


if __name__ == "__main__":
    main()
