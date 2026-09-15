"""Validate backward-policy checkpoints through the native MuJoCo runtime."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer


ROOT = Path(os.environ.get("OPEN_DUCK_ROOT", Path.home() / "open_duck"))
REPO = Path(os.environ.get("OPEN_DUCK_PLAYGROUND", ROOT / "projects/Open_Duck_Playground"))
OFFICIAL = Path(
    os.environ.get(
        "OPEN_DUCK_WALK_MODEL",
        ROOT / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx",
    )
)
TRAINING = Path(
    os.environ.get(
        "BACKWARD_TRAINING_DIR",
        ROOT / "training/backward_v11_signed_reward_20260915",
    )
)
OUTPUT = Path(
    os.environ.get(
        "BACKWARD_VALIDATION_DIR",
        ROOT / "outputs/backward_v11_validation_20260915",
    )
)

CANDIDATES = {
    "v11_2p13m": TRAINING / "step_2129920.onnx",
    "v11_4p26m": TRAINING / "step_4259840.onnx",
    "v11_6p39m": TRAINING / "step_6389760.onnx",
    "v11_10p65m": TRAINING / "step_10649600.onnx",
    "v11_12p78m": TRAINING / "final.onnx",
}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "results.jsonl"
    with result_path.open("w", encoding="utf-8") as output:
        for name, model_path in CANDIDATES.items():
            for speed in (-0.03, -0.04, -0.06):
                # Warm up with the deployed stable policy, then switch only the
                # policy network.  This matches the eventual action transition.
                simulation = DuckSimulation(REPO, OFFICIAL, output_root=OUTPUT, warmup_s=3.0)
                simulation.sim.policy = OnnxInfer(str(model_path), awd=True)
                simulation._active_policy_name = name
                start = simulation.sim.get_floating_base_qpos(simulation.sim.data.qpos).copy()
                qw, qx, qy, qz = (float(x) for x in start[3:7])
                yaw = math.atan2(
                    2.0 * (qw * qz + qx * qy),
                    1.0 - 2.0 * (qy * qy + qz * qz),
                )
                result = simulation._run_command(
                    "backward",
                    [speed, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    duration_s=5.0,
                    source="checkpoint_validation",
                )
                end = simulation.sim.get_floating_base_qpos(simulation.sim.data.qpos)
                dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
                local_forward_displacement = math.cos(yaw) * dx + math.sin(yaw) * dy
                signed_speed = local_forward_displacement / max(
                    float(result["executed_duration_s"]), 1.0e-6
                )
                row = {
                    "candidate": name,
                    "model": str(model_path),
                    "command_mps": speed,
                    "mean_local_forward_speed_mps": signed_speed,
                    "local_forward_displacement_m": local_forward_displacement,
                    **result,
                }
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
                output.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
    print("RESULTS", result_path, flush=True)


if __name__ == "__main__":
    main()
