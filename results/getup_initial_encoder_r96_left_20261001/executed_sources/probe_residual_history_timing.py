"""Read-only native simulation probe of the training action-history timing.

Does not alter the robot agent or its production observation path. The optional
shim delays only the three action-history blocks, not sensors/targets/phase.
"""

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer
from diagnostics.reference_residual_policy import ReferenceResidualPolicy


class LaggedHistoryActor:
    def __init__(self, actor):
        self.actor = actor
        self.previous_history = None

    def infer(self, obs):
        obs = np.asarray(obs, dtype=np.float32).copy()
        current = obs[41:83].copy()
        if self.previous_history is None:
            # The fourth pre-probe action is unavailable. This affects only the
            # first sample; later samples use exact observed history snapshots.
            obs[41:83] = np.concatenate([current[14:], np.zeros(14)])
        else:
            obs[41:83] = self.previous_history
        self.previous_history = current
        return self.actor.infer(obs)


def heading(qpos):
    w, x, y, z = qpos[3:7]
    return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/shijinsheng/open_duck"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--duration-s", type=float, default=10)
    parser.add_argument("--warmup-s", type=float, default=3)
    parser.add_argument("--models", nargs="+", choices=("r2", "r3"), default=["r2", "r3"])
    parser.add_argument("--modes", nargs="+", choices=("native_history", "training_lag_history"),
                        default=["native_history", "training_lag_history"])
    args = parser.parse_args()
    rows = []
    for name in args.models:
        model = args.root / f"training/backward_reference_residual_{name}/final.onnx"
        for mode in args.modes:
            for seed in range(args.seeds):
                robot = DuckSimulation(
                    args.root / "projects/Open_Duck_Playground",
                    args.root / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx",
                    output_root=args.output.parent / "history_probe_trajectories", warmup_s=args.warmup_s,
                )
                sim = robot.sim
                actor = OnnxInfer(str(model), awd=True)
                if mode == "training_lag_history":
                    actor = LaggedHistoryActor(actor)
                sim.policy = ReferenceResidualPolicy(actor, sim, 0.12, 1)
                sim.imitation_i = 0
                sim.imitation_phase = np.array([1, 0], dtype=np.float32)
                sim.data.qvel[:] += np.random.default_rng(seed).uniform(-.02, .02, sim.model.nv)
                mujoco.mj_forward(sim.model, sim.data)
                start = sim.get_floating_base_qpos(sim.data.qpos).copy()
                initial_heading = heading(start)
                previous_heading = initial_heading
                total_yaw = 0.
                min_up = 1.
                elapsed = 0.
                fallen = False
                while elapsed < args.duration_s - 1e-8:
                    result = robot._run_command(
                        "backward", [-.074, 0, 0, 0, 0, 0, 0],
                        min(.2, args.duration_s - elapsed), source="history_timing_probe",
                    )
                    elapsed += result["executed_duration_s"]
                    current_heading = heading(sim.get_floating_base_qpos(sim.data.qpos))
                    total_yaw += math.atan2(math.sin(current_heading-previous_heading), math.cos(current_heading-previous_heading))
                    previous_heading = current_heading
                    min_up = min(min_up, result["up_vector_z"])
                    fallen = result["fallen"]
                    if fallen:
                        break
                end = sim.get_floating_base_qpos(sim.data.qpos)
                dx, dy = end[:2] - start[:2]
                forward = math.cos(initial_heading)*dx + math.sin(initial_heading)*dy
                lateral = -math.sin(initial_heading)*dx + math.cos(initial_heading)*dy
                row = dict(model=name, mode=mode, seed=seed, warmup_s=args.warmup_s, duration_s=elapsed,
                           fallen=fallen, forward_speed_mps=float(forward/max(elapsed, .02)),
                           lateral_m=float(lateral), unwrapped_heading_change_deg=math.degrees(total_yaw),
                           sampled_min_up_z=min_up)
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"rows": rows, "history_slice": [41, 83],
                                     "first_sample_fourth_action": "zero (unavailable)"}, indent=2), encoding="utf-8")
    print("SAVED", args.output, flush=True)


if __name__ == "__main__":
    main()
