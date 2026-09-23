"""Export a standalone Orbax PPO checkpoint to an ONNX policy.

The training callback receives Brax's typed normalization state, while a fresh
Orbax restore returns the same state as a plain dictionary.  Convert that first
element back to ``RunningStatisticsState`` before using the project's existing
ONNX exporter.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from brax.training.acme import running_statistics
from mujoco_playground.config import locomotion_params
from orbax import checkpoint as ocp

from playground.common.export_onnx import export_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--action-size", type=int, default=14)
    parser.add_argument("--observation-size", type=int, default=101)
    args = parser.parse_args()

    params = ocp.PyTreeCheckpointer().restore(str(args.checkpoint.resolve()))
    state = params[0]
    if isinstance(state, dict):
        params[0] = running_statistics.RunningStatisticsState(
            mean=state["mean"],
            std=state["std"],
            count=state["count"],
            summed_variance=state["summed_variance"],
            std_eps=float(state["std_eps"]),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ppo_params = locomotion_params.brax_ppo_config(
        "BerkeleyHumanoidJoystickFlatTerrain"
    )
    export_onnx(
        params,
        args.action_size,
        ppo_params,
        args.observation_size,
        output_path=str(args.output.resolve()),
    )
    print(f"EXPORTED: {args.output.resolve()}")


if __name__ == "__main__":
    main()
