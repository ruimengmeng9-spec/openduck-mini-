"""Supervise a compact ONNX policy on the executable reverse reference gait.

The policy intentionally reads only command vx and the two phase features from
the 101-value runtime observation.  This makes stage 1 a precise behaviour-
cloning problem; PPO can later add state feedback and robustness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path("/data/shijinsheng/open_duck")
REPO = ROOT / "projects/Open_Duck_Playground"
REFERENCE = REPO / "playground/open_duck_mini_v2/data/polynomial_coefficients.pkl"
DEFAULT_OUTPUT = ROOT / "training/backward_reference_bc_v21/final.onnx"
DEFAULT_ACTUATOR = np.asarray(
    [
        0.002,
        0.053,
        -0.630,
        1.368,
        -0.784,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.003,
        -0.065,
        0.635,
        1.379,
        -0.796,
    ],
    dtype=np.float32,
)
REF_TO_ACT = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15]


class PhasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 14),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        phase_command = torch.stack(
            (obs[:, 6], obs[:, 99], obs[:, 100]), dim=1
        )
        return self.network(phase_command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--speed", type=float, default=-0.074)
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    args = parser.parse_args()

    sys.path.insert(0, str(REPO))
    from playground.common.poly_reference_motion_numpy import PolyReferenceMotion

    prm = PolyReferenceMotion(str(REFERENCE))
    observations = np.zeros((prm.nb_steps_in_period, 101), dtype=np.float32)
    targets = np.zeros((prm.nb_steps_in_period, 14), dtype=np.float32)
    for phase_i in range(prm.nb_steps_in_period):
        angle = phase_i / prm.nb_steps_in_period * 2.0 * np.pi
        observations[phase_i, 6] = args.speed
        observations[phase_i, 99:] = [np.cos(angle), np.sin(angle)]
        ref = np.asarray(
            prm.get_reference_motion(args.speed, 0.0, 0.0, phase_i),
            dtype=np.float32,
        )
        targets[phase_i] = np.clip(
            (ref[REF_TO_ACT] - DEFAULT_ACTUATOR) / 0.25, -1.0, 1.0
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(21)
    model = PhasePolicy().to(device)
    obs_tensor = torch.from_numpy(observations).to(device)
    target_tensor = torch.from_numpy(targets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    best_loss = float("inf")
    best_state = None
    for epoch in range(1, args.epochs + 1):
        prediction = model(obs_tensor)
        loss = torch.mean(torch.square(prediction - target_tensor))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_state = {
                key: tensor.detach().cpu().clone()
                for key, tensor in model.state_dict().items()
            }
        if epoch == 1 or epoch % 1000 == 0:
            print(f"epoch={epoch} mse={value:.8f}", flush=True)

    assert best_state is not None
    model.load_state_dict(best_state)
    model = model.cpu().eval()
    with torch.no_grad():
        final_prediction = model(torch.from_numpy(observations)).numpy()
    final_mse = float(np.mean(np.square(final_prediction - targets)))
    max_error = float(np.max(np.abs(final_prediction - targets)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros((1, 101), dtype=torch.float32),
        str(args.output),
        input_names=["obs"],
        output_names=["continuous_actions"],
        dynamic_axes={"obs": {0: "batch"}, "continuous_actions": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"device={device}")
    print(f"final_mse={final_mse:.10f} max_abs_error={max_error:.8f}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
