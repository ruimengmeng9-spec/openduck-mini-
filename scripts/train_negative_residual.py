"""Fit an experimental negative-turn residual to simulated mirror-teacher data.

The base v5 ONNX actor remains unchanged. This small network predicts an
action correction only for yaw commands near -0.15 rad/s. A closed-loop
simulation validation is required; offline fit alone is not acceptance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(115, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 14),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net((features - self.mean) / self.std)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--additional-dataset", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-residual-gain", type=float, default=1.0)
    args = parser.parse_args()
    if args.target_residual_gain <= 0:
        parser.error("--target-residual-gain must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)

    arrays = {key: [] for key in ("obs", "base", "teacher", "seed")}
    for path in [args.dataset, *args.additional_dataset]:
        with np.load(path, allow_pickle=False) as data:
            for key in arrays:
                arrays[key].append(data[key])
    obs = np.concatenate(arrays["obs"]).astype(np.float32)
    base = np.concatenate(arrays["base"]).astype(np.float32)
    teacher = np.concatenate(arrays["teacher"]).astype(np.float32)
    seeds = np.concatenate(arrays["seed"]).astype(np.int32)
    features = np.concatenate([obs, base], axis=1)
    targets = (teacher - base) * args.target_residual_gain
    unique_seeds = np.unique(seeds)
    validation_seeds = unique_seeds[::5]
    validation = np.isin(seeds, validation_seeds)
    training = ~validation
    if not training.any() or not validation.any():
        raise ValueError("Dataset requires at least one train and validation seed")
    mean = features[training].mean(axis=0)
    std = np.maximum(features[training].std(axis=0), 1.0e-3)
    model = ResidualMLP(mean, std)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    x_train = torch.from_numpy(features[training])
    y_train = torch.from_numpy(targets[training])
    x_valid = torch.from_numpy(features[validation])
    y_valid = torch.from_numpy(targets[validation])
    best = float("inf")
    best_epoch = -1
    best_state = None
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        permutation = torch.randperm(len(x_train))
        losses = []
        for indices in permutation.split(256):
            prediction = model(x_train[indices])
            loss = torch.mean((prediction - y_train[indices]) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            valid_loss = float(torch.mean((model(x_valid) - y_valid) ** 2))
        if valid_loss < best - 1.0e-7:
            best = valid_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if epoch % 10 == 0 or patience == 0:
            print(
                f"epoch={epoch} train_mse={np.mean(losses):.7f} "
                f"valid_mse={valid_loss:.7f} best={best:.7f}", flush=True
            )
        if patience >= 20:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    torch.save(model.state_dict(), args.output_dir / "residual.pt")
    torch.onnx.export(
        model,
        torch.zeros(1, 115),
        str(args.output_dir / "residual.onnx"),
        input_names=["features"],
        output_names=["delta"],
        opset_version=17,
        dynamo=False,
    )
    summary = {
        "dataset": str(args.dataset),
        "additional_datasets": [str(path) for path in args.additional_dataset],
        "train_samples": int(training.sum()),
        "validation_samples": int(validation.sum()),
        "validation_seeds": validation_seeds.tolist(),
        "best_epoch": best_epoch,
        "validation_mse": best,
        "teacher_residual_rmse": float(np.sqrt(np.mean(targets ** 2))),
        "target_residual_gain": args.target_residual_gain,
        "model_type": "residual_mlp_115x128x128x14",
        "observed_negative_command_range_rad_s": [
            float(obs[:, 8].min()), float(obs[:, 8].max())
        ],
        "simulation_only": True,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
