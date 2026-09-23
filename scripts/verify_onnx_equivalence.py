"""Compare two ONNX policies on deterministic random observations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    args = parser.parse_args()

    sessions = [
        ort.InferenceSession(str(path.resolve()), providers=["CPUExecutionProvider"])
        for path in (args.reference, args.candidate)
    ]
    rng = np.random.default_rng(args.seed)
    maximum = 0.0
    mean_total = 0.0
    for _ in range(args.samples):
        observation = rng.normal(size=(1, 101)).astype(np.float32)
        outputs = [
            session.run(None, {session.get_inputs()[0].name: observation})[0]
            for session in sessions
        ]
        difference = np.abs(outputs[0] - outputs[1])
        maximum = max(maximum, float(difference.max()))
        mean_total += float(difference.mean())

    mean = mean_total / args.samples
    print(f"samples={args.samples} max_abs_error={maximum:.9g} mean_abs_error={mean:.9g}")
    if maximum > args.atol:
        raise SystemExit(
            f"FAILED: max_abs_error {maximum:.9g} exceeds atol {args.atol:.9g}"
        )
    print("ONNX EQUIVALENCE: PASSED")


if __name__ == "__main__":
    main()
