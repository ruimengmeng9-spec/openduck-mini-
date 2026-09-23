"""Seed a restartable Brax PPO checkpoint from an exported ONNX actor.

The official Open Duck walking policy and this repository's PPO actor use the
same 101-512-256-128-28 SiLU MLP.  This utility copies the actor weights and
observation normalization statistics from ONNX into an existing PPO checkpoint
while retaining its critic parameters.  The resulting checkpoint can be used
with ``--restore_checkpoint_path`` for PPO fine-tuning.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import onnx
from flax.training import orbax_utils
from onnx import numpy_helper
from orbax import checkpoint as ocp


def _initializer_map(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    return {
        item.name: numpy_helper.to_array(item).astype(np.float32)
        for item in model.graph.initializer
    }


def _actor_arrays(model: onnx.ModelProto) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    arrays = _initializer_map(model)
    sub = next(node for node in model.graph.node if node.op_type == "Sub")
    mul = next(
        node
        for node in model.graph.node
        if node.op_type == "Mul" and node.input[0] == sub.output[0]
    )
    mean = arrays[sub.input[1]]
    std = 1.0 / arrays[mul.input[1]]

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    for node in (node for node in model.graph.node if node.op_type == "Gemm"):
        layers.append((arrays[node.input[1]], arrays[node.input[2]]))
    if [kernel.shape for kernel, _ in layers] != [
        (101, 512),
        (512, 256),
        (256, 128),
        (128, 28),
    ]:
        raise ValueError("ONNX actor architecture is not 101-512-256-128-28")
    return mean, std, layers


def _count_as_float(count: dict[str, object] | object) -> float:
    if isinstance(count, dict):
        return float(np.asarray(count["hi"])) * 2.0**32 + float(
            np.asarray(count["lo"])
        )
    return float(np.asarray(count))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("teacher_onnx", type=Path)
    parser.add_argument("base_checkpoint", type=Path)
    parser.add_argument("output_checkpoint", type=Path)
    args = parser.parse_args()

    teacher = onnx.load(str(args.teacher_onnx.resolve()))
    mean, std, layers = _actor_arrays(teacher)
    params = ocp.PyTreeCheckpointer().restore(str(args.base_checkpoint.resolve()))

    actor = params[1]["params"]
    for index, (kernel, bias) in enumerate(layers):
        layer = actor[f"hidden_{index}"]
        if tuple(layer["kernel"].shape) != kernel.shape:
            raise ValueError(f"kernel shape mismatch for hidden_{index}")
        if tuple(layer["bias"].shape) != bias.shape:
            raise ValueError(f"bias shape mismatch for hidden_{index}")
        layer["kernel"] = jnp.asarray(kernel)
        layer["bias"] = jnp.asarray(bias)

    stats = params[0]
    stats["mean"]["state"] = jnp.asarray(mean)
    stats["std"]["state"] = jnp.asarray(std)
    count = _count_as_float(stats["count"])
    std_eps = float(np.asarray(stats["std_eps"]))
    variance = np.maximum(std.astype(np.float64) ** 2 - std_eps, 0.0)
    stats["summed_variance"]["state"] = jnp.asarray(
        variance * count, dtype=jnp.float32
    )

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpointer.save(
        str(args.output_checkpoint.resolve()),
        params,
        force=True,
        save_args=save_args,
    )
    print(f"IMPORTED: {args.output_checkpoint.resolve()}")
    print(f"NORMALIZER COUNT: {count:.0f}")


if __name__ == "__main__":
    main()
