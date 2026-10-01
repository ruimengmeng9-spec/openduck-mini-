"""Paired-joint dynamic probes after R34 single-joint authority mapping.

Every pair restarts from the same actual fallen pose, retains the original
physics/motor implementation and is only a search diagnostic. It cannot pass
the 30-second loaded-standing acceptance gate by construction.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import digest
from diagnostics.probe_getup_joint_authority_r34 import measure_row
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def pair_targets(home, lower, upper, ranked, amplitude, top_n=6):
    """Yield distinct two-joint targets from the best valid single probes."""
    chosen = ranked[:top_n]
    for first, second in combinations(chosen, 2):
        if first["joint"] == second["joint"]:
            continue
        target = np.asarray(home, dtype=float).copy()
        for row in (first, second):
            j = row["joint"]
            target[j] = np.clip(home[j] + row["sign"] * amplitude,
                                lower[j], upper[j])
        label = f"{first['name']}__{second['name']}"
        yield label, (first["joint"], second["joint"]), target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/shijinsheng/open_duck"))
    parser.add_argument("--singles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--controls", type=int, default=75)
    parser.add_argument("--top-n", type=int, default=6)
    args = parser.parse_args()
    if args.controls <= 0 or not 2 <= args.top_n <= 14:
        raise ValueError("Invalid controls or top-n")
    source = json.loads(args.singles.read_text())
    original = source["manifest"]
    amplitude = original["amplitude_rad"]
    scene = args.root / "training/getup_decomposed_r4/model/scene.xml"
    stand = args.root / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
    if digest(scene) != original["scene_sha256"] or digest(stand) != original["stand_actor_sha256"]:
        raise RuntimeError("Model or standing actor changed since single-joint probe")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(purpose="pair interaction probe, NOT recovery acceptance",
                    simulation_only=True, hardware_readiness=False,
                    physics_and_motor_limits_unchanged=True,
                    controls=args.controls, top_n=args.top_n,
                    amplitude_rad=amplitude, singles_sha256=digest(args.singles),
                    source_sha256=digest(Path(__file__)),
                    shared_helper_sha256=digest(Path(measure_row.__code__.co_filename)),
                    scene_sha256=digest(scene), stand_actor_sha256=digest(stand))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sim = StrictSim(scene, stand)
    all_results = []
    for row in source["results"]:
        pose, seed = row["pose"], row["seed"]
        initial = sim.prepare(pose, seed, True)
        initial_hash = state_hash(sim)
        if initial_hash != row["initial_hash"]:
            raise RuntimeError(f"{pose}: initial state does not match single-joint probe")
        baseline = measure_row(sim, initial, initial_hash, "home", None, 0,
                               sim.home, args.controls)
        trials = []
        for name, joints, target in pair_targets(
                sim.home, sim.lower, sim.upper,
                row["valid_ranked_by_max_up_gain"], amplitude, args.top_n):
            result = measure_row(sim, initial, initial_hash, name, None, 0,
                                 target, args.controls)
            result["joint_indices"] = list(joints)
            result["target_offsets_rad"] = (target - sim.home).tolist()
            # The single-joint field is inapplicable to these paired targets.
            result.pop("actual_target_offset_rad")
            result["best_up_gain_vs_home"] = result["best_up_z"] - baseline["best_up_z"]
            result["best_load_gain_vs_home"] = (result["best_foot_load_fraction"]
                                                 - baseline["best_foot_load_fraction"])
            trials.append(result)
        valid = sorted((r for r in trials if r["physically_valid"]),
                       key=lambda r: r["best_up_gain_vs_home"], reverse=True)
        all_results.append(dict(pose=pose, seed=seed, initial_hash=initial_hash,
                                baseline=baseline, all_trials=trials,
                                valid_ranked_by_max_up_gain=valid))
        (args.output / "results.json").write_text(json.dumps(
            dict(manifest=manifest, results=all_results), indent=2))
        print(json.dumps(dict(pose=pose, valid_pairs=len(valid),
                              best=[(x["name"], round(x["best_up_gain_vs_home"], 4),
                                     round(x["best_foot_load_fraction"], 4))
                                    for x in valid[:5]])), flush=True)
    print("JOINT_PAIR_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
