"""Measure single-joint recovery authority from actual fallen starts.

This is a diagnostic, not a recovery policy or acceptance test. Each candidate
is replayed from its own identical, physically settled fallen start. The
original scene, actuator force/position limits, 50 Hz target slew and substep
physical audit are kept unchanged. No learned or hardware controller is used.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, POSES, digest
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def candidate_targets(home, lower, upper, amplitude):
    home = np.asarray(home, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if home.shape != (14,) or lower.shape != (14,) or upper.shape != (14,):
        raise ValueError("Expected 14 joints")
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("Positive finite amplitude required")
    if np.any(lower > home) or np.any(home > upper):
        raise ValueError("Home must lie within joint limits")
    yield "home", None, 0, home.copy()
    for joint in range(14):
        for sign in (-1, 1):
            target = home.copy()
            target[joint] = np.clip(home[joint] + sign * amplitude,
                                    lower[joint], upper[joint])
            yield f"joint_{joint}_{'neg' if sign < 0 else 'pos'}", joint, sign, target


def measure_row(sim, initial, initial_hash, name, joint, sign, target, controls):
    sim.restore(initial)
    if state_hash(sim) != initial_hash:
        raise RuntimeError("Initial physical state changed between paired trials")
    sim.clear_audit()
    first = sim.measure()
    best_up = float(first["up_z"])
    best_up_t = 0.0
    best_load = float(first["foot_load_fraction"])
    best_load_t = 0.0
    strict_run = 0
    longest_strict_run = 0
    valid = True
    executed = 0
    for i in range(controls):
        sim.step_target(target)
        executed += 1
        current = sim.measure()
        if current["up_z"] > best_up:
            best_up, best_up_t = float(current["up_z"]), (i + 1) * DT
        if current["foot_load_fraction"] > best_load:
            best_load, best_load_t = float(current["foot_load_fraction"]), (i + 1) * DT
        valid = sim.physical_valid()
        strict_run = strict_run + 1 if valid and current["stable"] else 0
        longest_strict_run = max(longest_strict_run, strict_run)
        if not valid:
            break
    final = sim.measure()
    return dict(name=name, joint=joint, sign=sign,
                actual_target_offset_rad=(float(target[joint] - sim.home[joint])
                                          if joint is not None else 0.0),
                initial_hash=initial_hash, executed_controls=executed,
                physically_valid=bool(valid), best_up_z=best_up,
                best_up_time_s=best_up_t, best_foot_load_fraction=best_load,
                best_load_time_s=best_load_t,
                longest_strict_standing_s=longest_strict_run * DT,
                final_up_z=float(final["up_z"]),
                final_height_m=float(final["height_m"]),
                final_foot_load_fraction=float(final["foot_load_fraction"]),
                final_torso_contact=bool(final["torso_contact"]),
                peak_physics={k: float(v) for k, v in sim.peaks.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/shijinsheng/open_duck"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--amplitude-rad", type=float, default=0.6)
    parser.add_argument("--controls", type=int, default=50)
    parser.add_argument("--seed-base", type=int, default=1104000)
    args = parser.parse_args()
    if args.controls <= 0:
        raise ValueError("Positive control count required")
    if args.seed_base < 1000000 or args.seed_base >= 1300000:
        raise ValueError("Development seeds must avoid training and final acceptance")
    scene = args.root / "training/getup_decomposed_r4/model/scene.xml"
    stand = args.root / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(purpose="diagnostic joint authority, NOT recovery acceptance",
                    simulation_only=True, hardware_readiness=False,
                    physics_and_motor_limits_unchanged=True,
                    initial_root_mutation_only_in_normal_reset=True,
                    seed_base=args.seed_base, amplitude_rad=args.amplitude_rad,
                    controls=args.controls, physical_dt_s=0.002, motor_dt_s=DT,
                    source_sha256=digest(Path(__file__)), scene_sha256=digest(scene),
                    stand_actor_sha256=digest(stand))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sim = StrictSim(scene, stand)
    all_results = []
    for pose_index, pose in enumerate(POSES):
        seed = args.seed_base + 1000 * pose_index
        initial = sim.prepare(pose, seed, True)
        first = sim.measure()
        if not (first["up_z"] < .5 and first["torso_contact"]):
            raise RuntimeError(f"{pose} is not an actual fallen start")
        initial_hash = state_hash(sim)
        rows = [measure_row(sim, initial, initial_hash, name, joint, sign,
                            target, args.controls)
                for name, joint, sign, target in candidate_targets(
                    sim.home, sim.lower, sim.upper, args.amplitude_rad)]
        baseline = rows[0]
        for row in rows:
            row["best_up_gain_vs_home"] = row["best_up_z"] - baseline["best_up_z"]
            row["best_load_gain_vs_home"] = (row["best_foot_load_fraction"]
                                             - baseline["best_foot_load_fraction"])
        result = dict(pose=pose, seed=seed, initial_hash=initial_hash,
                      initial_up_z=float(first["up_z"]),
                      initial_height_m=float(first["height_m"]),
                      baseline=baseline,
                      valid_ranked_by_max_up_gain=sorted(
                          (r for r in rows[1:] if r["physically_valid"]),
                          key=lambda r: r["best_up_gain_vs_home"], reverse=True),
                      all_trials=rows)
        all_results.append(result)
        (args.output / "results.json").write_text(json.dumps(
            dict(manifest=manifest, results=all_results), indent=2))
        print(json.dumps(dict(pose=pose, baseline_up=baseline["best_up_z"],
                              top_valid=[(r["name"], round(r["best_up_gain_vs_home"], 4),
                                          round(r["best_load_gain_vs_home"], 4))
                                         for r in result["valid_ranked_by_max_up_gain"][:5]])),
              flush=True)
    print("JOINT_AUTHORITY_DIAGNOSTIC_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
