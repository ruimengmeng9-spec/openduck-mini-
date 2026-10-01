"""Two-stage contact-transition search from actual fallen starts.

This is a physical sequence search, not PPO training and not an acceptance
test. Candidate joints come from independently replayed R34 authority probes.
Each sequence is replayed from the same complete fallen initial state, without
teleporting the base or altering scene, friction, forces, joint or slew limits.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.train_getup_fullpath_r27 import progress, state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def target_library(pair_pose, single_pose, home, lower, upper, pair_count=6):
    """Build named targets without mixing multiple teacher trajectories."""
    home = np.asarray(home, dtype=float)
    pairs = []
    for row in pair_pose["valid_ranked_by_max_up_gain"][:pair_count]:
        offsets = np.asarray(row["target_offsets_rad"], dtype=float)
        if offsets.shape != home.shape or not np.isfinite(offsets).all():
            raise ValueError("Invalid paired target offsets")
        target = home + offsets
        if np.any(target < lower - 1e-8) or np.any(target > upper + 1e-8):
            raise ValueError("Paired target exceeds physical joint limits")
        pairs.append((row["name"], target))
    singles = []
    for row in single_pose["valid_ranked_by_max_up_gain"][:2]:
        j = row["joint"]
        target = home.copy()
        target[j] = np.clip(home[j] + row["actual_target_offset_rad"],
                            lower[j], upper[j])
        singles.append((row["name"], target))
    if len(pairs) < 4:
        raise ValueError("Need at least four physically valid paired targets")
    return pairs[:4], pairs + singles + [("home", home.copy())]


def simulate(sim, start, start_hash, first, second, controls, record=False):
    sim.restore(start)
    if state_hash(sim) != start_hash:
        raise RuntimeError("Search candidate initial state changed")
    sim.clear_audit()
    first_measure = sim.measure()
    best_progress = progress(first_measure)
    best_up = float(first_measure["up_z"])
    best_load = float(first_measure["foot_load_fraction"])
    best_progress_time_s = 0.0
    strict_run = longest_strict_run = executed = 0
    valid = True
    trajectory = dict(time=[], qpos=[], qvel=[], ctrl=[], phase=[])
    stages = (("first", first, controls[0]),
              ("second", second, controls[1]),
              ("home", sim.home, controls[2]))
    for phase, target, count in stages:
        for _ in range(count):
            sim.step_target(target)
            executed += 1
            m = sim.measure()
            valid = sim.physical_valid()
            p = progress(m)
            if p > best_progress:
                best_progress = p
                best_progress_time_s = executed * DT
            best_up = max(best_up, float(m["up_z"]))
            best_load = max(best_load, float(m["foot_load_fraction"]))
            strict_run = strict_run + 1 if valid and m["stable"] else 0
            longest_strict_run = max(longest_strict_run, strict_run)
            if record:
                trajectory["time"].append(float(sim.data.time))
                trajectory["qpos"].append(sim.data.qpos.copy())
                trajectory["qvel"].append(sim.data.qvel.copy())
                trajectory["ctrl"].append(sim.data.ctrl.copy())
                trajectory["phase"].append(phase)
            if not valid:
                break
        if not valid:
            break
    final = sim.measure()
    row = dict(initial_hash=start_hash, physically_valid=bool(valid),
               executed_controls=executed, best_progress=float(best_progress),
               best_progress_time_s=best_progress_time_s,
               best_up_z=best_up, best_foot_load_fraction=best_load,
               longest_strict_standing_s=longest_strict_run * DT,
               final_up_z=float(final["up_z"]),
               final_height_m=float(final["height_m"]),
               final_foot_load_fraction=float(final["foot_load_fraction"]),
               final_torso_contact=bool(final["torso_contact"]),
               final_stable=bool(final["stable"]),
               peak_physics={k: float(v) for k, v in sim.peaks.items()})
    return row, trajectory


def rank_key(row):
    return (row["physically_valid"], row["longest_strict_standing_s"],
            row["best_progress"], row["best_up_z"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("/data/shijinsheng/open_duck"))
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--singles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-controls", type=int, default=75)
    parser.add_argument("--second-controls", type=int, default=75)
    parser.add_argument("--home-controls", type=int, default=100)
    args = parser.parse_args()
    controls = (args.first_controls, args.second_controls, args.home_controls)
    if any(n <= 0 for n in controls):
        raise ValueError("Each physical control stage must have positive length")
    pair_data = json.loads(args.pairs.read_text())
    single_data = json.loads(args.singles.read_text())
    if [x["pose"] for x in pair_data["results"]] != [x["pose"] for x in single_data["results"]]:
        raise ValueError("Pose rows do not match")
    scene = args.root / "training/getup_decomposed_r4/model/scene.xml"
    stand = args.root / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx"
    for source in (pair_data["manifest"], single_data["manifest"]):
        if source["scene_sha256"] != digest(scene) or source["stand_actor_sha256"] != digest(stand):
            raise RuntimeError("Physical scene or standing actor changed")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(purpose="two-stage physical search, NOT final recovery acceptance",
                    simulation_only=True, hardware_readiness=False,
                    physics_and_motor_limits_unchanged=True,
                    phase_controls=dict(first=controls[0], second=controls[1], home=controls[2]),
                    model_sha256=digest(scene), standing_actor_sha256=digest(stand),
                    pair_source_sha256=digest(args.pairs),
                    single_source_sha256=digest(args.singles),
                    executable_sha256=digest(Path(__file__)),
                    selection="per-pose top physical-valid progress on the source development seed; separate follow-up seeds are diagnostics only",
                    final_acceptance="20 independent seeds in each of four poses, >=18 successes each, 30s continuous strict loaded standing; NOT run by this search")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    sim = StrictSim(scene, stand)
    all_results = []
    for pair_pose, single_pose in zip(pair_data["results"], single_data["results"]):
        pose, seed = pair_pose["pose"], pair_pose["seed"]
        if single_pose["pose"] != pose or single_pose["seed"] != seed:
            raise ValueError("Development seed mismatch")
        start = sim.prepare(pose, seed, True)
        start_hash = state_hash(sim)
        if start_hash != pair_pose["initial_hash"]:
            raise RuntimeError("Prepared fallen start differs from source probes")
        m = sim.measure()
        if not (m["up_z"] < .5 and m["torso_contact"]):
            raise RuntimeError("Search must begin actually fallen")
        firsts, seconds = target_library(pair_pose, single_pose,
                                          sim.home, sim.lower, sim.upper)
        rows = []
        for first_name, first in firsts:
            for second_name, second in seconds:
                row, _ = simulate(sim, start, start_hash, first, second, controls)
                row.update(first=first_name, second=second_name)
                rows.append(row)
        best = max(rows, key=rank_key)
        first = next(x for name, x in firsts if name == best["first"])
        second = next(x for name, x in seconds if name == best["second"])
        replay, trace = simulate(sim, start, start_hash, first, second,
                                 controls, record=True)
        if any(abs(replay[key] - best[key]) > 1e-8 for key in (
                "best_progress", "best_up_z", "final_up_z")):
            raise RuntimeError("Best-sequence replay differs from search")
        np.savez_compressed(args.output / f"{pose}_best_development.npz",
                            **{k: np.asarray(v) for k, v in trace.items()})
        followup = []
        for next_seed in (seed + 4000, seed + 5000):
            other = sim.prepare(pose, next_seed, True)
            other_m = sim.measure()
            if not (other_m["up_z"] < .5 and other_m["torso_contact"]):
                raise RuntimeError("Follow-up is not actually fallen")
            result, _ = simulate(sim, other, state_hash(sim), first, second,
                                 controls)
            result["seed"] = next_seed
            followup.append(result)
        all_results.append(dict(pose=pose, development_seed=seed,
                                development_initial_hash=start_hash,
                                proposal_count=len(rows), all_proposals=rows,
                                best_development=best, followup_development=followup))
        (args.output / "results.json").write_text(json.dumps(
            dict(manifest=manifest, results=all_results), indent=2))
        print(json.dumps(dict(pose=pose, proposed=len(rows),
                              best=[best["first"], best["second"]],
                              best_up=round(best["best_up_z"], 4),
                              strict_s=round(best["longest_strict_standing_s"], 3),
                              followup_up=[round(x["best_up_z"], 4) for x in followup])),
              flush=True)
    print("TWO_STAGE_SEARCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
