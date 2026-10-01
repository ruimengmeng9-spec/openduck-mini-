"""Test geometry-derived joint references with genuine forward dynamics.

Prescribed root poses from the kinematic audit are NOT copied into simulation.
Only bounded joint targets are replayed from a normally settled fallen start.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import RecoverySim, feature_targets, digest
from diagnostics.getup_feedback_reference import run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.workspace/'results.json').read_text())['rows']
    by_angle = {row['pitch_deg']: row for row in rows}
    sim = RecoverySim(args.scene, args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    results, best = [], None
    for pose, sign in (('supine', -1), ('prone', 1)):
        feature_path = [by_angle[sign*a]['features_rad'] for a in (90,75,60,45,30,0)]
        targets = feature_targets(feature_path, sim.home, sim.lower, sim.upper)
        targets = np.concatenate([targets, sim.home[None]])
        for segment in (.2,.35,.5,.8,1.2):
            duration = segment*len(targets)
            for seed in (0, 141, 142):
                initial = sim.prepare(pose, seed, perturb=seed != 0)
                result, trace = run(sim, initial, targets, duration, record=True)
                result.update(pose=pose, seed=seed, nominal_segment_s=segment,
                              root_pose_edits_after_initialization=0, reference_duration_s=duration)
                results.append(result)
                print(json.dumps(result), flush=True)
                if best is None or result['score'] > best['score']:
                    best = result
                    np.savez_compressed(args.output/'best_reference.npz', targets=targets, duration_s=duration)
                    np.savez_compressed(args.output/'best_trajectory.npz', time=[r[0] for r in trace],
                                        qpos=[r[1] for r in trace], qvel=[r[2] for r in trace],
                                        ctrl=[r[3] for r in trace], phase=[r[4] for r in trace])
    summary = dict(method='geometry-guided reference duration sweep; genuine forward dynamics',
                   simulation_only=True, hardware_readiness=False, root_pose_teleports=False,
                   successful_runs=sum(r['success'] for r in results), total_runs=len(results),
                   results=results, best_result=best,
                   hashes={str(path): digest(path) for path in (args.scene, args.workspace/'results.json',Path(__file__))})
    (args.output/'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
