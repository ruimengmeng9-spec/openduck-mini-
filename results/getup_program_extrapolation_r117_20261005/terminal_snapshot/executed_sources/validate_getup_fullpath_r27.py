"""Strict held-out completion gate; search's transition gate is not authority."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from diagnostics.getup_independent_native import POSES, digest
from diagnostics.train_getup_fullpath_r27 import AuditedSim, rollout, save_trace
import numpy as np


def strict_standing(m):
    # Retain the R20 final home-standing window in addition to loaded_entry.
    return bool(m['stable'] and .155 < m['height_m'] < .185
                and m['joint_home_error_mean_rad'] < .2)


class StrictSim(AuditedSim):
    def measure(self):
        m = super().measure()
        m['stable'] = strict_standing(m)
        return m


def evaluate(job):
    root, directory, pose, seed, nominal = job
    root, directory = Path(root), Path(directory)
    z = np.load(directory/'best_reference.npz', allow_pickle=False)
    sim = StrictSim(root/'training/getup_decomposed_r4/model/scene.xml',
                    root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    r, _ = rollout(sim, pose, z['targets'], z['durations_s'], seed, not nominal, 31.)
    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    jobs = []
    for index, pose in enumerate(POSES):
        directory = args.experiment/pose
        if not (directory/'results.json').exists():
            raise RuntimeError(f'Incomplete search: {pose}')
        jobs.extend((str(args.root), str(directory), pose, 700000+1000*index+i, False) for i in range(20))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, jobs, chunksize=1))
    summary = {}
    for index, pose in enumerate(POSES):
        rs = rows[index*20:(index+1)*20]
        summary[pose] = dict(successes=sum(r['success'] for r in rs), trials=20,
                             passed=sum(r['success'] for r in rs)>=18)
        r = next((r for r in rs if not r['success']), rs[0])
        z = np.load(args.experiment/pose/'best_reference.npz', allow_pickle=False)
        sim = StrictSim(args.root/'training/getup_decomposed_r4/model/scene.xml',
                        args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
        _, trace = rollout(sim, pose, z['targets'], z['durations_s'], r['seed'], True, 31., True)
        save_trace(args.output/f'{pose}_review.npz', trace)
    result = dict(summary=summary, results=rows, full_task_completed=all(r['passed'] for r in summary.values()),
                  simulation_only=True, hardware_readiness=False, default_controller_replaced=False,
                  final_gate='R6 loaded_entry AND height .155-.185m AND mean home error <.2rad; continuous tail >=30s',
                  maximum_perturbations=dict(tilt_rad=.05, joint_rad=.02, velocity=.01),
                  source_sha256=digest(__file__),
                  reference_hashes={pose: digest(args.experiment/pose/'best_reference.npz') for pose in POSES})
    (args.output/'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print('STRICT FULL TASK COMPLETED:', result['full_task_completed'], flush=True)


if __name__ == '__main__':
    main()
