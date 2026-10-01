"""Structured, substep-audited support-transfer hypotheses; simulation only.

R27 locked head yaw/roll. First test unlocking them on the identical failed path.
Then test normal-knee tuck/head-lever/return-home templates from actual fallen
starts. A template is never labeled a teacher unless strict dynamics succeed.
"""
import argparse
from itertools import product
import json
from pathlib import Path
import shutil
import time

import numpy as np

from diagnostics.getup_independent_native import digest
from diagnostics.train_getup_fullpath_r27 import rollout, save_trace
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def templates(sim):
    for hip, ankle, knee, neck, head, reverse in product(
            (.8, 1.15), (-.6, 0.), (.2, .8, 1.5), (.5, 1.05), (-.6, .6), (False, True)):
        q = np.tile(sim.home, (4, 1))
        q[:2, [1, 10]] = 0.
        q[:2, 2], q[:2, 11] = -hip, hip
        q[0, [3, 12]] = 1.5
        q[1, [3, 12]] = knee
        q[:2, [4, 13]] = ankle
        q[:2, 5] = neck
        q[:2, 6] = head
        if reverse:
            q[1, 5] = -.25
            q[1, 6] = -head
        yield dict(hip=hip, ankle=ankle, knee=knee, neck=neck, head=head, reverse=reverse), \
            np.clip(q, sim.lower, sim.upper), np.array([.6, .8, .8, .6])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    for source in (Path(__file__), Path(__file__).with_name('train_getup_fullpath_r27.py'),
                   Path(__file__).with_name('validate_getup_fullpath_r27.py')):
        shutil.copy2(source, args.output/source.name)
    rows, started, best, controls = [], time.monotonic(), {}, 0
    for pose in ('left_side', 'right_side'):
        prior = args.root/'training/getup_fullpath_r27'/pose/'best_reference.npz'
        # Snapshot the exact reference for paired and reproducible comparisons.
        z = np.load(prior, allow_pickle=False)
        original, durations = z['targets'].copy(), z['durations_s'].copy()
        np.savez_compressed(args.output/f'{pose}_source_reference.npz', targets=original, durations_s=durations)
        for roll, yaw in ((0., 0.), (-.45, 0.), (.45, 0.), (0., -1.4), (0., 1.4),
                          (-.45, -1.4), (.45, 1.4)):
            q = original.copy()
            early = np.cumsum(durations) <= 2.5
            q[early, 8], q[early, 7] = roll, yaw
            r, _ = rollout(sim, pose, q, durations, hold_s=4.)
            controls += r['control_steps']
            rows.append(dict(kind='head_unlock', pose=pose, parameters=dict(roll=roll, yaw=yaw), result=r))
            if pose not in best or r['score'] > best[pose][0]['score']:
                best[pose] = (r, q.copy(), durations.copy())
        # All variants must start at precisely the same physical state.
        same = [row['result']['initial_hash'] for row in rows if row['pose']==pose]
        if len(set(same)) != 1:
            raise RuntimeError('paired physical start mismatch')
        print('HEAD UNLOCK',pose,[(row['parameters'], row['result']['valid'], row['result']['success'],
                                   row['result']['score']) for row in rows if row['pose']==pose],flush=True)
    for pose in ('prone', 'supine'):
        for params, q, d in templates(sim):
            r, _ = rollout(sim, pose, q, d, hold_s=4.)
            controls += r['control_steps']
            rows.append(dict(kind='normal_knee_head_lever', pose=pose, parameters=params, result=r))
            if pose not in best or r['score'] > best[pose][0]['score']:
                best[pose] = (r, q.copy(), d.copy())
            if len(rows)%16==0:
                (args.output/'progress.json').write_text(json.dumps(dict(completed=len(rows),controls=controls,
                    best={k:v[0] for k,v in best.items()},wall_seconds=time.monotonic()-started),indent=2))
                print('TEMPLATE',pose,'completed',len(rows),'best_score',best[pose][0]['score'],
                      'strict_success',best[pose][0]['success'],flush=True)
    validation = {}
    for pose, (r, q, d) in best.items():
        np.savez_compressed(args.output/f'{pose}_best_reference.npz', targets=q, durations_s=d)
        nominal, trace = rollout(sim, pose, q, d, hold_s=31., record=True)
        save_trace(args.output/f'{pose}_nominal_trajectory.npz', trace)
        trials = []
        count, hold = (20, 31.) if nominal['success'] else (3, 4.)
        for seed in range(800000, 800000+count):
            row, _ = rollout(sim, pose, q, d, seed, True, hold)
            trials.append(row)
        validation[pose] = dict(nominal=nominal, successes=sum(r['success'] for r in trials),
                                trial_count=count, heldout_hold_s=hold, trials=trials)
    report = dict(rows=rows, validation=validation, search_controls=controls,
                  wall_seconds=time.monotonic()-started, simulation_only=True,
                  hardware_readiness=False, default_controller_replaced=False,
                  full_task_completed=all(v['nominal']['success'] and v['trial_count']==20
                                          and v['successes']>=18 for v in validation.values()),
                  hashes={str(f):digest(f) for f in (Path(__file__),scene,stand)})
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('STRICT VALIDATION', {k:v['successes'] for k,v in validation.items()},flush=True)


if __name__=='__main__':
    main()
