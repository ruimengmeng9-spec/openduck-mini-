"""Training-only dwell-time probe after R71 found body, not joint, divergence.

Frozen commands, feedback gains, physics, and standing gates. Change only
terminal phase durations and regenerate the nominal sensor reference; reject
any timing without a physically successful canonical recovery.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_reference_feedback_r64 import record_reference, rollout
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS
from diagnostics.validate_getup_fullpath_r27 import StrictSim

_CTX = None


def timed_targets(sim, ck, durations, hold):
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    targets, phases = list(ck['prefix_targets'].copy()), list(ck['prefix_phases'])
    for phase, duration in enumerate((*durations, hold)):
        target = sim.home.copy()
        if phase < 3:
            target[ids] = np.clip(sim.home[ids] + ck['offsets'][phase],
                                  sim.lower[ids], sim.upper[ids])
        count = round(duration / DT)
        targets.extend([target.copy() for _ in range(count)])
        phases.extend([phase + 4] * count)
    return np.array(targets), np.array(phases)


def init_worker(scene, stand, ck):
    global _CTX
    sim = StrictSim(scene, stand)
    cases = []
    for seed in [None] + list(range(769000, 769008)) + list(range(773000, 773016)):
        sim.prepare('left_side', 0 if seed is None else seed, seed is not None)
        m = sim.measure()
        if not (m['up_z'] < .5 and m['torso_contact']):
            raise RuntimeError('Not actually fallen')
        sim.clear_audit()
        cases.append((sim.snapshot(), sim.peaks.copy()))
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, cases, ck, ids


def evaluate(durations):
    sim, cases, ck, ids = _CTX
    targets, phases = timed_targets(sim, ck, durations, 2.)
    try:
        ref = record_reference(sim, *cases[0], True, targets)
    except RuntimeError as error:
        return {'durations_s': list(durations), 'nominal_success': False,
                'rejected': str(error), 'successes': 0}
    gains = np.concatenate([ck['prefix_gains'], ck['terminal_gains']])
    nominal, _ = rollout(sim, *cases[0], True, targets, phases, ids, ref, gains)
    passed = lambda r: bool(r['valid'] and r['completed_steps'] == len(targets)
                           and r['strict_tail_s'] >= 1.)
    if not passed(nominal):
        return {'durations_s': list(durations), 'nominal_success': False,
                'nominal': nominal, 'successes': 0}
    rows = [rollout(sim, state, peaks, True, targets, phases, ids, ref, gains)[0]
            for state, peaks in cases[1:]]
    return {'durations_s': list(durations), 'nominal_success': True,
            'nominal': nominal, 'successes': sum(passed(r) for r in rows),
            'valid_cases': sum(r['valid'] for r in rows), 'cases': rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    if args.output.exists() or args.workers < 1:
        p.error('Need fresh output and positive workers')
    root = Path('/data/shijinsheng/open_duck')
    scene = root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    with np.load(args.checkpoint, allow_pickle=False) as z:
        ck = {k: z[k] for k in z.files}
    args.output.mkdir(parents=True)
    grid = [(.4, .6, .8)] + [(.4, d5, d6) for d5 in [.36, .46, .6, .74, .9]
                             for d6 in [.6, .8, 1.] if (d5, d6) != (.6, .8)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), ck)) as pool:
        for row in pool.map(evaluate, grid, chunksize=1):
            rows.append(row)
            print('TIMING', row['durations_s'], 'NOMINAL', row['nominal_success'],
                  'SUCCESS', row['successes'], '/24', flush=True)
    best = max([r for r in rows if r['nominal_success']],
               key=lambda r: (r['successes'], -sum(r['durations_s'])))
    np.savez_compressed(args.output / 'selected.npz', **ck,
                        terminal_durations_s=best['durations_s'])
    report = {'simulation_only': True, 'hardware_readiness': False,
              'full_task_completed': False, 'training_only': True,
              'hypothesis': 'fixed terminal timing misses the body/contact transition',
              'source_sha256': digest(__file__), 'scene_sha256': digest(scene),
              'stand_sha256': digest(stand), 'checkpoint_sha256': digest(args.checkpoint),
              'training_seeds': list(range(769000, 769008)) + list(range(773000, 773016)),
              'frozen': ['targets', 'gains', 'physics', 'strict_standing_gate'],
              'nominal_imu_reference_regenerated_per_timing': True,
              'baseline_successes': rows[0]['successes'],
              'selected_durations_s': best['durations_s'],
              'selected_training_successes': best['successes'], 'results': rows}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
