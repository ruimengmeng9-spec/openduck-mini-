"""Paired long-hold validation of centered feedback against R59.

The forty R64 held-out seeds and forty entirely new seeds are replayed with
both zero gains and candidate gains. Continuous 30 s strict standing is
required at the end of a 35 s home hold. No learned physics changes.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import digest
from diagnostics.search_getup_reference_feedback_r64 import (
    targets_for, record_reference, rollout)
from diagnostics.search_getup_robust_transition_r57 import perturb_snapshot
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, capture
from diagnostics.validate_getup_fullpath_r27 import StrictSim


_CTX = None


def init_worker(scene, stand, source, peaks, finite, targets, phases, ref, gains):
    global _CTX
    sim = StrictSim(scene, stand)
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, source, peaks, finite, targets, phases, ref, gains, ids


def evaluate_seed(seed):
    sim, source, peaks, finite, targets, phases, ref, gains, ids = _CTX
    state = source if seed is None else perturb_snapshot(sim, source, seed)
    base, _ = rollout(sim, state, peaks, finite, targets, phases, ids, ref,
                       np.zeros((4, 8)))
    candidate, _ = rollout(sim, state, peaks, finite, targets, phases, ids, ref, gains)
    for row in (base, candidate):
        row['success'] = bool(row['valid'] and row['completed_steps'] == len(targets)
                               and row['strict_tail_s'] >= 30. - 1e-8)
    return {'seed': seed, 'baseline': base, 'candidate': candidate}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    if args.output.exists() or args.workers < 1:
        p.error('Need fresh output and positive worker count')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    checkpoint = np.load(args.checkpoint, allow_pickle=False)
    offsets, gains = checkpoint['offsets'], checkpoint['best_phase_gains']
    sim = StrictSim(scene, stand)
    prefix, captured = capture(sim, np.load(args.reference, allow_pickle=False), 0)
    source, peaks, finite = captured.pop('snapshot'), captured.pop('peaks'), captured.pop('finite')
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    targets, phases = targets_for(sim, offsets, ids, 35.)
    ref = record_reference(sim, source, peaks, finite, targets)
    n = len(checkpoint['reference_features'])
    if not np.array_equal(ref[:n], checkpoint['reference_features']):
        raise RuntimeError('Long-hold reference prefix differs from training')
    seeds = [None] + list(range(766000, 766040)) + list(range(767000, 767040))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), source, peaks,
                                       finite, targets, phases, ref, gains)) as pool:
        rows = []
        for row in pool.map(evaluate_seed, seeds, chunksize=1):
            rows.append(row)
            print('SEED', row['seed'], 'BASE', int(row['baseline']['success']),
                  'CANDIDATE', int(row['candidate']['success']),
                  'TAIL', round(row['candidate']['strict_tail_s'], 2), flush=True)
    groups = {}
    for name, subset in [('heldout_r64', rows[1:41]), ('fresh_r65', rows[41:]),
                          ('combined', rows[1:])]:
        groups[name] = {'trials': len(subset),
                        'baseline_successes': sum(x['baseline']['success'] for x in subset),
                        'candidate_successes': sum(x['candidate']['success'] for x in subset),
                        'wins': sum(x['candidate']['success'] and not x['baseline']['success'] for x in subset),
                        'regressions': sum(x['baseline']['success'] and not x['candidate']['success'] for x in subset)}
    args.output.mkdir(parents=True)
    report = {'simulation_only': True, 'hardware_readiness': False,
              'full_task_completed': False,
              'stage': 'paired captured-transition 35 s hold; not full perturbed fall',
              'source_sha256': digest(__file__), 'scene_sha256': digest(scene),
              'stand_sha256': digest(stand), 'checkpoint_sha256': digest(args.checkpoint),
              'reference_sha256': digest(args.reference), 'prefix': prefix,
              'captured': captured, 'reference_prefix_exact': True,
              'hold_seconds': 35., 'required_strict_seconds': 30.,
              'canonical': rows[0], 'groups': groups, 'results': rows[1:]}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('GROUPS', json.dumps(groups), flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
