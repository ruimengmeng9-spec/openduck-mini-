"""Coordinated catch sequence after a replayed actual-fall rotation peak.

The search shares R42's unchanged audited dynamics, motor limits and
tail-weighted objective, but branches later in the fully replayed path.
Only a complete actual-fall replay can verify the winning development path.
"""
import argparse
from pathlib import Path
import json
import time

import numpy as np

from diagnostics.getup_independent_native import digest
from diagnostics.search_getup_peak_catch_r43 import reach_peak
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, DURATIONS, evaluate
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--r42-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=48)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--seed', type=int, default=144)
    args = p.parse_args()
    if args.output.exists() or args.generations < 1 or args.population < 8:
        p.error('Need new output directory, positive generations, population >=8')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    prior = np.load(args.r42_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, first, peak = reach_peak(sim, reference, prior, 0)
    snapshot, peaks, finite = (peak.pop('snapshot'), peak.pop('peaks'),
                               peak.pop('finite'))
    joint_ids = np.array([sim.model.actuator(n).id for n in JOINTS])
    mean = prior.copy()
    std = np.full_like(mean, .4)
    rng = np.random.default_rng(args.seed)
    best = None
    history = []
    args.output.mkdir(parents=True)
    started = time.monotonic()
    for generation in range(args.generations):
        population = [mean.copy(), prior.copy(), np.zeros_like(mean)]
        if best is not None:
            population.append(best[0].copy())
        while len(population) < args.population:
            base = (best[0] if best is not None and len(population) % 3 == 0
                    else prior if len(population) % 4 == 0 else mean)
            population.append(np.clip(base + rng.normal(0., std), -.9, .9))
        ranked = [(x, evaluate(sim, snapshot, peaks, finite, x, joint_ids))
                  for x in population]
        ranked.sort(key=lambda pair: -pair[1]['score'])
        if best is None or ranked[0][1]['score'] > best[1]['score']:
            best = ranked[0]
        elites = np.stack([x for x, _ in ranked[:max(4, args.population // 5)]])
        mean = .5 * mean + .5 * elites.mean(axis=0)
        std = np.maximum(.12, .7 * std + .3 * elites.std(axis=0))
        row = {'generation': generation + 1,
               'valid_candidates': sum(r['valid'] for _, r in ranked),
               'generation_best': ranked[0][1], 'overall_best': best[1],
               'wall_seconds': time.monotonic() - started}
        history.append(row)
        np.savez_compressed(args.output / 'checkpoint.npz', mean=mean, std=std,
                            best_offsets=best[0], prior=prior)
        (args.output / 'search_history.json').write_text(
            json.dumps(history, indent=2), encoding='utf-8')
        print('GEN', generation + 1, 'SCORE', round(best[1]['score'], 3),
              'MAX_UP', round(best[1]['max_up_z'], 3),
              'TAIL', round(best[1]['strict_tail_s'], 3),
              'VALID', row['valid_candidates'], flush=True)
        if best[1]['strict_tail_s'] >= 1.:
            break
    replay_prefix, _, replay_peak = reach_peak(sim, reference, prior, 0)
    if replay_peak['hash'] != peak['hash']:
        raise RuntimeError('Actual-fall replay did not reproduce the branch')
    replay = evaluate(sim, replay_peak['snapshot'], replay_peak['peaks'],
                      replay_peak['finite'], best[0], joint_ids)
    if abs(replay['score'] - best[1]['score']) > 1e-5:
        raise RuntimeError('Winner changed after actual-fall replay')
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'r42_checkpoint_sha256': digest(args.r42_checkpoint),
              'source_sha256': digest(__file__),
              'joint_names': JOINTS, 'durations_s': DURATIONS,
              'prefix': prefix,
              'first': {k: v for k, v in first.items()
                        if k not in ('snapshot', 'peaks', 'finite')},
              'peak': peak, 'generations_completed': len(history),
              'best_offsets_rad': best[0].tolist(), 'best': best[1],
              'actual_fall_replay': replay, 'full_task_completed': False}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()

