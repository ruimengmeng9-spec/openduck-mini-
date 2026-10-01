"""Optimize R49's three-stage recovery from one perturbed fixed transition."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from diagnostics.getup_independent_native import digest
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, DURATIONS
from diagnostics.search_getup_support_margin_r49 import evaluate
from diagnostics.validate_getup_fixed_transition_r52 import capture_fixed
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--prior-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--transition-seed', type=int, required=True)
    p.add_argument('--generations', type=int, default=64)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--search-seed', type=int, default=153)
    args = p.parse_args()
    if args.output.exists() or args.generations < 1 or args.population < 8:
        p.error('Need new output, positive generations, population >=8')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    prior = np.load(args.prior_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, captured = capture_fixed(sim, reference, args.transition_seed)
    snapshot, peaks, finite = (captured.pop('snapshot'),
                               captured.pop('peaks'), captured.pop('finite'))
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    rng = np.random.default_rng(args.search_seed)
    mean, std = prior.copy(), np.full_like(prior, .35)
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
        ranked = [(x, evaluate(sim, snapshot, peaks, finite, x, ids)[0])
                  for x in population]
        ranked.sort(key=lambda pair: -pair[1]['score'])
        if best is None or ranked[0][1]['score'] > best[1]['score']:
            best = ranked[0]
        elites = np.stack([x for x, _ in ranked[:max(4, args.population // 5)]])
        mean = .5 * mean + .5 * elites.mean(axis=0)
        std = np.maximum(.1, .7 * std + .3 * elites.std(axis=0))
        row = {'generation': generation + 1,
               'valid_candidates': sum(r['valid'] for _, r in ranked),
               'generation_best': ranked[0][1], 'overall_best': best[1],
               'wall_seconds': time.monotonic() - started}
        history.append(row)
        np.savez_compressed(args.output / 'checkpoint.npz', mean=mean,
                            std=std, best_offsets=best[0], prior=prior)
        (args.output / 'search_history.json').write_text(
            json.dumps(history, indent=2), encoding='utf-8')
        b = best[1]
        print('GEN', generation + 1, 'SEED', args.transition_seed,
              'SCORE', round(b['score'], 2),
              'SUPPORT', round(b['supported_longest_s'], 2),
              'STRICT', round(b['strict_tail_s'], 2),
              'MARGIN', round(b['best']['support']['margin_m'], 3),
              'VALID', row['valid_candidates'], flush=True)
        if b['strict_tail_s'] >= 1.:
            break
    replay_prefix, replay_capture = capture_fixed(
        sim, reference, args.transition_seed)
    if replay_capture['state_hash'] != captured['state_hash']:
        raise RuntimeError('Perturbed transition replay was not deterministic')
    replay, trace = evaluate(sim, replay_capture['snapshot'],
                             replay_capture['peaks'], replay_capture['finite'],
                             best[0], ids, True)
    if abs(replay['score'] - best[1]['score']) > 1e-5:
        raise RuntimeError('Winner changed on actual-fall replay')
    np.savez_compressed(args.output / 'winner_trace.npz',
                        time=np.array([x[0] for x in trace]),
                        qpos=np.stack([x[1] for x in trace]),
                        qvel=np.stack([x[2] for x in trace]),
                        margin_m=np.array([x[3] for x in trace]),
                        supported=np.array([x[4] for x in trace]),
                        strict=np.array([x[5] for x in trace]),
                        phase=np.array([x[6] for x in trace]))
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'prior_checkpoint_sha256': digest(args.prior_checkpoint),
              'source_sha256': digest(__file__),
              'transition_seed': args.transition_seed,
              'joint_names': JOINTS, 'durations_s': DURATIONS,
              'prefix': prefix, 'captured': captured,
              'generations_completed': len(history),
              'best_offsets_rad': best[0].tolist(), 'best': best[1],
              'actual_fall_replay': replay, 'full_task_completed': False}
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
