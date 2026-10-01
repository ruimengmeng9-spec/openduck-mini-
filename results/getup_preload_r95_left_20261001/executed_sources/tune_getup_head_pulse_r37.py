"""Tune the brief neck/head pulse across distinct supine partial-tilt starts.

This is a physical curriculum-bridge search. It cannot prove fully fallen
recovery, which remains governed by the separate full-task acceptance test.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from diagnostics.getup_independent_native import DT, digest
from diagnostics.validate_getup_boundary_pair_r36 import evaluate
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--offsets', nargs='+', type=float, default=[.15, .25, .35, .45])
    p.add_argument('--pulse-s', nargs='+', type=float, default=[.3, .5, .7])
    p.add_argument('--fractions', nargs='+', type=float, default=[.10, .15])
    p.add_argument('--seeds', nargs='+', type=int, default=[1300000, 1300001, 1300002])
    p.add_argument('--duration-s', type=float, default=4.)
    args = p.parse_args()
    if any(x <= 0. for x in args.offsets + args.pulse_s) or any(
            not 0. < f < 1. for f in args.fractions):
        p.error('Positive offsets/pulses and partial-tilt fractions in (0,1) required')
    if args.duration_s <= max(args.pulse_s):
        p.error('Duration must exceed every pulse')
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    candidates = []
    for neck in args.offsets:
        for head in args.offsets:
            for pulse_s in args.pulse_s:
                rows = []
                for fraction in args.fractions:
                    for seed in args.seeds:
                        row, _ = evaluate(sim, 'supine', fraction, seed,
                                          [(5, neck), (6, head)],
                                          round(pulse_s / DT),
                                          round(args.duration_s / DT))
                        rows.append(row)
                successes = {str(f): sum(r['strict_1s'] for r in rows
                                         if r['tilt_fraction'] == f)
                             for f in args.fractions}
                candidate = {'neck_offset_rad': neck, 'head_offset_rad': head,
                             'pulse_s': pulse_s, 'successes_by_fraction': successes,
                             'physical_failures': sum(not r['physical_valid'] for r in rows),
                             'mean_longest_strict_s': sum(r['longest_strict_s'] for r in rows)
                                                      / len(rows),
                             'rows': rows}
                candidates.append(candidate)
                print('neck', neck, 'head', head, 'pulse', pulse_s,
                      'successes', successes, flush=True)
                report = {'experiment': 'R37 supine partial-tilt pulse tuning',
                          'created_at': datetime.now().isoformat(),
                          'scene_sha256': digest(scene), 'script_sha256': digest(__file__),
                          'simulation_only': True, 'hardware_readiness': False,
                          'full_fall_recovery_claim': False, 'candidates': candidates}
                (args.output / 'results.json').write_text(
                    json.dumps(report, indent=2), encoding='utf-8')
    hardest = max(args.fractions)
    best = max(candidates, key=lambda x: (
        x['successes_by_fraction'][str(hardest)],
        sum(x['successes_by_fraction'].values()),
        x['mean_longest_strict_s'], -x['physical_failures']))
    report['best'] = best
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('BEST', best['neck_offset_rad'], best['head_offset_rad'],
          best['pulse_s'], best['successes_by_fraction'], flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
