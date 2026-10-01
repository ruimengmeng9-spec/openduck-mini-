"""Expand verified near-standing rescue trajectories; NOT PPO or fallen get-up.

Keep native physics and success gates unchanged. Search only new training starts;
only paired 4s + 30s rescues enter the atlas. Select radius on separate starts,
freeze it, and compare the R25 atlas on identical fresh test states.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from diagnostics import search_getup_rescue_r22 as rescue
from diagnostics import train_getup_sequence_r24 as seq
from diagnostics.distill_getup_rescue_r23 import physical_contract
from diagnostics.getup_independent_native import digest
from diagnostics.validate_getup_sequence_r25 import audit_atlas


def extend_library(previous, scanned, searched):
    by_seed = {r['seed']: r for r in searched}
    if len(by_seed) != len(searched):
        raise ValueError('duplicate search seed')
    if len({r['seed'] for r in scanned}) != len(scanned):
        raise ValueError('duplicate training seed')
    if set(by_seed) - {r['seed'] for r in scanned}:
        raise ValueError('search seed absent from training scan')
    anchors = []
    for row in scanned:
        seed = row['seed']
        if seed in previous['seed']:
            raise ValueError('training seed already in previous atlas')
        trial = by_seed.get(seed)
        kind = 'home_success' if row['success'] else 'unsupported_abstention'
        knots = np.zeros((2, 8))
        if trial:
            checked = (trial['baseline'], trial['selected'], trial['selected_long'])
            if len({r['initial_hash'] for r in checked} | {row['initial_hash']}) != 1:
                raise ValueError('search/scan initialization mismatch')
            if row['success']:
                raise ValueError('searched a successful baseline')
            if trial['selected']['success'] and trial['selected_long']['success'] and trial['selected_long']['success_4s']:
                kind = 'verified_rescue'
                knots = np.asarray(trial['knots'], dtype=float)
        if knots.shape != (2, 8) or not np.isfinite(knots).all() or np.any(np.abs(knots) > rescue.BOUNDS + 1e-8):
            raise ValueError('invalid rescue trajectory')
        anchors.append(dict(seed=int(seed), initial_hash=row['initial_hash'],
                            initial_obs=row['initial_obs'], kind=kind, knots=knots.tolist()))
    extra = dict(seed=np.asarray([r['seed'] for r in anchors]),
                 obs=np.asarray([r['initial_obs'] for r in anchors]),
                 knots=np.asarray([r['knots'] for r in anchors]),
                 rescue=np.asarray([r['kind'] == 'verified_rescue' for r in anchors], dtype=bool))
    library = {key: np.concatenate((previous[key], extra[key])) for key in ('seed', 'obs', 'knots', 'rescue')}
    assert library['obs'].shape == (len(library['seed']), 50)
    assert len(np.unique(library['seed'])) == len(library['seed'])
    return library, anchors


def compare_paired(new_block, old_block, radius, previous_radius):
    new = {r['seed']: r for r in new_block['results'] if r['radius'] == radius}
    old = {r['seed']: r for r in old_block['results'] if r['radius'] == previous_radius}
    if new.keys() != old.keys() or any(new[s]['initial_hash'] != old[s]['initial_hash'] for s in new):
        raise ValueError('R25/R26 test states differ')
    return dict(runs=len(new), r25_successes=sum(r['success'] for r in old.values()),
                r26_successes=sum(r['success'] for r in new.values()),
                gains_over_r25=sum(new[s]['success'] and not old[s]['success'] for s in new),
                losses_vs_r25=sum(old[s]['success'] and not new[s]['success'] for s in new))


def init_eval(contract, library, tilt):
    seq.init_worker(contract, library)
    seq._context[0].tilt_max = tilt


def evaluate(contract, library, radii, seeds, steps, tilt, workers):
    with ProcessPoolExecutor(max_workers=workers, initializer=init_eval,
                             initargs=(str(contract), str(library), tilt)) as pool:
        return seq.paired_block(pool, sorted(set(radii)), seeds, steps)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scan-seeds', type=int, default=64)
    p.add_argument('--failures', type=int, default=8)
    p.add_argument('--generations', type=int, default=16)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.scan_seeds <= 100 or not 1 <= args.failures <= 16 or not 1 <= args.workers <= 4:
        raise ValueError('invalid bounded training budget')
    if not 1 <= args.generations <= 24 or not 4 <= args.population <= 48:
        raise ValueError('invalid search budget')
    audit = audit_atlas(args.previous)
    previous_contract = args.previous / 'controller_contract.json'
    c = physical_contract(json.loads(args.contract.read_text()))
    prior = json.loads(previous_contract.read_text())
    assert c['scene_path'] == prior['scene_path'] and c['starts_rad'] == prior['starts_rad']
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(stage='near-standing recovery ONLY; NOT full fallen get-up',
                  training_method='per-start CEM trajectory search + nonparametric sequence atlas',
                  learned_neural_actor=False, simulation_only=True, hardware_readiness=False,
                  default_controller_replaced=False, physics_unchanged=True, success_gate_unchanged=True,
                  auto_resets=0, root_edits_after_reset=0, previous_audit=audit,
                  search_seed_base=470000, search_seed_count=args.scan_seeds,
                  selection_seed_base=480000, final_test_seed_bases=[490000, 500000, 510000],
                  budget=dict(failures=args.failures, generations=args.generations,
                              population=args.population, workers=args.workers), complete=False)
    source_dir = args.output / 'executed_sources'
    source_dir.mkdir()
    sources = [Path(__file__), Path(rescue.__file__), Path(seq.__file__),
               Path(sys.modules['diagnostics.getup_native_curriculum'].__file__),
               Path(sys.modules['diagnostics.getup_independent_native'].__file__),
               Path(sys.modules['diagnostics.distill_getup_rescue_r23'].__file__),
               Path(sys.modules['diagnostics.validate_getup_sequence_r25'].__file__)]
    for source in sources:
        shutil.copy2(source, source_dir / source.name)
    report['hashes'] = {str(f): digest(f) for f in sources + [args.contract, previous_contract,
                                                           args.previous / 'trajectory_library.npz', Path(c['scene_path'])]}
    output = args.output / 'results.json'

    def save():
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    search_dir = args.output / 'rescue_search'
    command = [sys.executable, '-u', '-m', 'diagnostics.search_getup_rescue_r22',
               '--contract', str(args.contract), '--output', str(search_dir),
               '--seed-base', '470000', '--scan-seeds', str(args.scan_seeds),
               '--failures', str(args.failures), '--population', str(args.population),
               '--generations', str(args.generations), '--workers', str(args.workers)]
    subprocess.run(command, check=True)
    search_report = json.loads((search_dir / 'results.json').read_text())
    scanned = json.loads((search_dir / 'baseline_scan.json').read_text())
    with np.load(args.previous / 'trajectory_library.npz', allow_pickle=False) as a:
        previous = {key: a[key] for key in a.files}
    library, anchors = extend_library(previous, scanned, search_report['results'])
    np.savez_compressed(args.output / 'trajectory_library.npz', **library)
    (args.output / 'new_training_anchors.json').write_text(json.dumps(anchors, indent=2), encoding='utf-8')
    report['search_control_steps'] = search_report['actual_control_steps']
    report['new_anchor_counts'] = {kind: sum(r['kind'] == kind for r in anchors)
                                 for kind in ('home_success', 'verified_rescue', 'unsupported_abstention')}
    report['library_rescue_anchors'] = int(library['rescue'].sum())
    report['library_anchors'] = len(library['seed'])
    c.update(stage=report['stage'], training_method=report['training_method'],
             initial_observation_size=50, feature_indices=seq.FEATURES.tolist(), feature_scales=seq.SCALES.tolist(),
             rescue_vs_home_distance_margin=.8, knots_s=[.2, .6, 1.2], training_seed_list=library['seed'].tolist(),
             learned_neural_actor=False, default_controller_replaced=False,
             abstention_anchors_are_success_labels=False)
    contract = args.output / 'controller_contract.json'
    contract.write_text(json.dumps(c, indent=2), encoding='utf-8')
    report['library_sha256'] = digest(args.output / 'trajectory_library.npz')
    selection = list(range(480000, 480060))
    blocks = [('independent_4s', 490000, 200, 200, .55),
              ('independent_30s', 500000, 40, 1500, .55),
              ('mild_4s', 510000, 100, 200, .25)]
    held_out = selection + [seed for _, base, count, _, _ in blocks for seed in range(base, base + count)]
    if set(held_out) & set(library['seed']):
        raise RuntimeError('training/evaluation contamination')
    save()
    print('R26 ANCHORS:', json.dumps(report['new_anchor_counts']), flush=True)
    report['selection_60'] = evaluate(contract, args.output / 'trajectory_library.npz',
                                    [0., .15, .3, .6, 1.2], selection, 200, .55, args.workers)
    radius = seq.select_radius(report['selection_60']['summary'])
    report['selected_radius'] = radius
    report['selected_radius_frozen_before_final_tests'] = True
    save()
    print('R26 FROZEN RADIUS:', radius, json.dumps(report['selection_60']['summary']), flush=True)
    for name, base, count, steps, tilt in blocks:
        seeds = list(range(base, base + count))
        new_block = evaluate(contract, args.output / 'trajectory_library.npz', [0., radius],
                             seeds, steps, tilt, args.workers)
        old_block = evaluate(previous_contract, args.previous / 'trajectory_library.npz', [0., audit['selected_radius']],
                             seeds, steps, tilt, args.workers)
        report[name] = dict(r26=new_block, r25=old_block,
                            comparison=compare_paired(new_block, old_block, radius, audit['selected_radius']),
                            tilt_max_rad=tilt, actual_control_steps=new_block['actual_control_steps'] + old_block['actual_control_steps'])
        save()
        print('R26 HELD OUT:', name, json.dumps(report[name]['comparison']), flush=True)
    report['adoption_gate_passed'] = bool(report['independent_4s']['comparison']['gains_over_r25'] > 0 and
                                         all(report[name]['comparison']['losses_vs_r25'] == 0 for name, *_ in blocks))
    report['actual_control_steps'] = report['search_control_steps'] + report['selection_60']['actual_control_steps'] + sum(report[name]['actual_control_steps'] for name, *_ in blocks)
    report['complete'] = True
    save()
    print('R26 COMPLETE:', json.dumps({k: report[k] for k in ('actual_control_steps', 'adoption_gate_passed', 'new_anchor_counts')}), flush=True)


if __name__ == '__main__':
    main()
