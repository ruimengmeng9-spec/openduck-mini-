"""Development-only causal selector screen over the completed R87 library.

All action candidates were optimized on these 24 development cases. Leave-one-
out below is NOT independent validation of that library. Only pre-intervention
IMU and joint encoders are inputs; query outcomes never enter selector fitting.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from diagnostics.getup_independent_native import digest

SELECT_STEP = 25  # Observe before command 25; new head feedback is still zero.
FEATURES = ('imu', 'imu_history', 'joint', 'fusion')


def observation(errors, joint_positions, joint_velocities, feature):
    if feature not in FEATURES:
        raise ValueError('Unknown observable feature set')
    history = np.asarray(errors)[[5, 15, SELECT_STEP]].ravel()
    joint = np.r_[joint_positions, .15*np.asarray(joint_velocities)]
    return {'imu': np.asarray(errors)[SELECT_STEP], 'imu_history': history,
            'joint': joint, 'fusion': np.r_[history, joint]}[feature].copy()


def choose(query, x, y, params, neighbors, prior):
    """Fit scale and success utilities on reference rows only; prefer zero ties."""
    query, x, y, params = map(np.asarray, (query, x, y, params))
    if (x.ndim != 2 or query.shape != (x.shape[1],) or
            y.shape != (len(x), len(params)) or not 1 <= neighbors <= len(x) or
            not 0 <= prior <= 1 or not np.isfinite(x).all() or
            not np.isfinite(query).all()):
        raise ValueError('Invalid training-only selector inputs')
    # No query in normalization. Floor protects almost-constant encoder axes.
    scale = np.maximum(x.std(axis=0), .01)
    distances = np.linalg.norm((x-query)/scale, axis=1)
    selected = np.argsort(distances, kind='stable')[:neighbors]
    weight = 1/(.1+distances[selected])
    local = (y[selected]*weight[:, None]).sum(axis=0)/weight.sum()
    utility = (1-prior)*local + prior*y.mean(axis=0)
    tied = np.flatnonzero(utility >= utility.max()-1e-12)
    # Prefer the smallest actuation change, with baseline (index 0) first.
    return int(tied[np.argmin(np.linalg.norm(params[tied], axis=1))])


def screen(x, y, params, neighbors, prior):
    rows = []
    for i in range(len(x)):
        mask = np.arange(len(x)) != i
        arm = choose(x[i], x[mask], y[mask], params, neighbors, prior)
        rows.append({'index': i, 'selected_arm': arm, 'success': bool(y[i, arm]),
                     'baseline_success': bool(y[i, 0])})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): parser.error('Need fresh output')
    original = json.loads((args.input/'results.json').read_text())
    seeds = original['training_seeds']
    if len(seeds) != 24 or original['training_best']['successes'] != 13:
        raise RuntimeError('Unexpected R87 development dataset')
    baseline = {row['seed']: row['success'] for row in original['training_best']['cases']}
    library = [{'params': [0.]*8, 'cases': original['training_best']['cases'],
                'name': 'identity'}]
    seen = {tuple(library[0]['params'])}
    hashes = {'results.json': digest(args.input/'results.json')}
    for path in sorted((args.input/'candidates').glob('*/summary.json')):
        row = json.loads(path.read_text()); key = tuple(row['params'])
        if row['nominal_success'] and key not in seen:
            library.append({**row, 'name': str(path.relative_to(args.input))})
            seen.add(key); hashes[str(path.relative_to(args.input))] = digest(path)
    params = np.array([row['params'] for row in library])
    outcome = np.array([[{r['seed']: r['success'] for r in arm['cases']}[s]
                        for arm in library] for s in seeds], dtype=bool)
    assert np.array_equal(outcome[:, 0], [baseline[s] for s in seeds])
    sets = {name: [] for name in FEATURES}
    for seed in seeds:
        file = args.input/f'training_review/case_{seed}/baseline.npz'
        hashes[str(file.relative_to(args.input))] = digest(file)
        with np.load(file, allow_pickle=False) as z:
            # Trace qpos/qvel are post-command; row 24 is the state BEFORE 25.
            # Discard root position, root orientation and root velocity entirely.
            if z['qpos'].shape[1] != 21 or z['qvel'].shape[1] != 20:
                raise RuntimeError('Expected frozen 14-joint sensor layout')
            for name in FEATURES:
                sets[name].append(observation(z['imu_error'], z['qpos'][24, 7:],
                                             z['qvel'][24, 6:], name))
    sets = {k: np.stack(v) for k, v in sets.items()}
    rows = []
    for feature, x in sets.items():
        for neighbors in (1, 3, 5, 7):
            for prior in (0., .25, .5, 1.):
                predictions = screen(x, outcome, params, neighbors, prior)
                rows.append({'features': feature, 'neighbors': neighbors, 'prior': prior,
                             'successes': sum(r['success'] for r in predictions),
                             'rescues': [seeds[r['index']] for r in predictions if
                                         r['success'] and not r['baseline_success']],
                             'regressions': [seeds[r['index']] for r in predictions if
                                             r['baseline_success'] and not r['success']],
                             'predictions': predictions})
    best = max(rows, key=lambda r: (r['successes'], -len(r['regressions']),
                                   -list(FEATURES).index(r['features']), r['prior']))
    args.output.mkdir()
    np.savez_compressed(args.output/'development_library.npz',params=params,
                        outcomes=outcome, seeds=seeds, **sets)
    result = {'simulation_only': True, 'hardware_readiness': False,
              'full_task_completed': False, 'new_physical_rollouts': False,
              'qualification_run': False, 'heldout_seeds_used': False,
              'source_sha256': digest(__file__), 'input_hashes': hashes,
              'candidate_names': [r['name'] for r in library],
              'candidate_count': len(library), 'selection_step': SELECT_STEP,
              'selection_time_s': .5, 'baseline_successes': int(outcome[:,0].sum()),
              'oracle_union_not_policy': int(outcome.any(axis=1).sum()),
              'development_leave_one_out': rows, 'best_screen': best,
              'library_optimized_on_all_development_seeds': True,
              'not_independent_validation': True,
              'promotion_allowed': best['successes'] > int(outcome[:,0].sum())}
    (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: result[k] for k in ('candidate_count','baseline_successes',
                     'oracle_union_not_policy','promotion_allowed')}),flush=True)
    print('BEST_DEVELOPMENT_SCREEN',json.dumps({k:v for k,v in best.items()
                                               if k!='predictions'}),flush=True)


if __name__ == '__main__': main()
