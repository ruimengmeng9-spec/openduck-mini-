"""Fit a sensor-only wait selector from R77 training outcomes, not test seeds.

Saved states are read solely to reconstruct observations. This script does
not execute a rollout, inject a root state into a recovery, or qualify a skill.
Cross-validation is training-only evidence; it must not be called held-out
full-fall validation. The saved controller still needs prospective replay.
"""
import argparse
import json
from pathlib import Path
import shutil

import mujoco
import numpy as np

from diagnostics.getup_independent_native import digest


def fit(x, nominal):
    values = np.vstack([x, nominal])
    return values.mean(axis=0), np.maximum(values.std(axis=0), .03)


def predict(observation, x, y, nominal, k, margin):
    x, y = np.asarray(x), np.asarray(y)
    if (x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) == 0
            or k < 1 or margin < 0 or not np.isfinite(x).all()
            or not np.isfinite(observation).all() or not np.isfinite(nominal).all()):
        raise ValueError('Invalid finite sensor training data or selector settings')
    mean, scale = fit(x, nominal)
    # Nominal preservation is an explicit training anchor, not a seed exception.
    anchor_x = np.vstack([x, nominal])
    anchor_y = np.vstack([y, np.ones(y.shape[1])])
    distances = np.linalg.norm((anchor_x - observation) / scale, axis=1)
    neighbours = np.argsort(distances, kind='stable')[:min(k, len(anchor_x))]
    probability = anchor_y[neighbours].mean(axis=0)
    best = int(np.argmax(probability))
    # Column zero is the frozen two-second baseline. Avoid unsupported changes.
    if best != 0 and probability[best] <= probability[0] + margin + 1e-12:
        best = 0
    return best, probability


def cross_validate(x, y, nominal, k, margin):
    selections = []
    for i in range(len(x)):
        mask = np.arange(len(x)) != i
        selected, _ = predict(x[i], x[mask], y[mask], nominal, k, margin)
        selections.append(selected)
    selections = np.asarray(selections, dtype=int)
    return selections, int(y[np.arange(len(y)), selections].sum())


def observation(model, data, path):
    with np.load(path, allow_pickle=False) as trajectory:
        prefix = np.flatnonzero(trajectory['phase'] == 0)
        if len(prefix) != 308 or not np.array_equal(prefix, np.arange(308)):
            raise RuntimeError('Missing complete original early motion; cannot fit')
        qpos, qvel = trajectory['qpos'][prefix[-1]], trajectory['qvel'][prefix[-1]]
    # Offline measurement reconstruction only; never used as a rollout start.
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)
    def sensor(name):
        index = model.sensor(name).id
        a, n = model.sensor_adr[index], model.sensor_dim[index]
        return data.sensordata[a:a + n].copy()
    joints = model.actuator_trnid[:, 0]
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    home = model.keyframe('home').qpos[qa]
    result = np.concatenate([sensor('upvector'), .15 * sensor('gyro'),
                             qpos[qa] - home, .05 * qvel[va]])
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite wait-entry observation')
    return result, qpos.copy(), qvel.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Need a fresh output directory')
    report = json.loads((args.library / 'results.json').read_text())
    if not report['training_only'] or report['heldout_seeds_used']:
        parser.error('Need completed training-only R77 results')
    rows = [r for r in report['results'] if r['nominal_success']]
    assert rows[0]['wait_s'] == 2.
    seeds = report['training_seeds']
    assert len(seeds) == 24 and len(set(seeds)) == 24
    scene = Path('/data/shijinsheng/open_duck/training/getup_decomposed_r4/model/scene.xml')
    if digest(scene) != report['scene_sha256']:
        raise RuntimeError('Scene differs from the frozen R77 physical contract')
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    x, y = [], []
    hashes = {}
    for seed in seeds:
        base = args.library / 'wait_100' / f'train_{seed}.npz'
        feature, qpos, qvel = observation(model, data, base)
        x.append(feature)
        outcomes = []
        for row in rows:
            record = next(r for r in row['cases'] if r['seed'] == seed)
            outcomes.append(int(record['success']))
            path = args.library / f"wait_{round(row['wait_s'] / .02):03d}" / f'train_{seed}.npz'
            other, other_pos, other_vel = observation(model, data, path)
            if not (np.array_equal(qpos, other_pos) and np.array_equal(qvel, other_vel)
                    and np.array_equal(feature, other)):
                raise RuntimeError('Intervention changed pre-decision state')
            hashes[str(path.relative_to(args.library))] = digest(path)
        y.append(outcomes)
    nominal, _, _ = observation(model, data, args.library / 'wait_100/nominal.npz')
    x, y = np.array(x), np.array(y)
    settings = []
    for k in (1, 3, 5, 7):
        for margin in (0., .1, .2):
            selected, passed = cross_validate(x, y, nominal, k, margin)
            settings.append({'k': k, 'margin': margin, 'cv_successes': passed,
                             'changed_cases': int(np.count_nonzero(selected)),
                             'selected_columns': selected.tolist()})
    best = max(settings, key=lambda r: (r['cv_successes'], -r['changed_cases'], r['k'], r['margin']))
    baseline = int(y[:, 0].sum())
    prospective = best['cv_successes'] > baseline
    args.output.mkdir(parents=True)
    shutil.copy2(__file__, args.output / Path(__file__).name)
    np.savez_compressed(args.output / 'selector.npz', training_observations=x,
                        training_outcomes=y, nominal_anchor=nominal,
                        wait_options_s=[r['wait_s'] for r in rows],
                        neighbour_count=best['k'], improvement_margin=best['margin'])
    result = {'simulation_only': True, 'hardware_readiness': False,
              'full_task_completed': False, 'training_only': True,
              'no_physical_rollouts_executed': True,
              'method': 'sensor-only nearest-neighbour wait outcome prediction',
              'features': ['body_upvector_xyz', 'gyro_xyz_scaled_0.15',
                           'actuator_encoder_home_errors', 'actuator_velocities_scaled_0.05'],
              'decision_time': 'after original 308 early controls, before any wait or later motion',
              'training_seeds': seeds, 'heldout_seeds_used': [],
              'normalization_fitted_inside_each_cv_fold': True,
              'nominal_training_anchor_preserved': True,
              'cv_is_not_independent_validation': True,
              'baseline_training_successes': baseline,
              'library_training_oracle_successes': report['library_training_oracle_successes'],
              'selected': best, 'cv_settings': settings,
              'needs_prospective_fullpath_validation': prospective,
              'source_sha256': digest(__file__), 'scene_sha256': digest(scene),
              'library_result_sha256': digest(args.library / 'results.json'),
              'training_replay_hashes': hashes, 'selector_sha256': digest(args.output / 'selector.npz')}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print('TRAINING_CV', best['cv_successes'], '/24 BASELINE', baseline,
          'PROSPECTIVE_VALIDATION_JUSTIFIED', prospective, flush=True)


if __name__ == '__main__':
    main()
