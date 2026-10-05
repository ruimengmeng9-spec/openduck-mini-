"""Low-dimensional multi-joint continuation search from a real-fall replay.

Development candidates branch at a physically reached transient for speed.
The winner is replayed again from the original left-side fallen start. None
of the search scores replace the strict loaded-standing/30 s final gate.
"""
import argparse
from pathlib import Path
import json
import time

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_terminal_bridge_r39 import score as bridge_score
from diagnostics.train_getup_fullpath_r27 import rollout, state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


JOINTS = ('left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
          'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
          'neck_pitch', 'head_pitch')
DURATIONS = (.4, .6, .8)
HOME_HOLD_S = 2.


def capture(sim, reference, seed):
    prefix, _ = rollout(sim, 'left_side', reference['targets'],
                        reference['durations_s'], seed, False, 31.)
    if not prefix['valid'] or not prefix['initial_fallen']:
        raise RuntimeError('The original actual-fall prefix failed the physical audit')
    first = sim.home.copy()
    j = sim.model.actuator('right_hip_pitch').id
    first[j] = np.clip(first[j] - .6, sim.lower[j], sim.upper[j])
    selected = None
    for phase, duration, target in [('first', .4, first), ('return', 2., sim.home)]:
        for control in range(round(duration / DT)):
            sim.step_target(target)
            m = sim.measure()
            if not sim.physical_valid():
                raise RuntimeError('Known first pulse became physically invalid')
            if (m['up_z'] > .45 and not m['torso_contact']
                    and min(m['foot_normal_forces_n']) > 5.):
                value = bridge_score(m)
                if selected is None or value > selected['value']:
                    selected = {'phase': phase, 'control_index': control,
                                'value': value, 'metric': m,
                                'snapshot': sim.snapshot(),
                                'peaks': sim.peaks.copy(), 'finite': sim.finite,
                                'state_hash': state_hash(sim)}
    if selected is None:
        raise RuntimeError('Physically reached two-foot transient was not reproduced')
    return prefix, selected


def moment_quality(m):
    up = float(m['up_z'])
    flat = float(min(m['foot_up_alignment_to_home']))
    load = float(m['foot_load_fraction'])
    foot_force = float(min(m['foot_normal_forces_n']))
    height = float(np.clip((m['height_m'] - .12) / .05, 0., 1.))
    return (4 * up + 2 * flat + 3 * load + .1 * min(foot_force, 10.)
            + 2 * height - 2 * m['torso_contact']
            - .3 * min(m['angular_speed_rad_s'], 5.))


def evaluate(sim, snapshot, peaks, finite, offsets, joint_ids):
    sim.restore(snapshot)
    sim.peaks, sim.finite = peaks.copy(), finite
    qualities, strict = [], []
    max_up = -1.
    max_load_without_torso = 0.
    valid = True
    for duration, row in zip(DURATIONS, offsets):
        target = sim.home.copy()
        target[joint_ids] = np.clip(sim.home[joint_ids] + row,
                                    sim.lower[joint_ids], sim.upper[joint_ids])
        for _ in range(round(duration / DT)):
            sim.step_target(target)
            m = sim.measure()
            valid = sim.physical_valid()
            if not valid:
                break
            qualities.append(moment_quality(m))
            strict.append(bool(m['stable']))
            max_up = max(max_up, m['up_z'])
            if not m['torso_contact']:
                max_load_without_torso = max(max_load_without_torso,
                                             m['foot_load_fraction'])
        if not valid:
            break
    if valid:
        for _ in range(round(HOME_HOLD_S / DT)):
            sim.step_target(sim.home)
            m = sim.measure()
            valid = sim.physical_valid()
            if not valid:
                break
            qualities.append(moment_quality(m))
            strict.append(bool(m['stable']))
            max_up = max(max_up, m['up_z'])
            if not m['torso_contact']:
                max_load_without_torso = max(max_load_without_torso,
                                             m['foot_load_fraction'])
    if not valid:
        return {'score': -100., 'valid': False, 'max_up_z': max_up,
                'max_load_without_torso': max_load_without_torso,
                'strict_tail_s': 0., 'final': m, 'peaks': sim.peaks.copy()}
    tail_n = round(1. / DT)
    tail = qualities[-tail_n:]
    strict_tail = 0
    for passed in reversed(strict):
        if not passed:
            break
        strict_tail += 1
    value = (.4 * max(qualities) + 2. * float(np.mean(tail))
             + 4. * min(strict_tail * DT, 1.))
    return {'score': value, 'valid': True, 'max_up_z': max_up,
            'max_load_without_torso': max_load_without_torso,
            'strict_tail_s': strict_tail * DT, 'final': m,
            'peaks': sim.peaks.copy()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=48)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--seed', type=int, default=142)
    args = p.parse_args()
    if args.output.exists() or args.generations < 1 or args.population < 8:
        p.error('Need new output directory, positive generations and population >=8')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    sim = StrictSim(scene, stand)
    prefix, captured = capture(sim, reference, 0)
    snapshot, peaks, finite = (captured.pop('snapshot'), captured.pop('peaks'),
                               captured.pop('finite'))
    joint_ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    rng = np.random.default_rng(args.seed)
    mean = np.zeros((len(DURATIONS), len(JOINTS)))
    # The R40 second action is a physically measured prior, not a success label.
    prior = mean.copy()
    prior[0, JOINTS.index('right_hip_pitch')] = -.6
    prior[0, JOINTS.index('left_hip_pitch')] = .6
    std = np.full_like(mean, .4)
    best = None
    history = []
    args.output.mkdir(parents=True)
    started = time.monotonic()
    for generation in range(args.generations):
        population = [mean.copy(), prior.copy()]
        if best is not None:
            population.append(best[0].copy())
        while len(population) < args.population:
            if best is not None and len(population) % 3 == 0:
                base = best[0]
            elif len(population) % 4 == 0:
                base = prior
            else:
                base = mean
            population.append(np.clip(base + rng.normal(0., std), -.9, .9))
        ranked = []
        for x in population:
            result = evaluate(sim, snapshot, peaks, finite, x, joint_ids)
            ranked.append((x, result))
        ranked.sort(key=lambda item: -item[1]['score'])
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
        np.savez_compressed(args.output / 'checkpoint.npz', mean=mean,
                            std=std, best_offsets=best[0], prior=prior)
        (args.output / 'search_history.json').write_text(
            json.dumps(history, indent=2), encoding='utf-8')
        print('GEN', generation + 1, 'SCORE', round(best[1]['score'], 3),
              'MAX_UP', round(best[1]['max_up_z'], 3),
              'TAIL', round(best[1]['strict_tail_s'], 3),
              'VALID', row['valid_candidates'], flush=True)
        if best[1]['strict_tail_s'] >= 1.:
            break
    # Recreate the winning branch from the actual fall instead of evaluating
    # only a saved middle state. This verifies that the search snapshot was
    # not a hidden teleport or a stale-physics shortcut.
    replay_prefix, replay_capture = capture(sim, reference, 0)
    if replay_capture['state_hash'] != captured['state_hash']:
        raise RuntimeError('Actual-fall replay did not reproduce the search branch')
    replay_result = evaluate(sim, replay_capture['snapshot'],
                             replay_capture['peaks'], replay_capture['finite'],
                             best[0], joint_ids)
    if abs(replay_result['score'] - best[1]['score']) > 1e-5:
        raise RuntimeError('Winner changed after actual-fall replay')
    report = {'simulation_only': True, 'hardware_readiness': False,
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference),
              'source_sha256': digest(__file__),
              'joint_names': JOINTS, 'durations_s': DURATIONS,
              'home_hold_s': HOME_HOLD_S, 'prefix': prefix,
              'captured': captured, 'generations_completed': len(history),
              'best_offsets_rad': best[0].tolist(), 'best': best[1],
              'actual_fall_replay': replay_result,
              'full_task_completed': False}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
