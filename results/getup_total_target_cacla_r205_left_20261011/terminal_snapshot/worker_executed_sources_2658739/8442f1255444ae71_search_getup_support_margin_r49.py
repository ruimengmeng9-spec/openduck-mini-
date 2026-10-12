"""Search foot-supported COM placement from a replayed actual-side fall.

R44 reached a brief two-foot crouch but tipped while extending. This search
measures the ground contact polygon and robot subtree COM directly, and
selects for supported upright height rather than only base rotation. It
does not change scene physics, actuator limits, collision checks, or the
strict final standing criterion. Development uses seed 0; every selected
trajectory is replayed from the original fallen start.
"""
import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, DURATIONS, capture
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


HOME_HOLD_S = 2.


def signed_polygon_margin(points, p):
    """Signed minimum distance to a convex contact hull, in metres."""
    points = sorted(set(tuple(np.round(x, 8)) for x in points))
    p = np.asarray(p, dtype=float)
    if not points:
        return -.2
    if len(points) == 1:
        return -float(np.linalg.norm(p - points[0])) - .01

    def cross(a, b, c):
        return ((b[0] - a[0]) * (c[1] - a[1])
                - (b[1] - a[1]) * (c[0] - a[0]))

    lower = []
    for x in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], x) <= 0:
            lower.pop()
        lower.append(x)
    upper = []
    for x in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], x) <= 0:
            upper.pop()
        upper.append(x)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        a, b = np.asarray(points[0]), np.asarray(points[-1])
        ab = b - a
        t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0., 1.))
        return -float(np.linalg.norm(p - (a + t * ab))) - .01
    return min(float(cross(a, b, p) / np.linalg.norm(np.asarray(b) - a))
               for a, b in zip(hull, hull[1:] + hull[:1]))


def foot_support(sim):
    points = []
    force_by_foot = {b: 0. for b in sim.feet}
    for i, c in enumerate(sim.data.contact):
        if sim.floor not in c.geom or c.dist > .001:
            continue
        other = int(c.geom[1] if c.geom[0] == sim.floor else c.geom[0])
        body = int(sim.model.geom_bodyid[other])
        if body not in force_by_foot:
            continue
        wrench = np.zeros(6)
        mujoco.mj_contactForce(sim.model, sim.data, i, wrench)
        force = max(float(wrench[0]), 0.)
        if force < .2:
            continue
        force_by_foot[body] += force
        points.append(c.pos[:2].copy())
    com = sim.data.subtree_com[1].copy()
    return {'margin_m': signed_polygon_margin(points, com[:2]),
            'contact_count': len(points),
            'com_xyz_m': com.tolist(),
            'foot_contact_force_n': [force_by_foot[b] for b in sim.feet]}


def support_quality(m, support):
    up = float(np.clip((m['up_z'] - .75) / .25, 0., 1.))
    height = float(np.clip((m['height_m'] - .09) / .08, 0., 1.))
    forces = support['foot_contact_force_n']
    feet = float(np.clip(min(forces) / 5., 0., 1.))
    margin = float(np.clip((support['margin_m'] + .06) / .10, 0., 1.))
    flat = float(np.clip(min(m['foot_up_alignment_to_home']), 0., 1.))
    return (6 * up + 7 * height + 6 * feet + 12 * margin + 2 * flat
            - 5 * m['torso_contact']
            - min(m['angular_speed_rad_s'], 5.))


def supported_entry(m, support):
    return bool(m['up_z'] > .92 and m['height_m'] > .12
                and not m['torso_contact']
                and min(support['foot_contact_force_n']) > 2.
                and support['margin_m'] > -.005)


def evaluate(sim, snapshot, peaks, finite, offsets, ids, record=False):
    sim.restore(snapshot)
    sim.peaks, sim.finite = peaks.copy(), finite
    best = None
    supports = []
    strict = []
    trace = []
    valid = True
    phases = [(f'phase_{i}', seconds, row)
              for i, (seconds, row) in enumerate(zip(DURATIONS, offsets))]
    phases.append(('home_hold', HOME_HOLD_S, None))
    for phase, seconds, row in phases:
        target = sim.home.copy()
        if row is not None:
            target[ids] = np.clip(sim.home[ids] + row,
                                  sim.lower[ids], sim.upper[ids])
        for _ in range(round(seconds / DT)):
            sim.step_target(target)
            m = sim.measure()
            valid = sim.physical_valid()
            if not valid:
                break
            support = foot_support(sim)
            q = support_quality(m, support)
            if best is None or q > best['quality']:
                best = {'quality': q, 'time_s': float(sim.data.time),
                        'metric': m, 'support': support, 'phase': phase}
            supports.append(supported_entry(m, support))
            strict.append(bool(m['stable']))
            if record:
                trace.append((float(sim.data.time), sim.data.qpos.copy(),
                              sim.data.qvel.copy(), support['margin_m'],
                              int(supports[-1]), int(strict[-1]), phase))
        if not valid:
            break
    if not valid:
        return {'score': -100., 'valid': False,
                'supported_longest_s': 0., 'strict_tail_s': 0.,
                'final': m, 'peaks': sim.peaks.copy()}, trace
    longest = current = 0
    for passed in supports:
        current = current + 1 if passed else 0
        longest = max(longest, current)
    tail = 0
    for passed in reversed(strict):
        if not passed:
            break
        tail += 1
    final_support = foot_support(sim)
    score = (best['quality'] + 10 * min(longest * DT, 1.)
             + 20 * min(tail * DT, 1.))
    return {'score': score, 'valid': True,
            'supported_longest_s': longest * DT,
            'strict_tail_s': tail * DT,
            'best': best, 'final': m,
            'final_support': final_support,
            'peaks': sim.peaks.copy()}, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--r44-checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--generations', type=int, default=64)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--seed', type=int, default=149)
    args = p.parse_args()
    if args.output.exists() or args.generations < 1 or args.population < 8:
        p.error('Need new output directory, positive generations, population >=8')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    prior = np.load(args.r44_checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, captured = capture(sim, reference, 0)
    snapshot, peaks, finite = (captured.pop('snapshot'),
                               captured.pop('peaks'), captured.pop('finite'))
    ids = np.array([sim.model.actuator(n).id for n in JOINTS])
    rng = np.random.default_rng(args.seed)
    mean = prior.copy()
    std = np.full_like(mean, .3)
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
        print('GEN', generation + 1,
              'SCORE', round(b['score'], 2),
              'SUPPORT', round(b['supported_longest_s'], 2),
              'STRICT', round(b['strict_tail_s'], 2),
              'MARGIN', round(b['best']['support']['margin_m'], 3),
              'VALID', row['valid_candidates'], flush=True)
        if b['strict_tail_s'] >= 1.:
            break
    replay_prefix, replay_capture = capture(sim, reference, 0)
    if replay_capture['state_hash'] != captured['state_hash']:
        raise RuntimeError('Actual-fall replay did not reproduce search branch')
    replay, trace = evaluate(sim, replay_capture['snapshot'],
                             replay_capture['peaks'], replay_capture['finite'],
                             best[0], ids, True)
    if abs(replay['score'] - best[1]['score']) > 1e-5:
        raise RuntimeError('Winner changed after actual-fall replay')
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
              'r44_checkpoint_sha256': digest(args.r44_checkpoint),
              'source_sha256': digest(__file__), 'joint_names': JOINTS,
              'durations_s': DURATIONS, 'home_hold_s': HOME_HOLD_S,
              'prefix': prefix, 'captured': captured,
              'generations_completed': len(history),
              'best_offsets_rad': best[0].tolist(),
              'best': best[1], 'actual_fall_replay': replay,
              'full_task_completed': False}
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
