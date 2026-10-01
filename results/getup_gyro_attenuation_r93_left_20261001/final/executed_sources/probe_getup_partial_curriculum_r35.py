"""Find physically valid, recoverable tilt starts for an R35 curriculum.

This is a diagnostic, not a claim of recovery from a fully fallen pose. The
root is set only at reset, before simulation begins. During the settling and
recovery rollouts, the original scene and 50 Hz motor constraints are used.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from diagnostics.getup_independent_native import DT, POSES, canonical_quaternion, digest
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


BASE_ANGLE = {'prone': math.pi / 2, 'supine': -math.pi / 2,
              'left_side': -math.pi / 2, 'right_side': math.pi / 2}


def fractional_quaternion(pose, fraction):
    if pose not in POSES or not (0. <= fraction <= 1.):
        raise ValueError('Known fallen pose and tilt fraction in [0,1] required')
    return canonical_quaternion(pose, (fraction - 1.) * BASE_ANGLE[pose])


def prepare_partial(sim, pose, fraction, settle_controls, seed):
    """Generate a perturbed falling state, with no root edits after this reset."""
    mujoco.mj_resetDataKeyframe(sim.model, sim.data, sim.model.keyframe('home').id)
    rng = np.random.default_rng(seed)
    sim.data.qpos[3:7] = fractional_quaternion(pose, fraction)
    sim.data.qpos[sim.qadr] = np.clip(
        sim.home + rng.uniform(-.02, .02, sim.model.nu), sim.lower, sim.upper)
    sim.data.qpos[2] = 0.
    mujoco.mj_forward(sim.model, sim.data)
    sim.data.qpos[2] += .005 - sim.lowest_geom()
    sim.data.qvel[:] = rng.uniform(-.01, .01, sim.model.nv)
    sim.prev = sim.home.copy()
    sim.history = np.zeros_like(sim.history)
    mujoco.mj_forward(sim.model, sim.data)
    sim.clear_audit()
    for _ in range(settle_controls):
        sim.step_target(sim.home)
    initial = sim.measure()
    return initial, state_hash(sim), sim.physical_valid(), sim.peaks.copy()


def home_rollout(sim, duration_s):
    sim.clear_audit()
    tail = 0
    samples = []
    valid = True
    for control in range(round(duration_s / DT)):
        sim.step_target(sim.home)
        m = sim.measure()
        valid = sim.physical_valid()
        tail = tail + 1 if valid and m['stable'] else 0
        if control % 25 == 0 or not valid:
            samples.append({'t_s': round((control + 1) * DT, 3),
                            'up_z': m['up_z'], 'height_m': m['height_m'],
                            'foot_load_fraction': m['foot_load_fraction'],
                            'torso_contact': m['torso_contact'],
                            'strict_standing': m['stable']})
        if not valid:
            break
    return {'controls': control + 1, 'physical_valid': valid,
            'strict_tail_s': round(tail * DT, 6),
            'strict_1s_recovery': bool(valid and tail * DT >= 1.),
            'final': m, 'peaks': sim.peaks.copy(), 'samples': samples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fractions', type=float, nargs='+', default=[.2, .35, .5])
    p.add_argument('--settle-controls', type=int, nargs='+', default=[5, 10])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--duration-s', type=float, default=3.)
    args = p.parse_args()
    if args.duration_s < 1. or any(n < 0 for n in args.settle_controls):
        p.error('Need positive duration and nonnegative settling controls')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    rows = []
    for pose in POSES:
        for fraction in args.fractions:
            for settle in args.settle_controls:
                for seed in args.seeds:
                    initial, initial_hash, initial_valid, settle_peaks = prepare_partial(
                        sim, pose, fraction, settle, seed)
                    result = home_rollout(sim, args.duration_s) if initial_valid else None
                    row = {'pose': pose, 'tilt_fraction': fraction,
                           'settle_controls': settle, 'seed': seed,
                           'initial_hash': initial_hash,
                           'initial_physical_valid': initial_valid,
                           'initial': initial, 'settle_peaks': settle_peaks,
                           'rollout': result}
                    rows.append(row)
                    print(pose, fraction, settle, seed,
                          round(initial['up_z'], 3),
                          'valid' if initial_valid else 'invalid',
                          result and result['strict_1s_recovery'], flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'experiment': 'R35 partial-tilt HOME recoverability diagnostic',
              'created_at': datetime.now().isoformat(),
              'simulation_only': True, 'hardware_readiness': False,
              'full_fallen_recovery_claim': False,
              'unchanged_model_sha256': digest(scene),
              'standing_actor_sha256': digest(stand),
              'source_sha256': digest(__file__),
              'duration_s': args.duration_s, 'rows': rows}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
