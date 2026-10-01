"""Random shooting around the standing policy under exact motor dynamics.

This is a reachability diagnostic, not PPO and not hardware control. It never
alters the root after an episode reset or changes MuJoCo/motor parameters.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from diagnostics.getup_fullfallen_contract_r32 import decode_action
from diagnostics.getup_fullfallen_env_r32 import observation
from diagnostics.getup_independent_native import DT, POSES, digest
from diagnostics.probe_getup_partial_curriculum_r35 import prepare_partial
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_curriculum_r35 import BOUNDARY
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def run(sim, actor, pose, fraction, reset_seed, noise_seed, sigma, controls):
    if fraction == 1.:
        sim.prepare(pose, reset_seed, True)
        initial = sim.measure()
        initial_valid = sim.physical_valid()
    else:
        initial, _, initial_valid, _ = prepare_partial(
            sim, pose, fraction, 5, reset_seed)
    start_hash = state_hash(sim)
    sim.clear_audit()
    rng = np.random.default_rng(noise_seed)
    noise = np.zeros(14)
    tail = 0
    max_up = initial['up_z']
    max_height = initial['height_m']
    max_foot_load = initial['foot_load_fraction']
    trace = []
    valid = initial_valid
    m = initial
    count = 0
    if valid:
        for _ in range((controls + 4) // 5):
            obs = observation(sim)[None]
            mean_action = actor.run(None, {actor.get_inputs()[0].name: obs})[0][0]
            mean_latent = np.arctanh(np.clip(mean_action, -.999, .999))
            noise = .9 * noise + np.sqrt(1. - .9 ** 2) * rng.standard_normal(14)
            action = np.tanh(mean_latent + sigma * noise)
            target = decode_action(action, sim.lower, sim.upper)
            for _ in range(5):
                if count == controls:
                    break
                sim.step_target(target)
                count += 1
                m = sim.measure()
                valid = sim.physical_valid()
                tail = tail + 1 if valid and m['stable'] else 0
                max_up = max(max_up, m['up_z'])
                max_height = max(max_height, m['height_m'])
                max_foot_load = max(max_foot_load, m['foot_load_fraction'])
                trace.append((sim.data.qpos.copy(), sim.data.qvel.copy(),
                              float(sim.data.time), int(m['stable'])))
                if not valid:
                    break
            if not valid:
                break
    result = {'pose': pose, 'fraction': fraction, 'reset_seed': reset_seed,
              'noise_seed': noise_seed, 'sigma': sigma,
              'initial_hash': start_hash, 'initial_valid': initial_valid,
              'initial_up_z': initial['up_z'],
              'actual_fallen_start': bool(fraction == 1. and
                  initial['up_z'] < .5 and initial['torso_contact']),
              'controls': count, 'valid': valid,
              'strict_tail_s': round(tail * DT, 6),
              'strict_1s': bool(valid and tail * DT >= 1.),
              'max_up_z': max_up, 'max_height_m': max_height,
              'max_foot_load_fraction': max_foot_load,
              'final': m, 'peaks': sim.peaks.copy()}
    return result, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--poses', nargs='+', choices=POSES, default=['prone', 'supine'])
    p.add_argument('--include-full-falls', action='store_true')
    p.add_argument('--sigmas', nargs='+', type=float, default=[.05, .1, .2])
    p.add_argument('--rollouts', type=int, default=8)
    p.add_argument('--duration-s', type=float, default=5.)
    p.add_argument('--reset-seed', type=int, default=1300000)
    args = p.parse_args()
    if args.rollouts < 1 or args.duration_s < 1. or any(s < 0. for s in args.sigmas):
        p.error('Positive rollouts/duration and nonnegative sigmas required')
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = StrictSim(scene, stand)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    actor = ort.InferenceSession(str(args.model), sess_options=opts,
                                 providers=['CPUExecutionProvider'])
    rows = []
    best = {}
    for pose in args.poses:
        fractions = [BOUNDARY[pose]] + ([1.] if args.include_full_falls else [])
        for fraction in fractions:
            for sigma in args.sigmas:
                for index in range(args.rollouts):
                    noise_seed = 1500000 + index + 1000 * POSES.index(pose)
                    row, trace = run(sim, actor, pose, fraction, args.reset_seed,
                                     noise_seed, sigma, round(args.duration_s / DT))
                    rows.append(row)
                    key = (pose, fraction, sigma)
                    rank = (int(row['strict_1s']), row['strict_tail_s'],
                            row['max_up_z'], row['max_height_m'])
                    if key not in best or rank > best[key][0]:
                        best[key] = (rank, row, trace)
                    print(pose, fraction, sigma, index, 'valid', row['valid'],
                          'strict', row['strict_1s'], 'max_up',
                          round(row['max_up_z'], 3), flush=True)
    winners = []
    for (pose, fraction, sigma), (_, row, trace) in best.items():
        winners.append(row)
        if trace:
            name = f'{pose}_f{str(fraction).replace(".", "p")}_s{str(sigma).replace(".", "p")}.npz'
            np.savez_compressed(args.output / name,
                                qpos=np.stack([x[0] for x in trace]),
                                qvel=np.stack([x[1] for x in trace]),
                                time=np.array([x[2] for x in trace]),
                                stable=np.array([x[3] for x in trace]))
    report = {'experiment': 'R36 AR1 exploration reachability probe',
              'created_at': datetime.now().isoformat(),
              'model_sha256': digest(args.model), 'scene_sha256': digest(scene),
              'source_sha256': digest(__file__), 'simulation_only': True,
              'hardware_readiness': False, 'full_task_completed': False,
              'partial_tilts_not_full_fall_successes': True,
              'rows': rows, 'winners': winners}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
