"""Training-only audit: IMU error, encoder error and residual saturation.

Save complete failed and successful physical rollouts. Does not use the R70
held-out seeds for fitting or change limits/standing gates.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_fullfall_tracking_r67 import full_reference
from diagnostics.search_getup_reference_feedback_r64 import rollout
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS
from diagnostics.validate_getup_fullpath_r27 import StrictSim

_CTX = None


class Recorder(StrictSim):
    def __init__(self, scene, stand):
        super().__init__(scene, stand)
        self.recorded_qpos = []

    def step_target(self, target):
        result = super().step_target(target)
        self.recorded_qpos.append(self.data.qpos.copy())
        return result


def init_worker(scene, stand, ck, targets, phases, ref, ref_qpos, directory):
    global _CTX
    sim = StrictSim(scene, stand)
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, ck, targets, phases, ref, ref_qpos, ids, Path(directory)


def evaluate(seed):
    sim, ck, targets, phases, ref, ref_qpos, ids, directory = _CTX
    sim.prepare('left_side', 0 if seed is None else seed, seed is not None)
    initial = sim.measure()
    sim.clear_audit()
    gains = np.concatenate([ck['prefix_gains'], ck['terminal_gains']])
    value, trace = rollout(sim, sim.snapshot(), sim.peaks.copy(), True,
                           targets, phases, ids, ref, gains, True)
    qpos = np.array([r[1] for r in trace])
    error = np.array([r[6] for r in trace])
    correction = np.array([r[7] for r in trace])
    phases_run = np.array([r[3] for r in trace])
    joint_error = qpos[:, sim.qadr] - ref_qpos[:len(qpos), sim.qadr]
    summaries = []
    for phase in np.unique(phases_run):
        mask = phases_run == phase
        summaries.append({'phase': int(phase), 'seconds': int(mask.sum()) * DT,
                          'mean_imu_tilt_error': float(np.linalg.norm(error[mask, :2], axis=1).mean()),
                          'max_imu_tilt_error': float(np.linalg.norm(error[mask, :2], axis=1).max()),
                          'mean_joint_error_rad': float(np.abs(joint_error[mask]).mean()),
                          'max_joint_error_rad': float(np.abs(joint_error[mask]).max()),
                          'residual_cap_control_fraction': float(np.any(np.abs(correction[mask]) >= .18 - 1e-8, axis=1).mean())})
    name = 'nominal' if seed is None else str(seed)
    np.savez_compressed(directory / f'trace_{name}.npz',
                        time=[r[0] for r in trace], qpos=qpos,
                        qvel=[r[2] for r in trace], phase=phases_run,
                        strict=[r[4] for r in trace], margin_m=[r[5] for r in trace],
                        imu_error=error, residual_rad=correction, joint_error_rad=joint_error)
    return {'seed': seed, 'initial': initial,
            'success': bool(value['valid'] and value['strict_tail_s'] >= 1.
                            and value['completed_steps'] == len(targets)),
            'result': value, 'phases': summaries}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    if args.output.exists() or args.workers < 1:
        p.error('Need fresh output and positive workers')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    with np.load(args.checkpoint, allow_pickle=False) as z:
        ck = {key: z[key] for key in z.files}
    sim = Recorder(scene, stand)
    sim.prepare('left_side', 0, False)
    sim.clear_audit()
    sim.recorded_qpos = []
    targets, phases, ref = full_reference(sim, sim.snapshot(), sim.peaks.copy(),
                                          ck['prefix_targets'], ck['prefix_phases'], ck['offsets'], 2.)
    ref_qpos = np.array(sim.recorded_qpos)
    if len(ref_qpos) != len(targets):
        raise RuntimeError('Encoder reference time alignment failed')
    args.output.mkdir(parents=True)
    seeds = [None] + list(range(769000, 769008)) + list(range(773000, 773016))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), ck, targets, phases,
                                       ref, ref_qpos, str(args.output))) as pool:
        rows = list(pool.map(evaluate, seeds, chunksize=1))
    report = {'simulation_only': True, 'hardware_readiness': False,
              'full_task_completed': False, 'training_only_audit': True,
              'source_sha256': digest(__file__), 'checkpoint_sha256': digest(args.checkpoint),
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_qpos_alignment': 'post-control reference versus post-control actual',
              'successes': sum(r['success'] for r in rows[1:]), 'trials': 24, 'results': rows}
    np.savez_compressed(args.output / 'reference.npz', targets=targets,
                        phases=phases, imu_features=ref, qpos_post=ref_qpos)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    for label, subset in [('passed', [r for r in rows[1:] if r['success']]),
                           ('failed', [r for r in rows[1:] if not r['success']])]:
        print(label, len(subset), flush=True)
        for phase in range(8):
            ps = [p for r in subset for p in r['phases'] if p['phase'] == phase]
            if ps:
                print('PHASE', phase, 'IMU', round(np.mean([p['mean_imu_tilt_error'] for p in ps]), 3),
                      'ENCODER', round(np.mean([p['mean_joint_error_rad'] for p in ps]), 3),
                      'CAP', round(np.mean([p['residual_cap_control_fraction'] for p in ps]), 3), flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
