"""Long loaded-standing replay for the R49 support-margin winner.

The candidate is always reconstructed from the recorded actual left-side fall.
It must then hold the unchanged home target for 30 seconds under the original
scene, contacts, actuator limits, and strict loaded-standing measurement.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, DURATIONS, capture
from diagnostics.search_getup_support_margin_r49 import foot_support
from diagnostics.validate_getup_fullpath_r27 import StrictSim


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path,
                   default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hold-seconds', type=float, default=35.)
    p.add_argument('--required-strict-seconds', type=float, default=30.)
    args = p.parse_args()
    if (args.output.exists() or args.required_strict_seconds < 30.
            or args.hold_seconds < args.required_strict_seconds):
        p.error('Need a new output directory and >=30 s strict window')

    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    reference = np.load(args.reference, allow_pickle=False)
    offsets = np.load(args.checkpoint, allow_pickle=False)['best_offsets']
    sim = StrictSim(scene, stand)
    prefix, captured = capture(sim, reference, 0)
    captured_hash = captured['state_hash']
    replay_prefix, replay_capture = capture(sim, reference, 0)
    if replay_capture['state_hash'] != captured_hash:
        raise RuntimeError('Actual-fall capture was not deterministic')
    captured = replay_capture
    sim.restore(captured['snapshot'])
    sim.peaks, sim.finite = captured['peaks'].copy(), captured['finite']
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])

    times, qpos, qvel, strict, supported, margins, phases = [], [], [], [], [], [], []
    valid = True

    def record(phase):
        m = sim.measure()
        support = foot_support(sim)
        times.append(float(sim.data.time))
        qpos.append(sim.data.qpos.copy())
        qvel.append(sim.data.qvel.copy())
        strict.append(bool(m['stable']))
        supported.append(bool(m['up_z'] > .92 and m['height_m'] > .12
                              and not m['torso_contact']
                              and min(support['foot_contact_force_n']) > 2.
                              and support['margin_m'] > -.005))
        margins.append(support['margin_m'])
        phases.append(phase)
        return m, support

    for phase_i, (duration, row) in enumerate(zip(DURATIONS, offsets)):
        target = sim.home.copy()
        target[ids] = np.clip(sim.home[ids] + row,
                              sim.lower[ids], sim.upper[ids])
        for _ in range(round(duration / DT)):
            sim.step_target(target)
            valid = sim.physical_valid()
            m, support = record(f'phase_{phase_i}')
            if not valid:
                break
        if not valid:
            break

    hold_steps = round(args.hold_seconds / DT)
    hold_strict = []
    if valid:
        for step in range(hold_steps):
            sim.step_target(sim.home)
            valid = sim.physical_valid()
            m, support = record('home_hold')
            hold_strict.append(bool(m['stable']))
            if (step + 1) % round(5. / DT) == 0:
                print('HOLD', round((step + 1) * DT, 1),
                      'STRICT', int(m['stable']),
                      'UP', round(m['up_z'], 4),
                      'HEIGHT', round(m['height_m'], 4),
                      'MARGIN', round(support['margin_m'], 4), flush=True)
            if not valid:
                break

    tail = 0
    for passed in reversed(hold_strict):
        if not passed:
            break
        tail += 1
    final = sim.measure()
    final_support = foot_support(sim)
    success = bool(valid and len(hold_strict) == hold_steps
                   and tail * DT >= args.required_strict_seconds - DT)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'trajectory.npz',
                        time=np.asarray(times), qpos=np.stack(qpos),
                        qvel=np.stack(qvel), strict=np.asarray(strict),
                        supported=np.asarray(supported),
                        margin_m=np.asarray(margins), phase=np.asarray(phases))
    report = {
        'simulation_only': True,
        'hardware_readiness': False,
        'scene_sha256': digest(scene),
        'stand_sha256': digest(stand),
        'reference_sha256': digest(args.reference),
        'checkpoint_sha256': digest(args.checkpoint),
        'source_sha256': digest(__file__),
        'actual_fall_prefix': prefix,
        'captured_state_hash': captured_hash,
        'replay_capture_state_hash': replay_capture['state_hash'],
        'joint_names': JOINTS,
        'durations_s': DURATIONS,
        'best_offsets_rad': offsets.tolist(),
        'requested_home_hold_s': args.hold_seconds,
        'required_strict_standing_s': args.required_strict_seconds,
        'executed_home_hold_s': len(hold_strict) * DT,
        'strict_tail_s': tail * DT,
        'minimum_support_margin_m_during_hold': (
            float(np.min(np.asarray(margins)[np.asarray(phases) == 'home_hold']))
            if hold_strict else None),
        'physical_valid': valid,
        'final': final,
        'final_support': final_support,
        'success': success,
        'full_task_completed': False,
    }
    (args.output / 'results.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in (
        'executed_home_hold_s', 'strict_tail_s', 'physical_valid', 'success')},
        indent=2), flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
