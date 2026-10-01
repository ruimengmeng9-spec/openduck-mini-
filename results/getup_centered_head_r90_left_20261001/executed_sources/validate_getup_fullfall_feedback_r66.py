"""Pilot full fallen-start replay of a fixed R59 prefix plus R64 feedback.

Choose the branch time once from the nominal training reference, then freeze
the exact 50 Hz command stream. Independent perturbed falls receive that same
stream, never a hindsight-selected branch or a reset to the nominal transient.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import DT, digest
from diagnostics.search_getup_reference_feedback_r64 import (
    targets_for, record_reference, rollout)
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS, capture
from diagnostics.train_getup_fullpath_r27 import state_hash
from diagnostics.validate_getup_fullpath_r27 import StrictSim


_CTX = None


class CommandRecorder(StrictSim):
    def __init__(self, scene, stand):
        super().__init__(scene, stand)
        self.commands = []

    def step_target(self, target):
        self.commands.append((float(self.data.time), np.asarray(target).copy()))
        return super().step_target(target)


def frozen_prefix(scene, stand, reference):
    sim = CommandRecorder(scene, stand)
    prefix, captured = capture(sim, reference, 0)
    end = captured['snapshot']['time']
    # The first forty commands settle the initial pose; prepare repeats them.
    commands = np.array([q for t, q in sim.commands
                         if t >= .8 - 1e-8 and t < end - DT / 2])
    sim.prepare('left_side', 0, False)
    sim.clear_audit()
    for q in commands:
        sim.step_target(q)
    if state_hash(sim) != captured['state_hash']:
        raise RuntimeError('Frozen nominal command stream does not reproduce capture')
    return commands, prefix, captured


def init_worker(scene, stand, prefix, targets, phases, ref, gains):
    global _CTX
    sim = StrictSim(scene, stand)
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    _CTX = sim, prefix, targets, phases, ref, gains, ids


def evaluate_seed(seed):
    sim, prefix, targets, phases, ref, gains, ids = _CTX
    nominal = seed is None
    sim.prepare('left_side', 0 if nominal else seed, not nominal)
    initial = sim.measure()
    initial_fallen = bool(initial['up_z'] < .5 and initial['torso_contact'])
    initial_hash = state_hash(sim)
    sim.clear_audit()
    completed = 0
    for q in prefix:
        sim.step_target(q)
        completed += 1
        if not sim.physical_valid():
            break
    transition = sim.measure()
    reached_hash = state_hash(sim)
    state, peaks, finite = sim.snapshot(), sim.peaks.copy(), sim.finite
    rows = []
    for label, feedback in [('baseline', np.zeros((4, 8))), ('candidate', gains)]:
        if initial_fallen and completed == len(prefix) and sim.physical_valid():
            row, _ = rollout(sim, state, peaks, finite, targets, phases,
                             ids, ref, feedback)
            row['success'] = bool(row['valid'] and row['completed_steps'] == len(targets)
                                   and row['strict_tail_s'] >= 30. - 1e-8)
        else:
            row = {'success': False, 'valid': bool(sim.physical_valid()), 'strict_tail_s': 0.,
                   'reason': 'prefix audit or actual-fallen gate failed'}
        rows.append((label, row))
        # Each branch starts from its own physically reached state, not nominal.
        sim.restore(state)
        sim.peaks, sim.finite = peaks.copy(), finite
        if state_hash(sim) != reached_hash:
            raise RuntimeError('Paired branch modified physically reached state')
    return {'seed': seed, 'nominal': nominal, 'initial': initial,
            'initial_fallen': initial_fallen, 'initial_hash': initial_hash,
            'prefix_steps': completed, 'transition': transition,
            'transition_hash': reached_hash, **dict(rows)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, default=20)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    if args.output.exists() or args.cases < 1 or args.workers < 1:
        p.error('Need fresh output, positive cases and workers')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    z = np.load(args.reference, allow_pickle=False)
    ck = np.load(args.checkpoint, allow_pickle=False)
    prefix, original, captured = frozen_prefix(scene, stand, z)
    sim = StrictSim(scene, stand)
    ids = np.array([sim.model.actuator(name).id for name in JOINTS])
    targets, phases = targets_for(sim, ck['offsets'], ids, 35.)
    ref = record_reference(sim, captured['snapshot'], captured['peaks'],
                           captured['finite'], targets)
    n = len(ck['reference_features'])
    if not np.array_equal(ref[:n], ck['reference_features']):
        raise RuntimeError('Nominal reference mismatch')
    seeds = [None] + list(range(768000, 768000 + args.cases))
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), prefix, targets,
                                       phases, ref, ck['best_phase_gains'])) as pool:
        for row in pool.map(evaluate_seed, seeds, chunksize=1):
            rows.append(row)
            print('FULL_FALL', row['seed'], 'BASE', int(row['baseline']['success']),
                  'CANDIDATE', int(row['candidate']['success']), flush=True)
    rs = rows[1:]
    summary = {'trials': len(rs),
               'baseline_successes': sum(r['baseline']['success'] for r in rs),
               'candidate_successes': sum(r['candidate']['success'] for r in rs),
               'wins': sum(r['candidate']['success'] and not r['baseline']['success'] for r in rs),
               'regressions': sum(r['baseline']['success'] and not r['candidate']['success'] for r in rs)}
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / 'frozen_prefix.npz', targets=prefix, dt_s=DT)
    report = {'simulation_only': True, 'hardware_readiness': False,
              'full_task_completed': False, 'pose': 'left_side',
              'stage': 'full fallen-start pilot, not multi-orientation qualification',
              'prefix_is_fixed': True, 'nominal_prefix_exact': True,
              'branch_selection_on_heldout': False,
              'perturbations_before_settling': {'tilt_rad': .05, 'joint_rad': .02, 'velocity': .01},
              'hold_seconds': 35., 'required_strict_seconds': 30.,
              'source_sha256': digest(__file__), 'checkpoint_sha256': digest(args.checkpoint),
              'scene_sha256': digest(scene), 'stand_sha256': digest(stand),
              'reference_sha256': digest(args.reference), 'original_prefix': original,
              'canonical': rows[0], 'summary': summary, 'results': rs}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('SUMMARY', json.dumps(summary), flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
