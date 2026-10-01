"""Training-only ablation of the arbitrary 31 s home wait in the R67 prefix.

Freeze the command stream and vary only the number of identical home commands.
All new references are physically rolled out; no nominal middle-state reset.
The eight seeds were already R67 training seeds, never independent test seeds.
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


def compress(prefix, recovery_n, home, wait_s):
    hold_end = recovery_n + round(31. / DT)
    if len(prefix) <= hold_end or not np.array_equal(
            prefix[recovery_n:hold_end], np.tile(home, (round(31. / DT), 1))):
        raise ValueError('Only the audited constant home wait may be compressed')
    wait_n = round(wait_s / DT)
    commands = np.concatenate([prefix[:recovery_n], np.tile(home, (wait_n, 1)),
                               prefix[hold_end:]])
    index = np.arange(len(commands))
    end = recovery_n + wait_n
    phases = np.where(index < recovery_n, 0,
                      np.where(index < end, 1,
                               np.where(index < end + round(.4 / DT), 2, 3)))
    return commands, phases


def init_worker(scene, stand, configurations, gains):
    global _CTX
    sim = StrictSim(scene, stand)
    cases = []
    for seed in [None] + list(range(769000, 769008)):
        sim.prepare('left_side', 0 if seed is None else seed, seed is not None)
        m = sim.measure()
        if not (m['up_z'] < .5 and m['torso_contact']):
            raise RuntimeError('A training seed did not initialize an actual fall')
        cases.append(sim.snapshot())
    sim.clear_audit()
    ids = np.array([sim.model.actuator(n).id for n in JOINTS])
    _CTX = sim, cases, sim.peaks.copy(), ids, configurations, gains


def evaluate(job):
    config_index, case_index = job
    sim, cases, peaks, ids, configs, gains = _CTX
    targets, phases, ref = configs[config_index]
    value, _ = rollout(sim, cases[case_index], peaks, True, targets, phases, ids, ref, gains)
    value['success'] = bool(value['valid'] and value['completed_steps'] == len(targets)
                            and value['strict_tail_s'] >= 1.)
    return config_index, case_index, value


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    if args.output.exists() or args.workers < 1:
        p.error('Need fresh output and positive workers')
    scene = args.root / 'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root / 'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    z = np.load(args.reference, allow_pickle=False)
    ck = np.load(args.checkpoint, allow_pickle=False)
    sim = StrictSim(scene, stand)
    sim.prepare('left_side', 0, False)
    sim.clear_audit()
    initial, peaks = sim.snapshot(), sim.peaks.copy()
    recovery_n = sum(round(float(d) / DT) for d in z['durations_s'])
    configurations, metadata = [], []
    for wait in [31., 5., 2., 1., .5, 0.]:
        prefix, pp = compress(ck['prefix_targets'], recovery_n, sim.home, wait)
        try:
            config = full_reference(sim, initial, peaks, prefix, pp, ck['offsets'], 2.)
        except RuntimeError as e:
            metadata.append({'wait_s': wait, 'rejected_reference': str(e)})
            continue
        metadata.append({'wait_s': wait, 'prefix_seconds': len(prefix) * DT,
                         'config_index': len(configurations)})
        configurations.append(config)
    gains = np.concatenate([ck['prefix_gains'], ck['terminal_gains']])
    jobs = [(i, j) for i in range(len(configurations)) for j in range(9)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), configurations, gains)) as pool:
        results = list(pool.map(evaluate, jobs, chunksize=1))
    for item in metadata:
        if 'config_index' not in item:
            continue
        rs = [v for i, j, v in results if i == item['config_index']]
        item.update(nominal=rs[0], training=rs[1:],
                    training_successes=sum(r['success'] for r in rs[1:]), trials=8)
        print('WAIT', item['wait_s'], 'NOMINAL', int(rs[0]['success']),
              'TRAIN', item['training_successes'], '/ 8', flush=True)
    baseline = next(x for x in metadata if x['wait_s'] == 31.)
    eligible = [x for x in metadata if 'nominal' in x and x['nominal']['success']
                and x['training_successes'] >= baseline['training_successes']]
    chosen = min(eligible, key=lambda x: (-x['training_successes'], x['wait_s'])) if eligible else None
    args.output.mkdir(parents=True)
    if chosen is not None:
        prefix, pp = compress(ck['prefix_targets'], recovery_n, sim.home, chosen['wait_s'])
        np.savez_compressed(args.output / 'selected_prefix.npz', prefix_targets=prefix,
                            prefix_phases=pp, prefix_gains=ck['prefix_gains'],
                            terminal_gains=ck['terminal_gains'], offsets=ck['offsets'],
                            wait_s=chosen['wait_s'])
    report = {'simulation_only': True, 'hardware_readiness': False,
              'full_task_completed': False, 'training_only_probe': True,
              'training_seeds': list(range(769000, 769008)),
              'source_sha256': digest(__file__), 'checkpoint_sha256': digest(args.checkpoint),
              'scene_sha256': digest(scene), 'reference_sha256': digest(args.reference),
              'configurations': metadata, 'selected_wait_s': None if chosen is None else chosen['wait_s']}
    (args.output / 'results.json').write_text(json.dumps(report, indent=2))
    print('SELECTED_WAIT', report['selected_wait_s'], flush=True)
    print('RESULTS_SAVED', args.output, flush=True)


if __name__ == '__main__':
    main()
