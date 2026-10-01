"""Revalidate saved search references using current, stricter code and hashes."""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics.getup_independent_native import RecoverySim, episode, digest
from diagnostics.getup_feedback_reference import run
from diagnostics.getup_beam_reference import replay


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--experiment', type=Path, required=True)
    p.add_argument('--seeds', type=int, default=20)
    args = p.parse_args()
    original = json.loads((args.experiment/'results.json').read_text())
    reference = np.load(args.experiment/'best_reference.npz', allow_pickle=False)
    scene = args.experiment/'model/scene.xml'
    stand = args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = RecoverySim(scene, stand)
    rows = []
    for i in range(args.seeds):
        seed = 12000+i
        if 'primitive_duration_s' in reference:
            result, trace = replay(sim, original['pose'], reference['targets'],
                                   float(reference['primitive_duration_s']), seed, i>0, record=i==0)
        else:
            initial = sim.prepare(original['pose'], seed, perturb=i>0)
            function = run if 'handoff' in original['method'] else episode
            result, trace = function(sim, initial, reference['targets'], float(reference['duration_s']), record=i==0)
            result['seed'] = seed
        rows.append(result)
        if trace and len(trace[0]) == 5:
            np.savez_compressed(args.experiment/'verified_trajectory.npz', time=[r[0] for r in trace],
                                qpos=[r[1] for r in trace], qvel=[r[2] for r in trace],
                                ctrl=[r[3] for r in trace], phase=[r[4] for r in trace])
    files = [Path(__file__), Path(__file__).with_name('getup_independent_native.py'),
             Path(__file__).with_name('getup_feedback_reference.py'), Path(__file__).with_name('getup_beam_reference.py'),
             stand, scene, args.experiment/'model/robot_ground_mesh.xml', args.experiment/'best_reference.npz']
    summary = dict(pose=original['pose'], method=original['method'],
                   successful_validation_runs=sum(r['success'] for r in rows), validation_runs=len(rows),
                   results=rows, hashes={str(f): digest(f) for f in files},
                   simulation_only=True, hardware_readiness=False, self_collision_validated=False,
                   neural_distillation_started=False, gate='sustained supported standing >=2s after standing-policy handoff')
    (args.experiment/'verified_results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(args.experiment.name, summary['successful_validation_runs'], '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
