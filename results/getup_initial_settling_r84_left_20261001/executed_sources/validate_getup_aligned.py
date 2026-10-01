"""Full-chain held-out replay with the SAME strict entry gate as R6 search."""
import argparse
import json
from pathlib import Path

import numpy as np

from diagnostics import getup_dynamic_beam as dynamic
from diagnostics.getup_aligned_extension import AlignedSim, strict_entry
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--experiment',type=Path,required=True)
    args=p.parse_args()
    original=json.loads((args.experiment/'results.json').read_text())
    reference=np.load(args.experiment/'best_reference.npz',allow_pickle=False)
    path=list(zip(reference['targets'],reference['durations_s']))
    scene=Path(original['scene_path'])
    sim=AlignedSim(scene,args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    dynamic.ready_to_handoff=strict_entry
    rows=[]
    for seed in range(20):
        result,trace=dynamic.replay(sim,original['pose'],path,19000+seed,seed>0,seed==0)
        rows.append(result)
        if trace:
            np.savez_compressed(args.experiment/'verified_trajectory.npz',time=[r[0] for r in trace],qpos=[r[1] for r in trace],
                                qvel=[r[2] for r in trace],ctrl=[r[3] for r in trace],phase=[r[4] for r in trace])
    scripts=[Path(__file__),Path(__file__).with_name('getup_aligned_extension.py'),
             Path(__file__).with_name('getup_dynamic_beam.py'),Path(__file__).with_name('getup_crouch_extension.py')]
    summary=dict(pose=original['pose'],successful_validation_runs=sum(r['success'] for r in rows),validation_runs=len(rows),
                 results=rows,simulation_only=True,hardware_readiness=False,strict_entry_gate=True,
                 root_pose_edits_after_initialization=0,hashes={str(p):digest(p) for p in [*scripts,scene,args.experiment/'best_reference.npz']})
    (args.experiment/'verified_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('STRICT FULL-CHAIN VALIDATION:',summary['successful_validation_runs'],'/20',flush=True)


if __name__=='__main__':
    main()
