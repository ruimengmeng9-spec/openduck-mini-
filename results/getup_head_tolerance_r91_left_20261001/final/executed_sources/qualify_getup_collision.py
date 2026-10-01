"""Qualify a collision candidate against normal stand and geometric references."""
import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from diagnostics.getup_dynamic_beam import SelfCollisionSim
from diagnostics.getup_independent_native import DT, sustained_tail, feature_targets, digest
from diagnostics.build_getup_decomposed import self_contacts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--workspace',type=Path,required=True)
    args=p.parse_args()
    sim=SelfCollisionSim(args.scene,args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    standing=[]
    for seed in range(5):
        sim.prepare('standing',17000+seed,perturb=seed>0)
        flags,metrics=[],[]
        for _ in range(250):
            sim.step_target(sim.home+.25*sim.stand_action())
            m=sim.measure()
            flags.append(m['stable'])
            metrics.append(m)
        stable=sustained_tail(flags)
        passed=(stable>=2. and min(m['up_z'] for m in metrics)>.9
                and max(m['self_penetration_m'] for m in metrics)<.004
                and max(m['floor_penetration_m'] for m in metrics)<.005
                and all(m['finite'] for m in metrics))
        row=dict(seed=17000+seed,passed=bool(passed),stable_tail_s=stable,
                 min_up_z=min(m['up_z'] for m in metrics),final=metrics[-1],
                 maximum_self_penetration_m=max(m['self_penetration_m'] for m in metrics),
                 maximum_floor_penetration_m=max(m['floor_penetration_m'] for m in metrics))
        standing.append(row)
        print(json.dumps(row),flush=True)
    geometry=[]
    for row in json.loads((args.workspace/'results.json').read_text())['rows']:
        m,d=sim.model,sim.data
        d.qpos[:]=m.keyframe('home').qpos
        angle=math.radians(row['pitch_deg'])
        d.qpos[:3]=[0.,0.,row['implied_root_height_m']]
        d.qpos[3:7]=[math.cos(angle/2),0.,math.sin(angle/2),0.]
        d.qpos[sim.qadr]=feature_targets([row['features_rad']],sim.home,sim.lower,sim.upper)[0]
        d.qvel[:]=0.
        mujoco.mj_forward(m,d)
        pairs=self_contacts(m,d)
        r=dict(pitch_deg=row['pitch_deg'],self_contact_pairs=pairs,
               maximum_self_overlap_m=max(pairs.values(),default=0.),
               prescribed_root_pose=True,dynamic_recovery_success=False)
        geometry.append(r)
        print(json.dumps(r),flush=True)
    summary=dict(standing_passed_runs=sum(r['passed'] for r in standing),standing_runs=len(standing),
                 standing=standing,geometry_candidates=geometry,hardware_readiness=False,
                 note='normal-standing qualification only; approximate CAD decomposition still needs hardware clearance review',
                 hashes={str(p):digest(p) for p in (args.scene,Path(__file__),args.workspace/'results.json')})
    (args.scene.parent/'qualification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
