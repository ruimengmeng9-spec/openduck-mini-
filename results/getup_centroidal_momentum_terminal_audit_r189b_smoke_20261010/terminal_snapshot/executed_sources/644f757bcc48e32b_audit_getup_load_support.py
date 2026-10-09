"""Distinguish loaded foot support from mere foot-floor contact proximity."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from diagnostics.audit_getup_crouch import SupportSim
from diagnostics.getup_beam_reference import primitive
from diagnostics.getup_independent_native import digest


class LoadSupportSim(SupportSim):
    def measure(self):
        result = super().measure()
        by_body = {}
        for i, c in enumerate(self.data.contact):
            if self.floor not in c.geom or c.dist > .001:
                continue
            other = int(c.geom[1] if c.geom[0] == self.floor else c.geom[0])
            b = int(self.model.geom_bodyid[other])
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, f)
            by_body[b] = by_body.get(b, 0.) + max(float(f[0]), 0.)
        total = sum(by_body.values())
        feet = [by_body.get(b, 0.) for b in self.feet]
        result['floor_normal_force_sum_n'] = total
        result['foot_normal_forces_n'] = feet
        result['foot_load_fraction'] = sum(feet)/max(total, 1e-9)
        result['nonfoot_loaded_bodies'] = {
            self.model.body(b).name: force for b,force in by_body.items()
            if b not in self.feet and force > .05}
        result['stable'] = (result['stable'] and min(feet) > .1
                            and result['foot_load_fraction'] > .9
                            and min(result['foot_up_alignment_to_home']) > .85)
        return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    scene=args.root/'training/getup_decomposed_r4/model/scene.xml'
    sim=LoadSupportSim(scene,args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')
    rows=[]
    sim.prepare('standing')
    for _ in range(150):
        sim.step_target(sim.home+.25*sim.stand_action())
    rows.append(dict(case='official_standing',metric=sim.measure()))
    for name in ('getup_dynamic_r4b_prone','getup_crouch_r5_prone','getup_aligned_r6_prone'):
        ref=np.load(args.root/'training'/name/'best_reference.npz',allow_pickle=False)
        sim.prepare('prone')
        for target,duration in zip(ref['targets'],ref['durations_s']):
            primitive(sim,sim.snapshot(),target,float(duration))
        rows.append(dict(case=name,metric=sim.measure()))
    summary=dict(rows=rows,simulation_only=True,hardware_readiness=False,
                 root_pose_edits_after_initialization=0,
                 hashes={str(f):digest(f) for f in (Path(__file__),scene)})
    (args.output/'results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for r in rows:
        m=r['metric']
        print(json.dumps(dict(case=r['case'],foot_load_fraction=m['foot_load_fraction'],
                              nonfoot_loaded_bodies=m['nonfoot_loaded_bodies'])),flush=True)


if __name__=='__main__':
    main()
