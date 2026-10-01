"""Read-only preflight and prior-path substep collision diagnosis."""
import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from diagnostics.audit_getup_load_support import LoadSupportSim
from diagnostics.train_getup_fullpath_r27 import AuditedSim, initial_candidates, decode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    base, audited = LoadSupportSim(scene, stand), AuditedSim(scene, stand)
    base.prepare('standing'); audited.prepare('standing')
    max_qpos_error = 0.
    for i in range(100):
        target = base.home.copy()
        target[5] += .02*np.sin(i*.1)
        base.step_target(target); audited.step_target(target)
        max_qpos_error = max(max_qpos_error, float(np.abs(base.data.qpos-audited.data.qpos).max()))
    parity = dict(max_qpos_error=max_qpos_error, exact_state_parity=max_qpos_error==0.,
                  audited_stand_valid=audited.physical_valid(), stand_peaks=audited.peaks.copy())
    rows = []
    for pose in ('prone', 'supine', 'left_side', 'right_side'):
        sim = LoadSupportSim(scene, stand)
        sim.prepare(pose)
        initial = sim.measure()
        q, durations = decode(initial_candidates(args.root, sim, pose, 14)[0], sim)
        first_violation, peak, controls = None, 0., 0
        started = time.monotonic()
        # Explicit physical substeps diagnose which meshes violate the same gate.
        for target, duration in zip(q, durations):
            start = sim.prev.copy(); n = round(duration/.02)
            for i in range(n):
                a = min((i+1)/(.7*n), 1.); a = a*a*(3-2*a)
                desired = (1-a)*start+a*target
                applied = np.clip(desired, sim.prev-5.24*.02, sim.prev+5.24*.02)
                sim.data.ctrl[:] = applied; sim.prev = applied.copy()
                for _ in range(10):
                    mujoco.mj_step(sim.model, sim.data)
                    for c in sim.data.contact:
                        if sim.floor not in c.geom:
                            peak = max(peak, float(-c.dist))
                            if first_violation is None and c.dist < -.004:
                                names = [sim.model.geom(int(g)).name for g in c.geom]
                                bodies = [sim.model.body(int(sim.model.geom_bodyid[g])).name for g in c.geom]
                                first_violation = dict(time=float(sim.data.time), depth_m=float(-c.dist),
                                                       geom_names=names, body_names=bodies)
                controls += 1
            if first_violation:
                break
        rows.append(dict(pose=pose, initial=initial, first_substep_self_violation=first_violation,
                         maximum_sampled_self_penetration_m=peak, control_steps=controls,
                         wall_seconds=time.monotonic()-started))
    report = dict(parity=parity, rows=rows, simulation_only=True, hardware_readiness=False,
                  note='The old path is only a candidate; tighter sampling does not certify CAD collision geometry.')
    (args.output/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
