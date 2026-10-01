"""Offline training-trajectory audit. No simulator/contact-force reconstruction.

The floating-base quaternion's world vertical projection and root height are
computed from saved post-control qpos. This is not contact or IMU calibration
evidence. No held-out trajectory is used to select new controller parameters.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from diagnostics.getup_independent_native import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): p.error('Need fresh audit output')
    report_path = args.input / 'results.json'
    report = json.loads(report_path.read_text())
    rows = []
    for item in report['results']:
        name = 'nominal' if item['seed'] is None else str(item['seed'])
        path = args.input / f'trace_{name}.npz'
        with np.load(path, allow_pickle=False) as z:
            qpos, phases = z['qpos'], z['phase']
            q = qpos[:, 3:7]
            if not np.allclose(np.linalg.norm(q, axis=1), 1., atol=1e-6):
                raise RuntimeError('Invalid root quaternion in trajectory')
            up = 1. - 2. * (q[:, 1] ** 2 + q[:, 2] ** 2)
            ends = []
            for phase in np.unique(phases):
                indices = np.flatnonzero(phases == phase)
                k = indices[-1]
                ends.append({'phase': int(phase), 'root_up_z': float(up[k]),
                             'root_height_m': float(qpos[k, 2]),
                             'minimum_root_up_z': float(up[indices].min())})
        wait = next(r for r in ends if r['phase'] == 1)
        rows.append({'seed': item['seed'], 'short_training_success': item['success'],
                     'root_tilted_at_wait_end': wait['root_up_z'] < .5,
                     'trace_sha256': digest(path), 'phase_ends': ends})
    failed = [r for r in rows if r['seed'] is not None and not r['short_training_success']]
    output = {'simulation_only': True, 'hardware_readiness': False,
              'training_only_audit': True, 'source_sha256': digest(__file__),
              'input_report_sha256': digest(report_path),
              'metric': 'world vertical projection of floating-base +Z; post-control saved qpos',
              'not_contact_force_evidence': True,
              'failed_training_cases': len(failed),
              'failed_tilted_at_wait_end': sum(r['root_tilted_at_wait_end'] for r in failed),
              'results': rows}
    args.output.mkdir(parents=True)
    (args.output / 'results.json').write_text(json.dumps(output, indent=2))
    print('FAILED_WAIT_TILT', output['failed_tilted_at_wait_end'], '/', len(failed), flush=True)


if __name__ == '__main__': main()
