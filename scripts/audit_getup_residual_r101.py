"""Offline R100 development regression audit; no simulation edits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path('/data/shijinsheng/open_duck')
RUN=ROOT/'outputs/getup_reference_residual_r100_left_20261005'


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(exist_ok=False)
    reference=ROOT/'outputs/getup_path_audit_r99_20261005/frozen_path.npz'
    with np.load(reference) as data:phases=data['phases'].copy()
    baseline=json.loads((RUN/'baseline_development/results.json').read_text())
    by_seed={r['case_seed']:r for r in baseline['rows']}
    report=dict(baseline_successes=baseline['successes'],nominal_success=baseline['nominal_success'],
                simulation_only=True,hardware_readiness=False,full_task_completed=False,
                development_only=True,reference_sha256=digest(reference),checkpoints={})
    for label in ('checkpoint_0016','checkpoint_0032','checkpoint_0048','checkpoint_0064'):
        source=RUN/(label+'_development')
        result=json.loads((source/'results.json').read_text())
        rows=[]
        for row in result['rows']:
            seed=row['case_seed'];base=by_seed[seed]
            with np.load(source/f'case_{seed}.npz') as d, np.load(RUN/'baseline_development'/f'case_{seed}.npz') as b:
                n=min(len(d['qpos']),len(b['qpos']),len(phases))
                # Diagnostic root quaternion distance only, not acceptance.
                a=d['qpos'][:n,3:7];bb=b['qpos'][:n,3:7]
                alignment=np.clip(np.abs(np.sum(a*bb,axis=1)),0.,1.)
                angle=2*np.arccos(alignment)
                delta=d['applied'][:n]-b['applied'][:n]
                crossing=np.flatnonzero(angle>.15)
                phase_deltas={int(k):float(np.abs(delta[phases[:n]==k]).max())
                              for k in np.unique(phases[:n])}
                rows.append(dict(seed=seed,baseline_success=base['success'],candidate_success=row['success'],
                    paired_initial_hash_equal=base['initial_hash']==row['initial_hash'],
                    candidate_valid=row['valid'],entry_time=row['entry_time_s'],
                    first_root_orientation_divergence_s=(int(crossing[0])+1)*.02 if len(crossing) else None,
                    max_applied_delta_by_phase_rad=phase_deltas))
        counts=dict(successes=result['successes'],nominal_success=result['nominal_success'],
                    regressions=[r['seed'] for r in rows if r['baseline_success'] and not r['candidate_success']],
                    recoveries=[r['seed'] for r in rows if not r['baseline_success'] and r['candidate_success']],
                    physical_failures=result['physical_failures'],rows=rows)
        report['checkpoints'][label]=counts
        print('R100_COMPARISON',label,json.dumps({k:v for k,v in counts.items() if k!='rows'}),flush=True)
    report['hashes']={str(p):digest(p) for p in [Path(__file__),RUN/'results.json',RUN/'training/final.npz',reference]}
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('R101_AUDIT',args.output,flush=True)


if __name__=='__main__':main()
