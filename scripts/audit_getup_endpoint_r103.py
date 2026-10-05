"""Read-only complete-path R102 audit, no dynamics replay or gate changes."""
import json
from pathlib import Path
import numpy as np
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck/outputs')


def first_orientation(q,nominal,threshold):
    length=min(len(q),len(nominal));a=q[:length,3:7];b=nominal[:length,3:7]
    a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True)
    angles=2*np.arccos(np.clip(np.abs(np.sum(a*b,axis=1)),0.,1.))
    ids=np.flatnonzero(angles>threshold)
    return None if not len(ids) else float((ids[0]+1)*.02)


def main():
    run=ROOT/'getup_joint_anchor_r102_left_20261005';report=json.loads((run/'results.json').read_text())
    out=ROOT/'getup_endpoint_audit_r103_20261005';out.mkdir(exist_ok=False)
    with np.load(ROOT/'getup_path_audit_r99_20261005/frozen_path.npz') as z:
        phases=z['phases'];targets=z['targets'];reference=z['reference']
    with np.load(run/'development_baseline/case_None.npz') as z:nominal=z['qpos']
    rows=[];base={r['case_seed']:r for r in report['baseline']['rows']}
    for row in report['development']['rows']:
        seed=row['case_seed'];previous=base[seed]
        assert row['initial_hash']==previous['initial_hash']
        with np.load(run/f'development_candidate/case_{seed}.npz') as c:
            with np.load(run/f'development_baseline/case_{seed}.npz') as b:
                length=min(len(c['qpos']),len(b['qpos']))
                delta=np.abs(c['applied'][:length]-b['applied'][:length])
                item=dict(seed=seed,baseline_success=previous['success'],candidate_success=row['success'],
                    valid=row['valid'],entry_time_s=row['entry_time_s'],tail_s=row['strict_tail_s'],
                    initial_hash_match=True,first_orientation_divergence_from_nominal_s=first_orientation(c['qpos'],nominal,.15),
                    baseline_first_orientation_divergence_from_nominal_s=first_orientation(b['qpos'],nominal,.15),
                    maximum_applied_delta_by_phase={str(p):float(delta[phases[:length]==p].max())
                        for p in np.unique(phases[:length])},final=row['final'])
                rows.append(item)
    phase_stats=[]
    for p in np.unique(phases):
        ids=np.flatnonzero(phases==p);change=np.max(np.abs(np.diff(targets[ids],axis=0)),axis=1) if len(ids)>1 else np.zeros(1)
        phase_stats.append(dict(phase=int(p),start_s=float(ids[0]*.02),duration_s=float(len(ids)*.02),
            target_changes=int(np.count_nonzero(change>1e-8)),reference_feature_min=reference[ids].min(0).tolist(),
            reference_feature_max=reference[ids].max(0).tolist()))
    recoveries=[r['seed'] for r in rows if r['candidate_success'] and not r['baseline_success']]
    regressions=[r['seed'] for r in rows if not r['candidate_success'] and r['baseline_success']]
    result=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        baseline_successes=report['baseline']['successes'],candidate_successes=report['development']['successes'],
        nominal_success=report['development']['nominal_success'],physical_failures=report['development']['physical_failures'],
        candidate_promoted=report['candidate_promoted'],independent_qualification_run=report['qualification'] is not None,
        diagnostic_orientation_threshold_rad=.15,threshold_not_used_for_acceptance=True,
        rows=rows,phases=phase_stats,recoveries=recoveries,regressions=regressions,
        source_sha256=digest(Path(__file__)),result_sha256=digest(run/'results.json'))
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','phases')},indent=2))
    print('PHASES',json.dumps(phase_stats))
    print('FAILURE_TIMES',json.dumps([(r['seed'],r['baseline_first_orientation_divergence_from_nominal_s'],
        r['first_orientation_divergence_from_nominal_s']) for r in rows if not r['candidate_success']]))


if __name__=='__main__':main()
