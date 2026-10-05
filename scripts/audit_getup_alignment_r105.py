"""Read-only local phase matching on stored complete-development trajectories.

Searching a minimum necessarily reduces distance; that alone is not causal
evidence. No dynamics, state replay, controller or acceptance edits occur here.
Root orientation is used only for offline diagnostics, not online control.
"""
import json
from pathlib import Path
import numpy as np
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck')


def match(qpos,reference,radius=25):
    q=np.asarray(qpos);r=np.asarray(reference)
    assert q.shape[1]==r.shape[1] and q.shape[1]>=8
    a=q[:,3:7]/np.linalg.norm(q[:,3:7],axis=1,keepdims=True)
    b=r[:,3:7]/np.linalg.norm(r[:,3:7],axis=1,keepdims=True)
    offsets=[];clock=[];aligned=[]
    for k in range(min(len(q),529)):
        lo=max(0,k-radius);hi=min(529,len(r),k+radius+1)
        angles=2*np.arccos(np.clip(np.abs(b[lo:hi]@a[k]),0.,1.))
        joints=np.mean((r[lo:hi,7:]-q[k,7:])**2,axis=1)
        costs=angles**2+joints
        index=int(np.argmin(costs));j=lo+index
        offsets.append((j-k)*.02);clock.append(float(costs[k-lo]));aligned.append(float(costs[index]))
    return np.array(offsets),np.array(clock),np.array(aligned)


def main():
    run=ROOT/'outputs/getup_joint_anchor_r102_left_20261005'
    result=json.loads((run/'results.json').read_text())
    out=ROOT/'outputs/getup_alignment_audit_r105_20261005';out.mkdir(exist_ok=False)
    with np.load(run/'development_baseline/case_None.npz') as z:nominal=z['qpos'].copy()
    rows=[]
    for group,label in [('development_baseline','baseline'),('development_candidate','candidate')]:
        reports=result['baseline' if label=='baseline' else 'development']['rows']
        for row in reports:
            path=run/group/f"case_{row['case_seed']}.npz"
            with np.load(path) as z:offset,clock,aligned=match(z['qpos'],nominal)
            np.savez_compressed(out/f"{label}_{row['case_seed']}.npz",offset_s=offset,clock_cost=clock,aligned_cost=aligned)
            start,end=80,min(130,len(offset));slice_offset=offset[start:end]
            rows.append(dict(group=label,seed=row['case_seed'],success=row['success'],valid=row['valid'],
                early_median_offset_s=float(np.median(slice_offset)),
                early_boundary_fraction=float(np.mean(np.abs(slice_offset)>=.5-1e-9)),
                early_clock_cost=float(np.mean(clock[start:end])),early_aligned_cost=float(np.mean(aligned[start:end])),
                early_offset_std_s=float(np.std(slice_offset)),trace_sha256=digest(path)))
    summary={}
    for label in ('baseline','candidate'):
        summary[label]={}
        for success in (False,True):
            subset=[r for r in rows if r['group']==label and r['seed'] is not None and r['success']==success]
            summary[label]['success' if success else 'failure']=dict(count=len(subset),
                median_abs_offset_s=float(np.median([abs(r['early_median_offset_s']) for r in subset])),
                median_boundary_fraction=float(np.median([r['early_boundary_fraction'] for r in subset])),
                median_clock_cost=float(np.median([r['early_clock_cost'] for r in subset])),
                median_aligned_cost=float(np.median([r['early_aligned_cost'] for r in subset])))
    data=dict(rows=rows,summary=summary,radius_controls=25,diagnostic_window_s=[1.6,2.6],
        cost='quaternion_angle_squared + mean_joint_position_error_squared',
        stored_post_step_only=True,no_dynamics_run=True,no_reference_state_injection=True,
        not_causal_evidence=True,no_acceptance_labels_changed=True,qualification_seeds_unused=True,
        simulation_only=True,hardware_readiness=False,full_task_completed=False,
        source_sha256=digest(Path(__file__)),source_results_sha256=digest(run/'results.json'))
    (out/'results.json').write_text(json.dumps(data,indent=2))
    print('R105_SUMMARY',json.dumps(summary),flush=True)
    print('R105_ROWS',json.dumps([{k:v for k,v in r.items() if k!='trace_sha256'} for r in rows]),flush=True)


if __name__=='__main__':main()
