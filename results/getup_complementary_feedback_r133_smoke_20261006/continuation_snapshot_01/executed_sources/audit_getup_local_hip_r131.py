"""Read-only closed candidate sensitivity, no dynamics or label changes."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics.train_getup_local_hip_r130 import ROOT,OUTPUT as SOURCE
from diagnostics.getup_independent_native import digest

OUTPUT=ROOT/'outputs/getup_local_hip_audit_r131_20261006'


def main():
    OUTPUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    terminal=json.loads((SOURCE/'results.json').read_text());closed=json.loads((SOURCE/'training_closed.json').read_text())
    assert terminal['gains']==[0.]*6 and not terminal['original_development_gate']
    base=json.loads((SOURCE/'training/generation_0001/candidate_00/results.json').read_text())
    baseline={r['case_seed']:r for r in base['rows']};groups={};rows=[]
    for generation in closed['history']:
        for gains,folder in zip(generation['proposals'],generation['closed_trial_directories']):
            groups.setdefault(folder,gains)
    for folder,gains in groups.items():
        directory=Path(folder);report=json.loads((directory/'results.json').read_text())
        rescued=[];regressed=[];invalid=[];trace_rows=[]
        for r in report['rows']:
            case=r['case_seed'];old=baseline[case];assert old['initial_hash']==r['initial_hash']
            if case is not None and r['success'] and not old['success']:rescued.append(case)
            if case is not None and old['success'] and not r['success']:regressed.append(case)
            if not r['valid']:invalid.append(dict(case_seed=case,control_endpoint_s=r['controls']*.02,peaks=r['peaks']))
        # Two predeclared representatives: most short successes and best all-valid
        # nonzero proposal. All summary labels are retained for all 81 groups.
        rows.append(dict(directory=folder,gains=gains,successes=report['successes'],physical_failures=report['physical_failures'],
            return_sum=report['return_sum'],rescued=rescued,regressed=regressed,invalid=invalid,short_tail_only=True))
    nonzero=[r for r in rows if any(r['gains'])]
    first=max(nonzero,key=lambda r:(r['successes'],-r['physical_failures'],r['return_sum']))
    second=max((r for r in nonzero if r['physical_failures']==0),key=lambda r:(r['successes'],r['return_sum']))
    details=[]
    for group in (first,second):
        folder=Path(group['directory']);group_details=[]
        for marker in sorted(folder.glob('case_*/result.json')):
            row=json.loads(marker.read_text());case=row['case_seed']
            with np.load(marker.with_name('trajectory.npz'),allow_pickle=False) as z:
                extra=z['local_hip_extra_rad'];controls=len(extra)
                norms=np.abs(extra).max(1);active=np.flatnonzero(norms>0);near=np.flatnonzero(norms>=.18-1e-12)
                group_details.append(dict(case_seed=case,success=row['success'],valid=row['valid'],controls=controls,
                    first_nonzero_feedback_s=float(active[0]*.02) if len(active) else None,
                    first_near_cap_s=float(near[0]*.02) if len(near) else None,
                    max_extra_rad=float(norms.max()),max_early_extra_rad=float(norms[:45].max()),
                    physics_peaks=row['peaks'],trajectory_hash=digest(marker.with_name('trajectory.npz'))))
        details.append(dict(group=group,actual_feedback=group_details))
    a=terminal['candidate'];b=terminal['baseline'];equal=[]
    for r,s in zip(a['rows'],b['rows']):
        assert r['case_seed']==s['case_seed'] and r['initial_hash']==s['initial_hash']
        with np.load(SOURCE/'development_candidate'/f"case_{r['case_seed']}"/'trajectory.npz',allow_pickle=False) as x:
            with np.load(SOURCE/'development_baseline'/f"case_{r['case_seed']}"/'trajectory.npz',allow_pickle=False) as y:
                match={k:bool(np.array_equal(x[k],y[k])) for k in x.files};assert all(match.values())
        equal.append(dict(case_seed=r['case_seed'],fields_equal=match))
    result=dict(unique_groups=len(rows),unique_nonzero_groups=len(nonzero),groups=rows,representatives=details,
        complete_candidate_baseline_bitwise_equal=equal,read_only=True,no_dynamic_integration=True,
        short_tail_statistics_not_acceptance=True,no_saved_label_changes=True,no_unique_root_cause_claim=True,
        hashes={str(p):digest(p) for p in (Path(__file__),SOURCE/'results.json',SOURCE/'training_closed.json')})
    (OUTPUT/'results.json').write_text(json.dumps(result,indent=2))
    print('R131_TERMINAL',json.dumps(dict(groups=len(rows),best_nonzero={k:first[k] for k in ('successes','physical_failures','rescued','regressed')},best_valid_nonzero={k:second[k] for k in ('successes','physical_failures','rescued','regressed')})),flush=True)


if __name__=='__main__':main()
