"""Closed report and whitelisted saved target statistics; not a scalar audit."""
import json
import numpy as np
from diagnostics import probe_getup_phase_coordinate_r185b as run


def main():
    path=run.OUTPUT/'results.json'
    if not path.exists():print(json.dumps(dict(terminal_result_saved=False)));return
    result=json.loads(path.read_text());c=result['candidate'];b=result['baseline']
    baseline={r['case_seed']:r for r in b['rows']};stats=[]
    for row in c['rows']:
        case=row['case_seed']
        with np.load(run.OUTPUT/'development_candidate'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as z:
            raw=z['phase_reference_shift_rad'];pre=z['same_state_pre_slew_direct_new_rad'];post=z['planned_before_integration_rad']-z['same_state_unperturbed_planned_rad']
            first=np.flatnonzero(np.any(post!=0,axis=1))
            stats.append(dict(case=case,success=row['success'],valid=row['valid'],controls=row['controls'],raw_request_max_rad=float(np.abs(raw).max()),pre_slew_direct_max_rad=float(np.abs(pre).max()),post_slew_direct_max_rad=float(np.abs(post).max()),maximum_combined=row['maximum_combined_14_joint_correction_rad'],phase_coordinate_max=row['maximum_phase_offset_controls'],first_nonzero_direct=int(first[0]) if len(first) else None,peaks=row['peaks']))
    print(json.dumps(dict(terminal_result_saved=True,candidate=c['successes'],baseline=b['successes'],invalid=c['physical_failures'],standard=c['nominal_success'],gate=result['original_development_gate'],rescued=[r['case_seed'] for r in c['rows'] if r['success'] and not baseline[r['case_seed']]['success']],regressed=[r['case_seed'] for r in c['rows'] if baseline[r['case_seed']]['success'] and not r['success']],saved_statistics_not_independent_scalar_audit=True,rows=stats)),flush=True)


if __name__=='__main__':main()
