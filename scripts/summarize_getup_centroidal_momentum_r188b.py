"""Read saved causal signals and original labels, never re-label physics."""
import json
import numpy as np
from diagnostics import probe_getup_centroidal_momentum_r188b as run


def main():
    summary=[];pairings=[]
    for case in run.CASES:
        paths=[run.OUTPUT/'development_baseline'/f'case_{case}',run.OUTPUT/'development_candidate'/f'case_{case}']
        b,c=[json.loads((p/'result.json').read_text()) for p in paths]
        assert b['initial_hash']==c['initial_hash']
        with np.load(paths[1]/'trajectory.npz',allow_pickle=False) as z:
            request=z['momentum_request_rad'];pre=z['same_state_pre_slew_direct_new_rad']
            post=z['planned_before_integration_rad']-z['same_state_unperturbed_planned_rad']
            first=np.flatnonzero(np.any(post!=0,axis=1))
            summary.append(dict(case=case,controls=c['controls'],success=c['success'],valid=c['valid'],entry=c['entry_time_s'],tail=c['strict_tail_s'],raw_max_rad=float(np.abs(request).max()),pre_max_rad=float(np.abs(pre).max()),post_max_rad=float(np.abs(post).max()),total_max_rad=c['maximum_combined_14_joint_correction_rad'],first_nonzero_control=int(first[0]) if len(first) else None,momentum_error_max=float(np.abs(z['momentum_error_kg_m2_per_s']).max()),unconstrained_residual_max=float(np.abs(z['momentum_unconstrained_residual_kg_m2_per_s']).max()),peaks=c['peaks']))
        pairings.append(dict(case=case,baseline_success=b['success'],candidate_success=c['success'],candidate_valid=c['valid']))
    result=json.loads((run.OUTPUT/'results.json').read_text())
    print(json.dumps(dict(candidate_successes=result['candidate']['successes'],baseline_successes=result['baseline']['successes'],candidate_physical_failures=result['candidate']['physical_failures'],standard_success=result['candidate']['nominal_success'],gate=result['original_development_gate'],rescued=[p['case'] for p in pairings if p['candidate_success'] and not p['baseline_success']],regressed=[p['case'] for p in pairings if p['baseline_success'] and not p['candidate_success']],signals=summary),indent=2),flush=True)


if __name__=='__main__':main()
