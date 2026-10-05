"""Read-only prediction-error and warning-specificity analysis of R137."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics.audit_getup_sensor_geometry_r137 import ROOT,OUTPUT as SOURCE
from diagnostics.getup_independent_native import digest

OUTPUT=ROOT/'outputs/getup_geometry_prediction_r138_20261006'


def main():
    OUTPUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    original=json.loads((SOURCE/'results.json').read_text());records=[]
    for row in original['rows']:
        folder=SOURCE/(row['label']+'_p'+str(row['program'])+'_'+str(row['case_seed']))
        with np.load(folder/'causal_geometry.npz',allow_pickle=False) as z:g=z['geometry_features'].copy()
        assert g.ndim==3 and g.shape[1:]==(4,3)
        # Compare .02s forecast from control k with real control k+1 sensor FK.
        # Future FK is an OFFLINE evaluation target, not a risk/model input.
        err=g[:-1,3]-g[1:,0]
        warnings=[]
        for channel,name in enumerate(('trunk_right_knee_pair','trunk_head_pair','all_self_contacts')):
            for threshold in (.002,.004):
                now=(-g[:,0,channel] if channel<2 else g[:,0,channel])>=threshold
                projected=(-g[:,3,channel] if channel<2 else g[:,3,channel])>=threshold
                actual_next=now[1:];forecast=projected[:-1]
                warnings.append(dict(channel=name,diagnostic_threshold_m=threshold,
                    current_indices=np.flatnonzero(now).tolist(),projected_indices=np.flatnonzero(projected).tolist(),
                    predicted_warning_without_next_endpoint_warning=int(np.sum(forecast&~actual_next)),
                    next_endpoint_warning_without_predicted_warning=int(np.sum(~forecast&actual_next)),
                    warned_before_terminating_control=bool(not row['saved_valid'] and np.any(projected[:row['terminal_control']])),
                    predicted_at_final_available_control=bool(projected[-1]),
                    current_at_final_available_control=bool(now[-1])))
        records.append(dict(label=row['label'],program=row['program'],case_seed=row['case_seed'],saved_valid=row['saved_valid'],
            saved_success=row['saved_success'],frames=len(g),terminal_control=row['terminal_control'],
            forecast_distance_abs_error_median_m=np.median(np.abs(err),axis=0).tolist(),
            forecast_distance_abs_error_p95_m=np.quantile(np.abs(err),.95,axis=0).tolist(),
            forecast_distance_abs_error_max_m=np.abs(err).max(axis=0).tolist(),warnings=warnings,
            source_hash=digest(folder/'causal_geometry.npz')))
    summary=[]
    for channel in ('trunk_right_knee_pair','trunk_head_pair','all_self_contacts'):
        for threshold in (.002,.004):
            selected=[(r,next(w for w in r['warnings'] if w['channel']==channel and w['diagnostic_threshold_m']==threshold)) for r in records]
            summary.append(dict(channel=channel,diagnostic_threshold_m=threshold,
                original_invalid_episodes=sum(not r['saved_valid'] for r,w in selected),
                invalid_warned_before_terminal=sum(not r['saved_valid'] and w['warned_before_terminating_control'] for r,w in selected),
                original_valid_episodes=sum(r['saved_valid'] for r,w in selected),
                valid_episodes_with_projected_warning=sum(r['saved_valid'] and bool(w['projected_indices']) for r,w in selected),
                valid_episodes_with_current_warning=sum(r['saved_valid'] and bool(w['current_indices']) for r,w in selected)))
    report=dict(rows=records,summary=summary,diagnostic_thresholds_fixed_not_tuned=True,
        future_actual_sensor_used_only_as_offline_error_target=True,no_dynamic_integration=True,
        no_policy_training=True,no_acceptance_relabeling=True,
        endpoint_warning_not_substep_collision_label=True,not_risk_control_success=True,
        reserved_qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,
        hashes={str(p):digest(p) for p in [Path(__file__),SOURCE/'results.json']})
    (OUTPUT/'results.json').write_text(json.dumps(report,indent=2))
    print('R138_TERMINAL',json.dumps(summary),flush=True)


if __name__=='__main__':main()
