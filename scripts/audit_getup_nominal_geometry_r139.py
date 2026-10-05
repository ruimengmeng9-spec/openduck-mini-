"""Read-only same-phase nominal subtraction, no adaptive threshold search."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics.audit_getup_sensor_geometry_r137 import ROOT,OUTPUT as SOURCE
from diagnostics.getup_independent_native import digest

OUTPUT=ROOT/'outputs/getup_nominal_geometry_r139_20261006'


def main():
    OUTPUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    manifest=json.loads((SOURCE/'results.json').read_text())
    nominal_path=SOURCE/'standard_p0_None/causal_geometry.npz'
    with np.load(nominal_path,allow_pickle=False) as z:nominal=z['geometry_features'].copy()
    records=[]
    for row in manifest['rows']:
        folder=SOURCE/(row['label']+'_p'+str(row['program'])+'_'+str(row['case_seed']))
        with np.load(folder/'causal_geometry.npz',allow_pickle=False) as z:actual=z['geometry_features'].copy()
        delta=nominal[:len(actual)]-actual
        delta[:,:,2]*=-1  # All-self is penetration, not signed distance.
        if row['case_seed'] is None:np.testing.assert_array_equal(delta,np.zeros_like(delta))
        warnings=[]
        for channel,name in enumerate(('trunk_right_knee_pair','trunk_head_pair','all_self_contacts')):
            for hi,horizon in ((0,0.),(3,.02)):
                for threshold in (.002,.004):
                    ids=np.flatnonzero(delta[:,hi,channel]>=threshold)
                    warnings.append(dict(channel=name,horizon_s=horizon,diagnostic_delta_threshold_m=threshold,
                        first_warning_control=None if not len(ids) else int(ids[0]),warning_controls=ids.tolist(),
                        warned_before_terminal=bool(not row['saved_valid'] and np.any(ids<row['terminal_control']))))
        dest=OUTPUT/folder.name;dest.mkdir(exist_ok=False)
        np.savez_compressed(dest/'nominal_relative_geometry.npz',same_phase_relative_risk_m=delta)
        records.append(dict(label=row['label'],program=row['program'],case_seed=row['case_seed'],
            saved_valid=row['saved_valid'],saved_success=row['saved_success'],warnings=warnings,
            hash=digest(folder/'causal_geometry.npz')))
    summary=[]
    for channel in ('trunk_right_knee_pair','trunk_head_pair','all_self_contacts'):
        for horizon in (0.,.02):
            for threshold in (.002,.004):
                selected=[(r,next(w for w in r['warnings'] if w['channel']==channel and w['horizon_s']==horizon and w['diagnostic_delta_threshold_m']==threshold)) for r in records]
                summary.append(dict(channel=channel,horizon_s=horizon,diagnostic_delta_threshold_m=threshold,
                    invalid_total=5,valid_total=11,
                    invalid_warned_before_terminal=sum(not r['saved_valid'] and w['warned_before_terminal'] for r,w in selected),
                    valid_with_warning=sum(r['saved_valid'] and bool(w['warning_controls']) for r,w in selected)))
    report=dict(rows=records,summary=summary,standard_same_phase_delta_bitwise_zero=True,
        no_new_threshold_selection_or_fitting=True,no_dynamic_integration=True,no_policy_changes=True,
        known_saved_standard_sensor_reference_not_root_truth=True,no_case_input_in_geometry=True,
        no_acceptance_relabeling=True,no_risk_control_success_claim=True,
        independent_qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,
        hashes={str(p):digest(p) for p in [Path(__file__),SOURCE/'results.json',nominal_path]})
    (OUTPUT/'results.json').write_text(json.dumps(report,indent=2))
    print('R139_TERMINAL',json.dumps(summary),flush=True)


if __name__=='__main__':main()
