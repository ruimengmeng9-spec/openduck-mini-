"""Read-only R162 complete paired causal bilateral signals, no integration."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import probe_getup_fixed_bilateral_r162 as source
from diagnostics.getup_independent_native import digest

run=source.run
OUTPUT=run.ROOT/'outputs/getup_fixed_bilateral_audit_r163_20261006'

def main():
    assert not OUTPUT.exists();result=json.loads((source.OUTPUT/'results.json').read_text());assert not result['smoke']
    OUTPUT.mkdir();shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    local=run.prior.old.local;local.init_worker();ids=np.array([0,1,2,9,10,11]);rows=[];hashes={}
    for a,b in zip(result['candidate']['rows'],result['baseline']['rows']):
        case=a['case_seed'];assert case==b['case_seed'] and a['initial_hash']==b['initial_hash'];data=[]
        for name in ['candidate','baseline']:
            path=source.OUTPUT/name/f'case_{case}'
            for file in ['trajectory.npz','result.json']:hashes[str(path/file)]=digest(path/file)
            with np.load(path/'trajectory.npz',allow_pickle=False) as z:data.append({k:z[k].copy() for k in z.files})
        x,y=data;n=min(529,len(x['observations']));obs=x['observations'][:n];nom=local.NOMINAL[:n];p=np.array(a['parameters'])
        features=np.stack([run.bilateral_features(o,m,ids) for o,m in zip(obs,nom)])
        extra=[];activation=[];frozen=[]
        for o,m in zip(obs,nom):
            e,act=run.bilateral_feedback(p,o,m,obs[0],nom[0],ids)
            extra.append(e);activation.append(act);frozen.append(local.local_feedback(o,m,ids[3:],a['base_gains']))
        extra=np.array(extra);activation=np.array(activation);frozen=np.array(frozen)
        np.testing.assert_array_equal(extra,x['bilateral_extra_rad'][:n]);np.testing.assert_array_equal(activation,x['bilateral_activation'][:n]);np.testing.assert_array_equal(frozen,x['local_hip_extra_rad'][:n])
        common=-.18*np.tanh(p[:3]*activation[:,:3]+p[3:6]*activation[:,3:6])
        diff=-.18*np.tanh(p[6:9]*activation[:,6:9]+p[9:]*activation[:,9:])
        np.testing.assert_array_equal(np.clip(np.concatenate([common-diff,common+diff],axis=1),-.18,.18),extra)
        count=min(len(x['observations']),len(y['observations']));target_diff=np.abs(x['applied'][:count]-y['applied'][:count]).max(1)
        sensor_diff=np.abs(x['observations'][:count,:34].astype(float)-y['observations'][:count,:34].astype(float)).max(1)
        label='standard' if case is None else ('rescued' if a['success'] and not b['success'] else ('regressed' if b['success'] and not a['success'] else ('retained' if a['success'] else 'failed_both')))
        states=[]
        for hi in [min(n,50),n]:
            states.append(dict(end_control_exclusive=hi,mode_mean=features[:hi].mean(0).tolist(),activation_mean=activation[:hi].mean(0).tolist(),
                common_mean_rad=common[:hi].mean(0).tolist(),differential_mean_rad=diff[:hi].mean(0).tolist(),extra_mean_rad=extra[:hi].mean(0).tolist(),
                extra_max_abs_by_joint_rad=np.abs(extra[:hi]).max(0).tolist()))
        row=dict(case_seed=case,label=label,initial_hash=a['initial_hash'],candidate_success=a['success'],baseline_success=b['success'],candidate_valid=a['valid'],baseline_valid=b['valid'],
            controls=a['controls'],original_peaks=a['peaks'],actual_scalar_activation_extra_frozen_exact=True,
            first_target_difference_control=int(np.flatnonzero(target_diff>1e-8)[0]) if np.any(target_diff>1e-8) else None,
            first_sensor_difference_control=int(np.flatnonzero(sensor_diff>1e-8)[0]) if np.any(sensor_diff>1e-8) else None,
            maximum_combined_six_hip_correction_rad=a['maximum_combined_six_hip_correction_rad'],windows=states,
            diagnostic_threshold_not_acceptance=True,control_endpoint_not_first_invalid_physics_substep=True)
        dest=OUTPUT/f'case_{case}';dest.mkdir();np.savez_compressed(dest/'causal_bilateral_signals.npz',current_modes=features,causal_mode_change=features-features[0],
            actual_activation=activation,common_extra_rad=common,differential_extra_rad=diff,actual_bilateral_extra_rad=extra,
            same_state_frozen_right_feedback_rad=frozen,actual_target_difference_max_rad=target_diff,actual_sensor_difference=sensor_diff,
            current_gyro_up=obs[:,:6],baseline_gyro_up=y['observations'][:min(n,len(y['observations'])),:6])
        local.write_json(dest/'result.json',row);rows.append(row)
    assert all(digest(p)==h for p,h in hashes.items())
    local.write_json(OUTPUT/'results.json',dict(read_only=True,no_dynamic_integration=True,no_outcome_relabeling=True,rows=rows,
        rescued=result['rescued'],regressed=result['regressed'],candidate_successes=result['candidate']['successes'],baseline_successes=result['baseline']['successes'],
        candidate_physical_failures=result['candidate']['physical_failures'],baseline_physical_failures=result['baseline']['physical_failures'],
        scalar_complete_recovery_frames_exact=True,diagnostic_modes_not_world_momentum=True,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False))
    local.write_json(OUTPUT/'source_hashes.json',hashes)
    print('R163_READ_ONLY_EXACT',len(rows),result['rescued'],result['regressed'],flush=True)
    for row in rows:
        if row['label'] in ['rescued','regressed']:
            print('R163_PAIR',row['case_seed'],row['label'],row['candidate_valid'],row['controls'],row['first_target_difference_control'],row['first_sensor_difference_control'],flush=True)

if __name__=='__main__':main()
