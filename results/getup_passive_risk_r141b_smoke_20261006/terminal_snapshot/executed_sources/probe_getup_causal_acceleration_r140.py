"""Offline two-frame sensor extrapolation, never a controller or dynamic test."""
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics.audit_getup_sensor_geometry_r137 import ROOT,SCENE,SensorGeometry,OUTPUT as SOURCE
from diagnostics.getup_independent_native import digest

OUTPUT=ROOT/'outputs/getup_causal_acceleration_r140_20261006'


def predicted_pose(fk,current,past):
    current=np.asarray(current);past=np.asarray(past)
    if current.shape!=(55,) or past.shape!=(55,) or not np.isfinite(current).all() or not np.isfinite(past).all():
        raise ValueError('Current and previous causal sensor55 only')
    q=fk.pose(current,.02)
    acceleration=(current[20:34].astype(float)-past[20:34].astype(float))/.05/.02
    q[fk.qadr]+=.5*.02**2*acceleration
    return q


def main():
    OUTPUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    manifest=json.loads((SOURCE/'results.json').read_text());fk=SensorGeometry(mujoco.MjModel.from_xml_path(str(SCENE)))
    # Changing unrelated inputs cannot affect joint-only FK; identical nominal
    # inputs produce a scalar exact-zero relative feature, without dynamics.
    x=np.zeros(55,dtype=np.float32);y=x.copy();y[:6]=3.;y[34:]=7.
    np.testing.assert_array_equal(predicted_pose(fk,x,x),predicted_pose(fk,y,y))
    np.testing.assert_array_equal(fk.from_pose(predicted_pose(fk,x,x))-fk.from_pose(predicted_pose(fk,x,x)),np.zeros(3))
    rows=[]
    for row in manifest['rows']:
        label=row['label'];program=row['program'];case=row['case_seed']
        source=SOURCE/(label+'_p'+str(program)+'_'+str(case))/'causal_geometry.npz'
        with np.load(source,allow_pickle=False) as z:sensors=z['native_sensors'].copy();g=z['geometry_features'].copy()
        predicted=np.stack([fk.from_pose(predicted_pose(fk,sensors[k],sensors[k-1])) for k in range(1,len(sensors))])
        acceleration_error=predicted[:-1]-g[2:,0];velocity_error=g[1:-1,3]-g[2:,0]
        qerr_velocity=[];qerr_accel=[]
        for k in range(1,len(sensors)-1):
            truth=fk.pose(sensors[k+1])[fk.qadr]  # OFFLINE target only.
            qerr_velocity.append(fk.pose(sensors[k],.02)[fk.qadr]-truth)
            qerr_accel.append(predicted_pose(fk,sensors[k],sensors[k-1])[fk.qadr]-truth)
        warnings=[]
        for channel,name in enumerate(('trunk_right_knee_pair','trunk_head_pair','all_self_contacts')):
            for threshold in (.002,.004):
                mask=(-predicted[:,channel] if channel<2 else predicted[:,channel])>=threshold
                ids=np.flatnonzero(mask)+1
                warnings.append(dict(channel=name,diagnostic_threshold_m=threshold,warning_controls=ids.tolist(),
                    warned_before_terminating_control=bool(not row['saved_valid'] and np.any(ids<row['terminal_control']))))
        dest=OUTPUT/source.parent.name;dest.mkdir(exist_ok=False)
        np.savez_compressed(dest/'two_frame_prediction.npz',causal_acceleration_geometry=predicted,
            acceleration_next_geometry_error=acceleration_error,velocity_next_geometry_error=velocity_error)
        rows.append(dict(label=label,program=program,case_seed=case,saved_valid=row['saved_valid'],saved_success=row['saved_success'],warnings=warnings,
            velocity_joint_prediction_rmse_rad=float(np.sqrt(np.mean(np.array(qerr_velocity)**2))),
            acceleration_joint_prediction_rmse_rad=float(np.sqrt(np.mean(np.array(qerr_accel)**2))),
            velocity_geometry_mae_m=np.mean(np.abs(velocity_error),axis=0).tolist(),
            acceleration_geometry_mae_m=np.mean(np.abs(acceleration_error),axis=0).tolist(),source_hash=digest(source)))
    summary=[]
    for channel in ('trunk_right_knee_pair','trunk_head_pair','all_self_contacts'):
        for threshold in (.002,.004):
            selected=[(r,next(w for w in r['warnings'] if w['channel']==channel and w['diagnostic_threshold_m']==threshold)) for r in rows]
            summary.append(dict(channel=channel,diagnostic_threshold_m=threshold,invalid_total=5,valid_total=11,
                invalid_warned_before_terminal=sum(not r['saved_valid'] and w['warned_before_terminating_control'] for r,w in selected),
                valid_with_warning=sum(r['saved_valid'] and bool(w['warning_controls']) for r,w in selected)))
    report=dict(rows=rows,summary=summary,causal_two_frame_input_regression_passed=True,
        joint_prediction_rmse_improved_rows=sum(r['acceleration_joint_prediction_rmse_rad']<r['velocity_joint_prediction_rmse_rad'] for r in rows),
        no_fitting_or_threshold_search=True,no_dynamic_integration=True,no_policy_changes=True,
        future_sensor_only_offline_evaluation=True,no_root_truth_risk_input=True,no_label_changes=True,
        no_collision_avoidance_or_getup_improvement_claim=True,qualification_never_loaded=True,
        full_task_completed=False,hardware_readiness=False,
        hashes={str(p):digest(p) for p in [Path(__file__),SCENE,SOURCE/'results.json']})
    (OUTPUT/'results.json').write_text(json.dumps(report,indent=2))
    print('R140_TERMINAL',json.dumps(dict(rmse_improved_rows=report['joint_prediction_rmse_improved_rows'],rows=len(rows),summary=summary)),flush=True)


if __name__=='__main__':main()
