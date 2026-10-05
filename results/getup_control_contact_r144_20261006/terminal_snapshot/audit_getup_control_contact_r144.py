"""Read-only contact-manifold audit of causal control-frame predictions.

No integration, training, policy changes, threshold search or relabeling.
Future first-crossing states are offline evaluation targets only.
"""
import argparse
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import audit_getup_sensor_geometry_r137 as geometry
from diagnostics import probe_getup_causal_acceleration_r140 as acceleration
from diagnostics.getup_independent_native import digest

ROOT=geometry.ROOT
SOURCE=ROOT/'outputs/getup_passive_risk_r141b_left_20261006'
OUTPUT=ROOT/'outputs/getup_control_contact_r144_20261006'
# Predeclared windows enclose the already-observed early and late events.
# They are diagnostic sampling windows, not controller conditions.
CONTROLS=tuple(range(38,45))+tuple(range(298,306))


def contacts(fk,q):
    mujoco.mj_resetData(fk.model,fk.data)
    fk.data.qpos[:]=q
    mujoco.mj_forward(fk.model,fk.data)
    rows=[dict(pair=sorted(map(int,c.geom)),distance_m=float(c.dist))
          for c in fk.data.contact if fk.floor not in c.geom]
    maxima=[max([0.,*[-r['distance_m'] for r in rows if tuple(r['pair'])==p]]) for p in geometry.PAIRS]
    maxima.append(max([0.,*[-r['distance_m'] for r in rows]]))
    return np.array(maxima),rows


def canonical(fk,q):
    result=fk.model.qpos0.copy()
    result[:3]=[0.,0.,10.];result[3:7]=[1.,0.,0.,0.]
    result[fk.qadr]=np.asarray(q)[fk.qadr]
    return result


def regressions(fk):
    x=np.zeros(55,np.float32);y=x.copy();y[:6]=3.;y[34:]=7.
    for horizon in (0.,.02):
        np.testing.assert_array_equal(fk.pose(x,horizon),fk.pose(y,horizon))
        a,_=contacts(fk,fk.pose(x,horizon));b,_=contacts(fk,fk.pose(x,horizon))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a-b,np.zeros(3))
    np.testing.assert_array_equal(acceleration.predicted_pose(fk,x,x),acceleration.predicted_pose(fk,y,y))
    try:fk.pose(np.zeros(56),.02)
    except ValueError:pass
    else:raise AssertionError('Metadata-augmented input accepted')
    assert np.isfinite(fk.home).all()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--output',type=Path,default=OUTPUT);args=parser.parse_args()
    assert args.output.parent.resolve()==(ROOT/'outputs').resolve()
    model=mujoco.MjModel.from_xml_path(str(geometry.SCENE));fk=geometry.SensorGeometry(model)
    regressions(fk)
    manifest=json.loads((SOURCE/'results.json').read_text())
    assert len(manifest['rows'])==16 and manifest['all_full_trace_bitwise_equal']
    selected=manifest['rows']
    if args.smoke:selected=[selected[0],selected[1],next(r for r in selected if r['label']=='standard')]
    args.output.mkdir(exist_ok=False)
    for path in (Path(__file__),Path(geometry.__file__),Path(acceleration.__file__)):
        shutil.copy2(path,args.output/path.name)
    reports=[];input_hashes={}
    for row in selected:
        name=row['label']+'_p'+str(row['program'])+'_'+str(row['case_seed'])
        folder=SOURCE/name;dest=args.output/name;dest.mkdir(exist_ok=False)
        files=[folder/n for n in ('trajectory.npz','substep_contacts.npz','causal_decisions.json','threshold_crossings.json','result.json')]
        before={str(p):digest(p) for p in files}
        with np.load(files[0],allow_pickle=False) as z:
            obs=z['observations'].copy();applied=z['applied'].copy();qpos=z['qpos'].copy()
        with np.load(files[1],allow_pickle=False) as z:substeps=z['records'].copy()
        decisions=json.loads(files[2].read_text());crossings=json.loads(files[3].read_text())
        frames=[]
        for k in CONTROLS:
            if k>=len(obs):continue
            sensor=obs[k].copy()
            np.testing.assert_array_equal(decisions[k]['actual_native50'],sensor[:50])
            predictions=[];sets=[]
            poses=[fk.pose(sensor),fk.pose(sensor,.02),acceleration.predicted_pose(fk,sensor,obs[k-1])]
            for q in poses:
                value,contact_set=contacts(fk,q);predictions.append(value.tolist());sets.append(contact_set)
            actual=substeps[substeps[:,1]==k]
            assert len(actual)==10
            # Planned/executed target is only audited here; it is not included
            # in an invented risk controller or used as a future-state input.
            frames.append(dict(control=k,causal_current_velocity_acceleration_peaks_m=predictions,
                causal_contact_sets=sets,original_substeps=actual.tolist(),
                original_substep_peak_m=float(actual[:,4].max()),
                sensor_position_offset_rad=sensor[6:20].tolist(),
                sensor_velocity_rad_s=(sensor[20:34].astype(float)/.05).tolist(),
                executed_target_rad=applied[k].tolist(),
                executed_target_minus_current_rad=(applied[k]-(fk.home+sensor[6:20].astype(float))).tolist()))
        crossing_report=None
        if crossings:
            first=crossings[0];k=first['control_index']
            pre=np.array(first['pre_step_qpos']);post=np.array(first['post_step_qpos'])
            pre_peak,pre_contacts=contacts(fk,canonical(fk,pre))
            again,_=contacts(fk,canonical(fk,pre));np.testing.assert_array_equal(pre_peak,again)
            post_peak,post_contacts=contacts(fk,canonical(fk,post))
            pre_sensor=np.zeros(55,np.float32);pre_sensor[6:20]=(pre[fk.qadr]-fk.home).astype(np.float32)
            rounded,_=contacts(fk,fk.pose(pre_sensor))
            frame=next(f for f in frames if f['control']==k)
            assert abs(pre_peak[2]-first['self_penetration_m'])<1e-8
            np.testing.assert_array_equal(post,qpos[k]) if first['substep']==9 else None
            crossing_report=dict(original=first,pre_integration_contact_peaks_m=pre_peak.tolist(),
                post_integration_contact_peaks_m=post_peak.tolist(),
                pre_contacts=pre_contacts,post_contacts=post_contacts,
                float32_contact_peak_error_m=float(np.abs(pre_peak-rounded).max()),
                decision_predictions_m=frame['causal_current_velocity_acceleration_peaks_m'],
                max_actual_joint_displacement_from_control_start_rad=float(np.abs(pre[fk.qadr]-(fk.home+obs[k,6:20].astype(float))).max()),
                repeated_geometry_exact=True)
        assert before=={str(p):digest(p) for p in files}
        input_hashes.update(before)
        result=dict(label=row['label'],program=row['program'],case_seed=row['case_seed'],
            initial_hash=row['initial_hash'],original_valid=row['valid'],original_success=row['success'],
            frames=frames,first_crossing=crossing_report,input_files_unchanged=True)
        (dest/'result.json').write_text(json.dumps(result,indent=2))
        reports.append({k:v for k,v in result.items() if k!='frames'})
        print('R144_CONTACT',name,None if crossing_report is None else dict(
            actual=crossing_report['pre_integration_contact_peaks_m'],post=crossing_report['post_integration_contact_peaks_m'],
            predictions=crossing_report['decision_predictions_m']),flush=True)
    summary=dict(smoke=args.smoke,rows=reports,controls=CONTROLS,
        regression_checks_passed=True,metric='Original mj_forward contact sets, not mj_geomDistance',
        no_dynamics_or_policy_changes=True,no_force_inference=True,no_threshold_search=True,
        no_success_or_physical_label_changes=True,future_states_offline_targets_only=True,
        reserved_qualification_never_loaded=True,not_getup_improvement=True,
        full_task_completed=False,hardware_readiness=False,
        hashes={**input_hashes,str(geometry.SCENE):digest(geometry.SCENE),str(Path(__file__)):digest(Path(__file__))})
    (args.output/'results.json').write_text(json.dumps(summary,indent=2))
    print('R144_TERMINAL',len(reports),flush=True)


if __name__=='__main__':main()
