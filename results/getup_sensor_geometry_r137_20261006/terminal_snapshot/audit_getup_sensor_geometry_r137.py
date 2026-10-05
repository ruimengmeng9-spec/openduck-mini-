"""Offline causal sensor FK audit. No dynamics, force inference or relabeling.

Future joint truth is used only for offline comparison, never risk input.
All geometric evaluation lives in independent MjData, canonical root.
"""
import argparse
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import audit_getup_self_geometry_r125 as previous
from diagnostics.getup_independent_native import digest

ROOT=previous.ROOT
SCENE=previous.SCENE
OUTPUT=ROOT/'outputs/getup_sensor_geometry_r137_20261006'
R136=ROOT/'outputs/getup_common_prefix_r136_left_20261006'
R133=ROOT/'outputs/getup_complementary_feedback_r133_left_20261006'
R122=ROOT/'outputs/getup_history_program_r122_left_20261005'
PAIRS=((77,786),(20,607))
HORIZONS=(0.,.002,.01,.02)


class SensorGeometry:
    def __init__(self,model):
        self.model=model;self.data=mujoco.MjData(model)
        self.joints=model.actuator_trnid[:,0].copy()
        self.qadr=model.jnt_qposadr[self.joints].copy()
        self.home=model.keyframe('home').qpos[self.qadr].copy()
        assert model.nu==14 and len(set(self.qadr))==14
        assert set(self.qadr)==set(range(7,21))
        self.floor=model.geom('floor').id

    def pose(self,sensor,horizon=0.):
        sensor=np.asarray(sensor)
        if sensor.shape not in ((50,),(55,)) or not np.isfinite(sensor).all():
            raise ValueError('Only finite native50 or sensor55 permitted')
        if horizon not in HORIZONS:raise ValueError('Declared causal horizons only')
        # float32 encoder offsets and velocity scale are exactly the native API.
        q=self.model.qpos0.copy();q[:3]=[0.,0.,10.];q[3:7]=[1.,0.,0.,0.]
        q[self.qadr]=self.home+sensor[6:20].astype(float)+horizon*sensor[20:34].astype(float)/.05
        return q

    def from_pose(self,q):
        mujoco.mj_resetData(self.model,self.data);self.data.qpos[:]=q
        mujoco.mj_forward(self.model,self.data)
        distances=[float(mujoco.mj_geomDistance(self.model,self.data,a,b,.1,None)) for a,b in PAIRS]
        self_max=max([0.,*[-float(c.dist) for c in self.data.contact if self.floor not in c.geom]])
        return np.array([*distances,self_max])

    def risk(self,sensor,horizon=0.):return self.from_pose(self.pose(sensor,horizon))


def selected_jobs(smoke):
    result=json.loads((R136/'results.json').read_text());jobs={}
    invalid=[]
    for program,group in enumerate(result['reports']['10']):
        for row in group['rows']:
            if not row['valid']:invalid.append((program,row))
    assert len(invalid)==5
    if smoke:invalid=invalid[:1]
    for program,row in invalid:
        case=row['case_seed'];assert case<3200000
        for label,folder in [('delayed',R136/'delay_10'/f'program_{program:02d}'),
                             ('original',R133/f'program_{program:02d}'),('snapshot',R122/'snapshot')]:
            path=folder/f'case_{case}'
            other=json.loads((path/'result.json').read_text())
            assert other['initial_hash']==row['initial_hash']
            jobs[str(path)]=(label,program,case,path)
    for label,folder in [('standard',R136/'delay_10/program_00'),('successful_control',R133/'program_02')]:
        case=None if label=='standard' else 769000
        path=folder/f'case_{case}';jobs[str(path)]=(label,0 if case is None else 2,case,path)
    return list(jobs.values())


def audit_job(job,output,smoke):
    label,program,case,path=job
    model=mujoco.MjModel.from_xml_path(str(SCENE));fk=SensorGeometry(model)
    row=json.loads((path/'result.json').read_text());source=path/'trajectory.npz'
    with np.load(source,allow_pickle=False) as z:
        obs=z['observations'].copy();qpos=z['qpos'].copy();qvel=z['qvel'].copy()
    assert obs.shape[1]==55 and obs.dtype==np.float32
    # Every frame is the real observation BEFORE its corresponding control.
    # Saved qpos is AFTER control, so precontrol ground truth is previous row.
    count=min(len(obs),60 if smoke else 400)
    sensors=obs[:count].copy();features=[];position_errors=[];velocity_errors=[];geometry_errors=[]
    repeat_errors=[];root_errors=[]
    for k,sensor in enumerate(sensors):
        risks=np.stack([fk.risk(sensor,h) for h in HORIZONS]);features.append(risks)
        if k==0 or k==count-1 or k%50==0:
            repeat_errors.append(float(np.abs(risks[0]-fk.risk(sensor)).max()))
            moved=fk.pose(sensor);moved[:3]=[1.,-2.,8.];moved[3:7]=[.5,.5,.5,.5]
            root_errors.append(float(np.abs(risks[0]-fk.from_pose(moved)).max()))
        if k:
            previous_q=qpos[k-1];previous_v=qvel[k-1]
            np.testing.assert_array_equal(sensor[6:20],(previous_q[fk.qadr]-fk.home).astype(np.float32))
            vadr=model.jnt_dofadr[fk.joints]
            np.testing.assert_array_equal(sensor[20:34],(previous_v[vadr]*.05).astype(np.float32))
            position_errors.append(float(np.abs(fk.pose(sensor)[fk.qadr]-previous_q[fk.qadr]).max()))
            velocity_errors.append(float(np.abs(sensor[20:34].astype(float)/.05-previous_v[vadr]).max()))
            true=previous_q.copy();true[:3]=[0.,0.,10.];true[3:7]=[1.,0.,0.,0.]
            geometry_errors.append(float(np.abs(risks[0]-fk.from_pose(true)).max()))
    features=np.array(features)
    dest=output/(label+'_p'+str(program)+'_'+str(case));dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'causal_geometry.npz',native_sensors=sensors,geometry_features=features,horizons_s=HORIZONS)
    warning=[]
    final_control=row['controls']-1
    for hi,h in enumerate(HORIZONS):
        for threshold in (.002,.004):
            ids=np.flatnonzero(features[:,hi,2]>=threshold)
            warning.append(dict(horizon_s=h,diagnostic_threshold_m=threshold,
                first_warning_control=None if not len(ids) else int(ids[0]),
                warning_controls=ids.tolist(),
                lead_to_terminating_control_s=None if row['valid'] or not len(ids) else (final_control-int(ids[0]))*.02))
    report=dict(label=label,program=program,case_seed=case,saved_valid=row['valid'],saved_success=row['success'],
        saved_initial_hash=row['initial_hash'],saved_substep_peaks=row['peaks'],source_hash=digest(source),
        total_saved_controls=len(obs),analyzed_precontrol_frames=count,terminal_control=final_control,
        max_sensor_q_error_rad=max(position_errors,default=0.),max_sensor_velocity_error_rad_s=max(velocity_errors,default=0.),
        max_float32_geometry_error_m=max(geometry_errors,default=0.),repeat_error_m=max(repeat_errors),
        root_transform_error_m=max(root_errors),warnings=warning,
        risk_input_only_actual_current_sensor=True,truth_only_offline_comparison=True,
        diagnostic_thresholds_not_acceptance=True,no_success_relabeling=True)
    assert max(repeat_errors)==0. and max(root_errors)<1e-8
    (dest/'result.json').write_text(json.dumps(report,indent=2));return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve()
    args.output.mkdir(exist_ok=False)
    for file in [Path(__file__),Path(__file__).with_name('test_getup_sensor_geometry_r137.py')]:shutil.copy2(file,args.output/file.name)
    model=mujoco.MjModel.from_xml_path(str(SCENE));descriptions=[previous.describe(model,p) for p in PAIRS]
    rows=[]
    for job in selected_jobs(args.smoke):
        row=audit_job(job,args.output,args.smoke);rows.append(row)
        print('R137_GEOMETRY',row['label'],row['case_seed'],row['saved_valid'],row['max_float32_geometry_error_m'],flush=True)
    report=dict(smoke=args.smoke,rows=rows,known_pairs=descriptions,horizons_s=HORIZONS,
        no_dynamic_integration=True,no_contact_force_inference=True,no_policy_or_target_changes=True,
        full_recovery_success_not_relabelled=True,not_collision_avoidance_proof=True,
        reserved_qualification_never_loaded=True,hardware_readiness=False,full_task_completed=False,
        hashes={str(p):digest(p) for p in [SCENE,R136/'results.json',R133/'results.json',Path(__file__)]})
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('R137_TERMINAL',len(rows),'READONLY',flush=True)


if __name__=='__main__':main()
