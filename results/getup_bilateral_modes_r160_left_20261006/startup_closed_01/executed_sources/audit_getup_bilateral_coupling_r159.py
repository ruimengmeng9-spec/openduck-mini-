"""Read-only six-failure pairing; bilateral modes are actuator-coordinate diagnostics."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_set_decision_r157 as source
from diagnostics.getup_independent_native import digest

OUTPUT=source.ROOT/'outputs/getup_bilateral_coupling_r159_20261006'
JOINTS=('left_hip_yaw','left_hip_roll','left_hip_pitch','right_hip_yaw','right_hip_roll','right_hip_pitch')

def main():
    assert not OUTPUT.exists()
    terminal=json.loads((source.OUTPUT/'results.json').read_text());assert not terminal['smoke']
    source.old.local.init_worker();nominal=source.old.local.NOMINAL
    # Construct model only; no reset, prepare, mj_step or forward/integration.
    env=source.old.local.prior.audit.HistoryReferenceEpisode(str(source.old.local.prior.program.SCENE),str(source.old.local.prior.program.STAND),source.old.local.prior.program.REFERENCE,259,True)
    sim=env.sim
    sensors=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in JOINTS])
    actuators=np.array([sim.model.actuator(n).id for n in JOINTS]);np.testing.assert_array_equal(sensors,[0,1,2,9,10,11])
    OUTPUT.mkdir();shutil.copy2(__file__,OUTPUT/Path(__file__).name);hashes={};rows=[]
    def load(path):
        for n in ['trajectory.npz','result.json']:hashes[str(path/n)]=digest(path/n)
        with np.load(path/'trajectory.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
        return data,json.loads((path/'result.json').read_text())
    failures=[r for r in terminal['reports']['candidate']['rows'] if r['case_seed'] is not None and not r['success']]
    assert len(failures)==6
    def first(values,threshold):
        indexes=np.flatnonzero(np.any(np.abs(values)>threshold,axis=1))
        return int(indexes[0]) if len(indexes) else None
    for failed in failures:
        case=failed['case_seed'];baseline,_=load(source.OUTPUT/'candidate'/f'case_{case}')
        alternatives=[]
        for program in range(4):
            other,row=load(source.old.source.OUTPUT/f'program_{program:02d}'/f'case_{case}')
            assert failed['initial_hash']==row['initial_hash']
            np.testing.assert_array_equal(baseline['observations'][0],other['observations'][0])
            if not row['success'] or not row['valid']:continue
            length=min(len(baseline['observations']),len(other['observations']),529)
            obs=baseline['observations'][:length];alt=other['observations'][:length]
            qerr=(obs[:,6+sensors]-nominal[:length,6+sensors]).astype(float)/.05
            verr=(obs[:,20+sensors]-nominal[:length,20+sensors]).astype(float)/.05
            common_pos=(qerr[:,:3]+qerr[:,3:])/2;diff_pos=(qerr[:,3:]-qerr[:,:3])/2
            common_vel=(verr[:,:3]+verr[:,3:])/2;diff_vel=(verr[:,3:]-verr[:,:3])/2
            modes=np.concatenate([common_pos,common_vel,diff_pos,diff_vel],axis=1);changes=modes-modes[0]
            applied=other['applied'][:length]-baseline['applied'][:length]
            sensor=alt-obs;folder=OUTPUT/f'case_{case}_program_{program:02d}';folder.mkdir()
            np.savez_compressed(folder/'paired_sensor_modes.npz',baseline_current_native55=obs,alternative_current_native55=alt,
                bilateral_position_error=qerr,bilateral_velocity_error=verr,common_differential_modes=modes,causal_change_from_initial=changes,
                target_difference=applied,original_baseline_strict=baseline['strict'][:length],original_alternative_strict=other['strict'][:length])
            alternatives.append(dict(program=program,original_success=row['success'],original_valid=row['valid'],
                first_right_applied_difference_over_1e8_rad=first(applied[:,actuators[3:]],1e-8),
                first_left_applied_difference_over_1e8_rad=first(applied[:,actuators[:3]],1e-8),
                first_left_position_sensor_difference_over_1e8_rad=first(sensor[:,6+sensors[:3]],1e-8),
                first_up_sensor_difference_over_1e8=first(sensor[:,3:6],1e-8),
                first50_max_left_applied_difference_rad=np.abs(applied[:50,actuators[:3]]).max(0).tolist(),
                first50_max_right_applied_difference_rad=np.abs(applied[:50,actuators[3:]]).max(0).tolist(),
                first50_max_left_position_sensor_difference_rad=np.abs(sensor[:50,6+sensors[:3]]).max(0).tolist(),
                first50_max_causal_mode_change=np.abs(changes[:50]).max(0).tolist(),
                direct_extra_affects_right_hip_only=True,baseline_original_success=failed['success']))
        assert alternatives
        rows.append(dict(case_seed=case,initial_hash=failed['initial_hash'],failed_program=failed['global_program_choice'],alternatives=alternatives))
    for p,h in hashes.items():assert digest(p)==h
    result=dict(read_only=True,no_dynamic_integration=True,no_relabeling=True,rows=rows,joints=JOINTS,sensor_ids=sensors.tolist(),actuator_ids=actuators.tolist(),
        paired_alternatives=sum(len(r['alternatives']) for r in rows),thresholds_diagnostic_not_acceptance=True,
        common_differential_actuator_coordinates_not_world_momentum=True,
        hypothesis='Only three right-hip direct extra targets can change left-hip original targets through subsequent sensed dynamics. Test coupled left/right joint-error modes instead of more unilateral coefficients; no unique-cause or recovery claim.',
        independent_qualification_run=False,full_task_completed=False,hardware_readiness=False)
    source.old.local.write_json(OUTPUT/'results.json',result);source.old.local.write_json(OUTPUT/'source_hashes.json',hashes)
    print('R159_PAIRED',result['paired_alternatives'],flush=True)
    for r in rows:
        print(r['case_seed'],[(a['program'],a['first_right_applied_difference_over_1e8_rad'],a['first_left_applied_difference_over_1e8_rad'],a['first_left_position_sensor_difference_over_1e8_rad']) for a in r['alternatives']],flush=True)

if __name__=='__main__':main()
