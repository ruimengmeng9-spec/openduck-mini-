"""Read-only canonical kinematic scalar and paired terminal audit. No replay."""
import argparse
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
from diagnostics import train_getup_foot_task_r177 as run

OUTPUT=run.ROOT/'outputs/getup_foot_task_terminal_audit_r178_20261009'
SMOKE=run.ROOT/'outputs/getup_foot_task_terminal_audit_r178_smoke_20261009'
FIELDS=('observations','preparation_sensors','foot_task_extra_rad','causal_foot_task_error_m','foot_task_jacobian_m_per_rad','foot_task_dls_direction_rad','local_hip_extra_rad','original_double_pre_slew_target_rad','adjusted_double_pre_slew_target_rad','planned_before_integration_rad','same_state_unperturbed_planned_rad','same_state_pre_slew_direct_new_rad')


def scalar(path,nominal,fk,save):
    row=json.loads((path/'result.json').read_text());c=np.asarray(row['parameters'])
    with np.load(path/'trajectory.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in FIELDS}
    obs=data['observations'];initial=obs[0];extra_rows=[];errors=[];jacobians=[];directions=[];frozen=[]
    for k,x in enumerate(obs):
        if k<529:
            extra,error,jac,dq=run.foot_feedback(c,x,nominal[k],initial,nominal[0],fk)
            base=run.local.local_feedback(x,nominal[k],[9,10,11],row['base_gains'])
        else:extra=np.zeros(14);error=np.zeros((2,3));jac=np.zeros((2,3,5));dq=np.zeros((2,5));base=np.zeros(3)
        extra_rows.append(extra);errors.append(error);jacobians.append(jac);directions.append(dq);frozen.append(base)
    for name,value in [('foot_task_extra_rad',extra_rows),('causal_foot_task_error_m',errors),('foot_task_jacobian_m_per_rad',jacobians),('foot_task_dls_direction_rad',directions),('local_hip_extra_rad',frozen)]:np.testing.assert_array_equal(value,data[name],err_msg=str(path)+' '+name)
    for name in ('foot_task_extra_rad','causal_foot_task_error_m','foot_task_dls_direction_rad'):np.testing.assert_array_equal(data[name][0],np.zeros_like(data[name][0]))
    np.testing.assert_array_equal(data['foot_task_extra_rad'][:,5:9],np.zeros((len(obs),4)))
    if len(obs)>529:
        for name in ('foot_task_extra_rad','causal_foot_task_error_m','foot_task_jacobian_m_per_rad','foot_task_dls_direction_rad'):np.testing.assert_array_equal(data[name][529:],np.zeros_like(data[name][529:]))
    if row['case_seed'] is None:
        for name in ('foot_task_extra_rad','causal_foot_task_error_m','foot_task_dls_direction_rad'):np.testing.assert_array_equal(data[name],np.zeros_like(data[name]))
    np.testing.assert_array_equal(data['adjusted_double_pre_slew_target_rad']-data['original_double_pre_slew_target_rad'],data['same_state_pre_slew_direct_new_rad'])
    if save is not None:
        dest=save/f"case_{row['case_seed']}";dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'signals.npz',**data)
    return dict(case_seed=row['case_seed'],initial_hash=row['initial_hash'],success=row['success'],valid=row['valid'],controls=row['controls'],entry_time_s=row['entry_time_s'],strict_tail_s=row['strict_tail_s'],peaks=row['peaks'],
        scalar_exact=True,maximum_extra=float(np.abs(extra_rows).max()),maximum_task_error_m=float(np.abs(errors).max()),maximum_pre_slew_direct=float(np.abs(data['same_state_pre_slew_direct_new_rad']).max()),
        maximum_post_slew_direct=float(np.abs(data['planned_before_integration_rad']-data['same_state_unperturbed_planned_rad']).max()),combined_target_cap=row['maximum_combined_14_joint_correction_rad'],physics_sha256=row['physics_sha256'],physics_unchanged=row['physics_unchanged'])


def parity():
    report=[]
    for case in run.CASES:
        c=run.OUTPUT/'development_candidate'/f'case_{case}';b=run.OUTPUT/'development_baseline'/f'case_{case}';r=run.selector.OUTPUT/'candidate'/f'case_{case}'
        rows=[json.loads((p/'result.json').read_text()) for p in (c,b,r)]
        assert rows[0]['initial_hash']==rows[1]['initial_hash']==rows[2]['initial_hash']
        # Root arrays are recorded evidence in this isolated equality comparison;
        # they never enter scalar FK reconstruction or the runtime controller.
        with np.load(c/'trajectory.npz',allow_pickle=False) as x,np.load(b/'trajectory.npz',allow_pickle=False) as y,np.load(r/'trajectory.npz',allow_pickle=False) as old:
            assert x.files==y.files
            for key in x.files:np.testing.assert_array_equal(x[key],y[key],err_msg=key)
            for key in old.files:np.testing.assert_array_equal(y[key],old[key],err_msg=key)
            np.testing.assert_array_equal(x['planned_before_integration_rad'],x['applied'])
        assert rows[0]['peaks']==rows[1]['peaks']==rows[2]['peaks']
        assert all(row['success']==rows[0]['success'] and row['valid']==rows[0]['valid'] for row in rows)
        report.append(dict(case_seed=case,initial_hash=rows[0]['initial_hash'],candidate_baseline_all_fields_equal=True,baseline_R157_all_fields_equal=True,original_labels_and_peaks_equal=True))
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();output=SMOKE if args.smoke else OUTPUT
    assert not output.exists()
    terminal=json.loads((run.OUTPUT/'results.json').read_text());closed=json.loads((run.OUTPUT/'training_closed.json').read_text())
    assert terminal['terminal_result_saved'] and len(closed['history'])==2 and not any(terminal['parameters'])
    paths=list(dict.fromkeys(p for g in closed['history'] for p in g['closed_trial_directories']))
    tracked=[*list((run.OUTPUT/'executed_sources').iterdir()),*list((run.OUTPUT/'frozen').iterdir()),run.OUTPUT/'training_closed.json',run.OUTPUT/'results.json',run.local.prior.program.SCENE]
    for path in map(Path,paths):tracked.extend(path.rglob('trajectory.npz'));tracked.append(path/'results.json')
    before={str(p):run.prior.digest(p) for p in tracked if p.is_file()}
    output.mkdir();src=output/'executed_sources';src.mkdir()
    for source in (Path(__file__),Path(run.__file__)):shutil.copy2(source,src/source.name)
    with np.load(run.OUTPUT/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    model=mujoco.MjModel.from_xml_path(str(run.local.prior.program.SCENE));fk=run.FootKinematics(model)
    base=json.loads((Path(paths[0])/'results.json').read_text());baseline={r['case_seed']:r for r in base['rows']}
    nonzero=[p for p in paths if any(json.loads((Path(p)/'results.json').read_text())['rows'][0]['parameters'])]
    best=max(nonzero,key=lambda p:run.local.rank(json.loads((Path(p)/'results.json').read_text())))
    valid=[p for p in nonzero if not json.loads((Path(p)/'results.json').read_text())['physical_failures']]
    best_valid=max(valid,key=lambda p:run.local.rank(json.loads((Path(p)/'results.json').read_text()))) if valid else None
    representatives={best:'best_nonzero'}
    if best_valid and best_valid!=best:representatives[best_valid]='best_all_valid'
    jobs=[(run.OUTPUT/name,save) for name,save in [('zero_parity','zero_startup'),('nonzero_smoke','probe_startup')]] if args.smoke else [(Path(p),representatives.get(p)) for p in paths]
    programs=[];count=0
    for path,save in jobs:
        report=json.loads((path/'results.json').read_text());rows=[]
        for row in report['rows']:rows.append(scalar(path/f"case_{row['case_seed']}",nominal,fk,output/save if save else None));count+=1
        rescued=[r['case_seed'] for r in report['rows'] if r['case_seed'] is not None and r['success'] and not baseline[r['case_seed']]['success']]
        regressed=[r['case_seed'] for r in report['rows'] if r['case_seed'] is not None and not r['success'] and baseline[r['case_seed']]['success']]
        programs.append(dict(path=str(path),parameters=report['rows'][0]['parameters'],successes=report['successes'],physical_failures=report['physical_failures'],rescued=rescued,regressed=regressed,rows=rows))
    if not args.smoke:
        for row in json.loads((run.OUTPUT/'nonzero_smoke/results.json').read_text())['rows']:scalar(run.OUTPUT/'nonzero_smoke'/f"case_{row['case_seed']}",nominal,fk,output/'probe_startup')
    pairing=parity();after={str(p):run.prior.digest(p) for p in tracked if p.is_file()};assert before==after
    result=dict(read_only=True,smoke=args.smoke,scalar_programs=len(jobs),scalar_trajectories_audited=count,programs=programs,nonzero_programs=len(nonzero),all_valid_nonzero_programs=len(valid),best_nonzero=best,best_all_valid=best_valid,
        terminal_pairing=pairing,source_hashes_unchanged=True,hashes_before=before,feedback_scalar_whitelist=list(FIELDS),root_arrays_only_separate_fullfield_parity=True,
        private_canonical_MjData_FK_only=True,no_environment_or_forward_or_integration_or_contact_queries=True,labels_and_physics_not_reclassified=True,standalone_joint_slew_planner_not_recomputed=True,
        terminal_result_saved=True,new_dynamic_replays=0,full_task_completed=False,hardware_readiness=False)
    run.local.write_json(output/'results.json',result);print(json.dumps(dict(smoke=args.smoke,programs=len(jobs),scalar_trajectories=count,paired_terminal=len(pairing),source_hashes_unchanged=True)),flush=True)


if __name__=='__main__':main()
