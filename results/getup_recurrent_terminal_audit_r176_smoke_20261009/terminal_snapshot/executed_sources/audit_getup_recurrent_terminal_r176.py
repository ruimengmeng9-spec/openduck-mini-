"""Read-only scalar recurrence and terminal parity audit; no dynamic replay."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_recurrent_readout_r175 as run
from diagnostics.getup_independent_native import digest

OUTPUT=run.ROOT/'outputs/getup_recurrent_terminal_audit_r176_20261009'
SMOKE=run.ROOT/'outputs/getup_recurrent_terminal_audit_r176_smoke_20261009'
FIELDS=('observations','preparation_sensors','recurrent_extra_rad','recurrent_hidden','recurrent_phi','causal_sensor_error_change','local_hip_extra_rad','original_double_pre_slew_target_rad','adjusted_double_pre_slew_target_rad','planned_before_integration_rad','same_state_unperturbed_planned_rad','same_state_pre_slew_direct_new_rad')


def scalar(path,nominal,a,b,save):
    row=json.loads((path/'result.json').read_text());w=np.asarray(row['parameters'])
    with np.load(path/'trajectory.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in FIELDS}
    obs=data['observations'];initial=obs[0];state=np.zeros(16);expected=[];states=[];phis=[];deltas=[];frozen=[]
    for k,x in enumerate(obs):
        if k<529:
            extra,state,phi,delta=run.recurrent_feedback(w,x,nominal[k],initial,nominal[0],state,a,b)
            base=run.local.local_feedback(x,nominal[k],[9,10,11],row['base_gains'])
        else:extra=np.zeros(14);state=np.zeros(16);phi=np.zeros(50);delta=np.zeros(50);base=np.zeros(3)
        expected.append(extra);states.append(state.copy());phis.append(phi);deltas.append(delta);frozen.append(base)
    for name,value in [('recurrent_extra_rad',expected),('recurrent_hidden',states),('recurrent_phi',phis),('causal_sensor_error_change',deltas),('local_hip_extra_rad',frozen)]:np.testing.assert_array_equal(value,data[name],err_msg=str(path)+' '+name)
    np.testing.assert_array_equal(data['recurrent_extra_rad'][0],np.zeros(14));np.testing.assert_array_equal(data['recurrent_hidden'][0],np.zeros(16))
    np.testing.assert_array_equal(data['adjusted_double_pre_slew_target_rad']-data['original_double_pre_slew_target_rad'],data['same_state_pre_slew_direct_new_rad'])
    if len(obs)>529:np.testing.assert_array_equal(data['recurrent_extra_rad'][529:],np.zeros((len(obs)-529,14)))
    assert np.abs(data['recurrent_hidden']).max()<=1
    if row['case_seed'] is None:
        for name in ('recurrent_extra_rad','recurrent_hidden','recurrent_phi','causal_sensor_error_change'):np.testing.assert_array_equal(data[name],np.zeros_like(data[name]))
    if save is not None:
        dest=save/f"case_{row['case_seed']}";dest.mkdir(parents=True,exist_ok=False)
        np.savez_compressed(dest/'signals.npz',**data)
    post=data['planned_before_integration_rad']-data['same_state_unperturbed_planned_rad']
    return dict(case_seed=row['case_seed'],initial_hash=row['initial_hash'],success=row['success'],valid=row['valid'],controls=row['controls'],entry_time_s=row['entry_time_s'],strict_tail_s=row['strict_tail_s'],peaks=row['peaks'],
        scalar_exact=True,maximum_hidden=float(np.abs(states).max()),maximum_extra=float(np.abs(expected).max()),maximum_pre_slew_direct=float(np.abs(data['same_state_pre_slew_direct_new_rad']).max()),maximum_post_slew_direct=float(np.abs(post).max()),
        combined_target_cap=row['maximum_combined_14_joint_correction_rad'],physics_sha256=row['physics_sha256'],physics_unchanged=row['physics_unchanged'],no_label_or_root_input=True)


def parity():
    report=[]
    for case in run.CASES:
        c=run.OUTPUT/'development_candidate'/f'case_{case}';b=run.OUTPUT/'development_baseline'/f'case_{case}';r=run.selector.OUTPUT/'candidate'/f'case_{case}'
        rows=[json.loads((p/'result.json').read_text()) for p in (c,b,r)]
        assert rows[0]['initial_hash']==rows[1]['initial_hash']==rows[2]['initial_hash']
        # Whole-field equality is separate from feedback reconstruction. Root arrays
        # are allowed only here as recorded evidence, never control inputs.
        with np.load(c/'trajectory.npz',allow_pickle=False) as x,np.load(b/'trajectory.npz',allow_pickle=False) as y,np.load(r/'trajectory.npz',allow_pickle=False) as old:
            assert x.files==y.files
            for k in x.files:np.testing.assert_array_equal(x[k],y[k],err_msg=k)
            for k in old.files:np.testing.assert_array_equal(y[k],old[k],err_msg=k)
            np.testing.assert_array_equal(x['planned_before_integration_rad'],x['applied'])
        assert rows[0]['peaks']==rows[1]['peaks']==rows[2]['peaks']
        assert all(row['success']==rows[0]['success'] and row['valid']==rows[0]['valid'] for row in rows)
        report.append(dict(case_seed=case,initial_hash=rows[0]['initial_hash'],candidate_baseline_all_fields_equal=True,baseline_R157_all_fields_equal=True,original_labels_and_peaks_equal=True))
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();output=SMOKE if args.smoke else OUTPUT
    assert not output.exists()
    terminal=json.loads((run.OUTPUT/'results.json').read_text());closed=json.loads((run.OUTPUT/'training_closed.json').read_text())
    assert terminal['terminal_result_saved'] and len(closed['history'])==2
    paths=list(dict.fromkeys(p for generation in closed['history'] for p in generation['closed_trial_directories']))
    assert len(paths)==13
    tracked=[*list((run.OUTPUT/'executed_sources').iterdir()),*list((run.OUTPUT/'frozen').iterdir()),run.OUTPUT/'training_closed.json',run.OUTPUT/'results.json']
    for path in [Path(p) for p in paths]:tracked.extend(path.rglob('trajectory.npz'));tracked.append(path/'results.json')
    before={str(p):digest(p) for p in tracked if p.is_file()}
    output.mkdir();src=output/'executed_sources';src.mkdir()
    shutil.copy2(Path(__file__),src/Path(__file__).name);shutil.copy2(Path(run.__file__),src/Path(run.__file__).name)
    with np.load(run.OUTPUT/'frozen/fixed_recurrence.npz',allow_pickle=False) as z:a=z['input_matrix'].copy();b=z['recurrent_matrix'].copy()
    with np.load(run.OUTPUT/'frozen/nominal_sensor_trajectory.npz',allow_pickle=False) as z:nominal=z['observations'].copy()
    base=json.loads((Path(paths[0])/'results.json').read_text());baseline={r['case_seed']:r for r in base['rows']}
    nonzero=[p for p in paths if np.any(json.loads((Path(p)/'results.json').read_text())['rows'][0]['parameters'])]
    representative=max(nonzero,key=lambda p:run.local.rank(json.loads((Path(p)/'results.json').read_text())))
    jobs=[(run.OUTPUT/name,save_name) for name,save_name in [('zero_parity','zero_startup'),('nonzero_smoke','probe_startup')]] if args.smoke else [(Path(p),'best_nonzero' if p==representative else None) for p in paths]
    programs=[];trajectories=0
    for path,save_name in jobs:
        report=json.loads((path/'results.json').read_text());audited=[]
        for row in report['rows']:audited.append(scalar(path/f"case_{row['case_seed']}",nominal,a,b,output/save_name if save_name else None));trajectories+=1
        rescued=[r['case_seed'] for r in report['rows'] if r['case_seed'] is not None and r['success'] and not baseline[r['case_seed']]['success']]
        regressed=[r['case_seed'] for r in report['rows'] if r['case_seed'] is not None and not r['success'] and baseline[r['case_seed']]['success']]
        programs.append(dict(path=str(path),successes=report['successes'],physical_failures=report['physical_failures'],rescued=rescued,regressed=regressed,rows=audited))
    if not args.smoke:
        for row in json.loads((run.OUTPUT/'nonzero_smoke/results.json').read_text())['rows']:scalar(run.OUTPUT/'nonzero_smoke'/f"case_{row['case_seed']}",nominal,a,b,output/'probe_startup')
    pairing=parity()
    after={str(p):digest(p) for p in tracked if p.is_file()};assert before==after
    result=dict(read_only=True,smoke=args.smoke,scalar_programs=len(jobs),scalar_trajectories_audited=trajectories,programs=programs,representative=representative,terminal_pairing=pairing,source_hashes_unchanged=True,hashes_before=before,
        feedback_scalar_whitelist=list(FIELDS),root_arrays_only_separate_fullfield_parity=True,no_environment_or_MjData_or_forward_or_integration=True,labels_and_physics_not_reclassified=True,
        standalone_joint_slew_planner_not_recomputed=True,terminal_result_saved=True,new_dynamic_replays=0,full_task_completed=False,hardware_readiness=False)
    run.local.write_json(output/'results.json',result)
    print(json.dumps(dict(smoke=args.smoke,programs=len(jobs),scalar_trajectories=trajectories,paired_terminal=len(pairing),source_hashes_unchanged=True)),flush=True)


if __name__=='__main__':main()
