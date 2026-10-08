"""Read-only R170 recurrence given recorded anti-windup masks; no dynamics."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_tracking_memory_r170 as run
from diagnostics.getup_independent_native import digest

OUTPUT=run.ROOT/'outputs/getup_tracking_memory_terminal_audit_r171_20261008'
SMOKE=run.ROOT/'outputs/getup_tracking_memory_terminal_audit_r171_smoke_20261008'
FIELDS=['observations','applied','local_hip_extra_rad','tracking_memory_extra_rad','tracking_memory_state','tracking_error_change_rad','tracking_memory_drive','antiwindup_rejected','tracking_memory_proposal','same_state_pre_slew_direct_new_rad']
def read(p):return json.loads(Path(p).read_text())
def label(a,b):return 'rescued' if a['success'] and not b['success'] else 'regressed' if b['success'] and not a['success'] else 'retained' if a['success'] else 'both_failed'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    assert args.output.parent==run.ROOT/'outputs' and not args.output.exists()
    source=run.OUTPUT;terminal=read(source/'results.json');closed=read(source/'training_closed.json');contract=read(source/'contract.json')
    assert terminal['terminal_result_saved'] and not terminal['smoke'] and len(closed['history'])==3
    assert not np.any(terminal['parameters']) and not terminal['expanded_development_run'] and not terminal['independent_qualification_run']
    paths=sorted({s for h in closed['history'] for s in h['closed_trial_directories']});reports={s:read(Path(s)/'results.json') for s in paths}
    zero=[s for s in paths if not np.any(reports[s]['rows'][0]['parameters'])];assert len(zero)==1
    base={r['case_seed']:r for r in reports[zero[0]]['rows']};nonzero=[s for s in paths if s!=zero[0]]
    best_any=max(nonzero,key=lambda s:(reports[s]['successes'],-reports[s]['physical_failures'],reports[s]['return_sum']))
    valid=[s for s in nonzero if not reports[s]['physical_failures']]
    best_valid=max(valid,key=lambda s:(reports[s]['successes'],reports[s]['return_sum'])) if valid else None
    source_hashes={}
    def remember(f):
        f=Path(f);source_hashes[str(f)]=digest(f);return f
    for f in [source/'results.json',source/'training_closed.json',source/'contract.json',Path(__file__),source/'executed_sources'/Path(run.__file__).name]:remember(f)
    assert digest(run.__file__)==digest(source/'executed_sources'/Path(run.__file__).name)
    assert all(digest(f)==h for f,h in contract['hashes'].items())
    with np.load(remember(source/'frozen/nominal_sensor_trajectory.npz'),allow_pickle=False) as z:nominal=z['observations'].copy()
    args.output.mkdir();shutil.copy2(__file__,args.output/Path(__file__).name)
    programs=[];selected=[zero[0],best_valid or best_any] if args.smoke else paths
    for s in selected:
        report=reports[s];remember(Path(s)/'results.json');rows=[]
        for row in report['rows']:
            case=row['case_seed']
            if args.smoke and case not in [None,769002,773004]:continue
            b=base[case];assert row['initial_hash']==b['initial_hash']
            f=remember(Path(s)/f'case_{case}/trajectory.npz');bf=remember(Path(zero[0])/f'case_{case}/trajectory.npz');remember(f.with_name('result.json'))
            with np.load(f,allow_pickle=False) as z:a={k:z[k].copy() for k in FIELDS}
            with np.load(bf,allow_pickle=False) as z:baseline={k:z[k].copy() for k in ['observations','applied']}
            groups=np.asarray(row['groups']);state=np.zeros(14);initial=a['observations'][0];expected={k:[] for k in ['local_hip_extra_rad','tracking_memory_extra_rad','tracking_memory_state','tracking_error_change_rad','tracking_memory_drive','tracking_memory_proposal']}
            for k,obs in enumerate(a['observations']):
                if k<529:
                    _,proposal,delta,drive,leaked=run.tracking_memory(row['parameters'],obs,nominal[k],initial,nominal[0],state,groups)
                    mask=a['antiwindup_rejected'][k];assert not np.any(mask[groups==2])
                    state=proposal.copy();state[mask]=leaked[mask]
                    extra=run.memory_output(np.asarray(row['parameters']),state,groups)
                    frozen=run.local.local_feedback(obs,nominal[k],np.array([9,10,11]),np.asarray(row['base_gains']))
                else:
                    state=np.zeros(14);proposal=np.zeros(14);delta=np.zeros(14);drive=np.zeros(14);extra=np.zeros(14);frozen=np.zeros(3)
                    assert not np.any(a['antiwindup_rejected'][k])
                for key,value in zip(expected,[frozen,extra,state,delta,drive,proposal]):expected[key].append(value.copy())
            for field,value in expected.items():np.testing.assert_array_equal(value,a[field],err_msg=field)
            assert np.abs(a['tracking_memory_state']).max()<=1.
            np.testing.assert_array_equal(a['tracking_memory_extra_rad'][:,groups==2],np.zeros((len(a['observations']),4)))
            for name in ['tracking_memory_extra_rad','tracking_memory_state','tracking_error_change_rad','tracking_memory_drive','tracking_memory_proposal','same_state_pre_slew_direct_new_rad']:
                np.testing.assert_array_equal(a[name][0],np.zeros(14))
            np.testing.assert_array_equal(a['applied'][0],baseline['applied'][0]);assert not np.any(a['antiwindup_rejected'][0])
            if case is None:
                for name in FIELDS[2:]:np.testing.assert_array_equal(a[name],np.zeros_like(a[name]))
            if not np.any(row['parameters']):
                assert not np.any(a['tracking_memory_extra_rad']) and not np.any(a['same_state_pre_slew_direct_new_rad'])
            n=min(len(a['observations']),len(baseline['observations']));target_diff=a['applied'][:n]-baseline['applied'][:n];sensor_diff=a['observations'][:n]-baseline['observations'][:n]
            first=lambda v:next((int(i) for i in np.flatnonzero(np.max(np.abs(v),axis=1)>1e-8)),None)
            r=dict(case_seed=case,initial_hash=row['initial_hash'],label=label(row,b),success=row['success'],valid=row['valid'],controls=row['controls'],peaks=row['peaks'],
                first_target_difference_control=first(target_diff),first_sensor_difference_control=first(sensor_diff),
                max_memory=float(np.abs(a['tracking_memory_state']).max()),max_extra_rad=float(np.abs(a['tracking_memory_extra_rad']).max()),
                rejected_channel_steps=int(np.count_nonzero(a['antiwindup_rejected'])),early_rejected_channel_steps=int(np.count_nonzero(a['antiwindup_rejected'][:50])),
                early_same_state_pre_slew_direct_difference_rad=float(np.abs(a['same_state_pre_slew_direct_new_rad'][:50]).max()),
                early_executed_target_difference_rad=float(np.abs(target_diff[:50]).max()),early_gyro_up_difference=float(np.abs(sensor_diff[:50,:6]).max()),
                maximum_combined_14_joint_correction_rad=row['maximum_combined_14_joint_correction_rad'],
                drive_proposal_accepted_memory_extra_exact_given_recorded_mask=True,original_right_hip_feedback_exact=True,
                antiwindup_mask_planning_decision_not_independently_recomputed=True)
            rows.append(r)
            if s in {best_any,best_valid}:
                dest=args.output/'representatives'/Path(s).relative_to(source/'training')/f'case_{case}';dest.mkdir(parents=True)
                np.savez_compressed(dest/'paired_signals.npz',**a,executed_target_difference_rad=target_diff,sensor_difference=sensor_diff,baseline_observations=baseline['observations'][:n],baseline_applied=baseline['applied'][:n])
        programs.append(dict(directory=s,parameters=report['rows'][0]['parameters'],successes=report['successes'],physical_failures=report['physical_failures'],
            rescued=[r['case_seed'] for r in rows if r['label']=='rescued'],regressed=[r['case_seed'] for r in rows if r['label']=='regressed'],rows=rows))
    parity=[]
    for a,b in zip(terminal['candidate']['rows'],terminal['baseline']['rows']):
        case=a['case_seed'];assert case==b['case_seed'] and a['initial_hash']==b['initial_hash'] and a['peaks']==b['peaks']
        files=[source/'development_candidate'/f'case_{case}/trajectory.npz',source/'development_baseline'/f'case_{case}/trajectory.npz',run.selector.OUTPUT/'candidate'/f'case_{case}/trajectory.npz']
        priorrow=read(remember(files[2].with_name('result.json')));assert b['initial_hash']==priorrow['initial_hash'] and b['peaks']==priorrow['peaks']
        for f in files:remember(f)
        with np.load(files[0],allow_pickle=False) as x,np.load(files[1],allow_pickle=False) as y,np.load(files[2],allow_pickle=False) as z:
            equal={k:bool(np.array_equal(x[k],y[k])) for k in x.files};prior={k:bool(np.array_equal(y[k],z[k])) for k in z.files}
        assert all(equal.values()) and all(prior.values());parity.append(dict(case_seed=case,all_candidate_baseline_fields_exact=equal,baseline_R157_fields_exact=prior))
    if not args.smoke:
        smoke=read(SMOKE/'results.json');assert smoke['smoke']
        for sp in smoke['programs']:
            fp=next(p for p in programs if p['directory']==sp['directory']);assert sp['rows']==[r for r in fp['rows'] if r['case_seed'] in [None,769002,773004]]
        for f in (SMOKE/'representatives').rglob('paired_signals.npz'):
            with np.load(f,allow_pickle=False) as a,np.load(args.output/f.relative_to(SMOKE),allow_pickle=False) as b:
                assert a.files==b.files
                for k in a.files:np.testing.assert_array_equal(a[k],b[k])
    assert all(digest(f)==h for f,h in source_hashes.items()),'Source changed during read-only audit'
    summary=dict(read_only=True,no_MjData_environment_forward_or_integration=True,no_outcome_or_validity_relabeling=True,
        smoke=args.smoke,distinct_actual_programs=len(paths),nonzero_programs=len(nonzero),actual_training_attempts=25*len(paths),scalar_trajectories_audited=sum(len(p['rows']) for p in programs),generations=3,
        selected_parameters=terminal['parameters'],candidate_successes=terminal['candidate']['successes'],baseline_successes=terminal['baseline']['successes'],candidate_physical_failures=terminal['candidate']['physical_failures'],development_gate=terminal['original_development_gate'],
        best_nonzero=best_any,best_all_valid_nonzero=best_valid,all_valid_nonzero_programs=len(valid),programs=programs,terminal_parity=parity,
        recurrence_recomputed_given_recorded_mask_not_independent_planning_decision_audit=True,
        planned_direct_difference_read_from_saved_actual_field_not_reconstructed_double_IMU=True,
        separate_same_state_pre_slew_and_executed_differences_not_dynamic_counterfactual=True,
        qpos_qvel_only_read_for_separate_original_field_parity_not_feedback_inputs=True,source_hashes_unchanged=True,independent_smoke_bitwise_equal=not args.smoke,
        qualification_never_loaded=True,terminal_result_saved=True,full_task_completed=False,hardware_readiness=False)
    run.local.write_json(args.output/'results.json',summary);run.local.write_json(args.output/'source_hashes.json',source_hashes)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['programs','terminal_parity','selected_parameters']}),flush=True)
    for p in programs:print('R171_PROGRAM',Path(p['directory']).relative_to(source),p['successes'],p['physical_failures'],p['rescued'],p['regressed'],flush=True)

if __name__=='__main__':main()
