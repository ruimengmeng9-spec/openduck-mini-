"""Read-only R165 complete-path terminal and actual velocity feedback audit."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_velocity_damping_r165 as run

OUTPUT=run.ROOT/'outputs/getup_velocity_terminal_audit_r166_20261008'
FIELDS=['observations','applied','local_hip_extra_rad','damping_extra_rad',
        'actual_velocity_error_rad_s','same_state_pre_slew_direct_new_rad',
        'same_state_post_slew_direct_new_rad']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
def label(a,b):
    return 'rescued' if a['success'] and not b['success'] else 'regressed' if b['success'] and not a['success'] else 'retained' if a['success'] else 'both_failed'

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true');args=p.parse_args()
    assert not args.output.exists() and args.output.parent==run.ROOT/'outputs'
    source=run.OUTPUT;terminal=read(source/'results.json');closed=read(source/'training_closed.json')
    assert not terminal['smoke'] and len(closed['history'])==4
    assert not terminal['expanded_development_run'] and not terminal['independent_qualification_run']
    paths=sorted({s for h in closed['history'] for s in h['closed_trial_directories']})
    reports={s:read(Path(s)/'results.json') for s in paths}
    zero=[s for s in paths if not np.any(reports[s]['rows'][0]['parameters'])];assert len(zero)==1
    base={r['case_seed']:r for r in reports[zero[0]]['rows']}
    source_hashes={}
    def remember(f):
        f=Path(f);source_hashes[str(f)]=digest(f);return f
    for f in [source/'results.json',source/'training_closed.json',source/'contract.json',Path(__file__),source/'executed_sources'/Path(run.__file__).name]:remember(f)
    assert digest(run.__file__)==digest(source/'executed_sources'/Path(run.__file__).name)
    nominal_file=remember(source/'frozen/nominal_sensor_trajectory.npz')
    with np.load(nominal_file,allow_pickle=False) as z:nominal=z['observations'].copy()
    args.output.mkdir();shutil.copy2(__file__,args.output/Path(__file__).name)
    nonzero=[s for s in paths if np.any(reports[s]['rows'][0]['parameters'])]
    best_any=max(nonzero,key=lambda s:(reports[s]['successes'],-reports[s]['physical_failures'],reports[s]['return_sum']))
    valid=[s for s in nonzero if not reports[s]['physical_failures']]
    best_valid=max(valid,key=lambda s:(reports[s]['successes'],reports[s]['return_sum'])) if valid else None
    representatives={best_any,best_valid}
    programs=[]
    for s in (paths[:2] if args.smoke else paths):
        report=reports[s];remember(Path(s)/'results.json');rows=[]
        for row in (report['rows'][:3] if args.smoke else report['rows']):
            case=row['case_seed'];b=base[case];assert row['initial_hash']==b['initial_hash']
            f=remember(Path(s)/f'case_{case}/trajectory.npz');bf=remember(Path(zero[0])/f'case_{case}/trajectory.npz')
            remember(Path(s)/f'case_{case}/result.json')
            with np.load(f,allow_pickle=False) as z:a={k:z[k].copy() for k in FIELDS}
            with np.load(bf,allow_pickle=False) as z:baseline={k:z[k].copy() for k in ['observations','applied']}
            count=len(a['observations']);expected=[];errors=[];frozen=[]
            for k,obs in enumerate(a['observations']):
                if k<529:
                    extra,error=run.damping_feedback(row['parameters'],obs,nominal[k],row['groups'])
                    old=run.local.local_feedback(obs,nominal[k],np.array([9,10,11]),row['base_gains'])
                else:extra=np.zeros(14);error=np.zeros(14);old=np.zeros(3)
                expected.append(extra);errors.append(error);frozen.append(old)
            np.testing.assert_array_equal(expected,a['damping_extra_rad'])
            np.testing.assert_array_equal(errors,a['actual_velocity_error_rad_s'])
            np.testing.assert_array_equal(frozen,a['local_hip_extra_rad'])
            for name in ['same_state_pre_slew_direct_new_rad','same_state_post_slew_direct_new_rad']:
                assert np.max(a[name]*a['actual_velocity_error_rad_s'])<=1e-13
            if case is None:
                np.testing.assert_array_equal(expected,np.zeros((count,14)))
                np.testing.assert_array_equal(errors,np.zeros((count,14)))
            n=min(count,len(baseline['observations']));target_diff=a['applied'][:n]-baseline['applied'][:n]
            response=target_diff-a['same_state_post_slew_direct_new_rad'][:n]
            first=lambda v:next((int(i) for i in np.flatnonzero(np.max(np.abs(v),axis=1)>1e-8)),None)
            r=dict(case_seed=case,initial_hash=row['initial_hash'],label=label(row,b),success=row['success'],valid=row['valid'],controls=row['controls'],peaks=row['peaks'],
                first_target_difference_control=first(target_diff),first_sensor_difference_control=first(a['observations'][:n]-baseline['observations'][:n]),
                max_new_rad=float(np.abs(a['damping_extra_rad']).max()),maximum_combined_14_joint_correction_rad=row['maximum_combined_14_joint_correction_rad'],
                early_direct_rad=float(np.abs(a['same_state_post_slew_direct_new_rad'][:min(n,50)]).max()),
                early_subsequent_state_history_response_rad=float(np.abs(response[:min(n,50)]).max()),
                max_velocity_error_rad_s=float(np.abs(a['actual_velocity_error_rad_s']).max()),
                scalar_feedback_exact=True,sign_invariant_after_limits=True)
            rows.append(r)
            if s in representatives:
                dest=args.output/'representatives'/Path(s).relative_to(source/'training')/f'case_{case}';dest.mkdir(parents=True)
                np.savez_compressed(dest/'paired_signals.npz',**a,target_difference_rad=target_diff,subsequent_state_history_response_rad=response,
                    baseline_observations=baseline['observations'][:n],baseline_applied=baseline['applied'][:n])
        programs.append(dict(directory=s,parameters=report['rows'][0]['parameters'],successes=report['successes'],physical_failures=report['physical_failures'],
            rescued=[r['case_seed'] for r in rows if r['label']=='rescued'],regressed=[r['case_seed'] for r in rows if r['label']=='regressed'],rows=rows))
    parity=[]
    for a,b in zip(terminal['candidate']['rows'],terminal['baseline']['rows']):
        assert a['case_seed']==b['case_seed'] and a['initial_hash']==b['initial_hash'] and a['peaks']==b['peaks']
        case=a['case_seed'];files=[source/'development_candidate'/f'case_{case}/trajectory.npz',source/'development_baseline'/f'case_{case}/trajectory.npz',run.previous.prior.OUTPUT/'candidate'/f'case_{case}/trajectory.npz']
        for f in files:remember(f)
        with np.load(files[0],allow_pickle=False) as x,np.load(files[1],allow_pickle=False) as y,np.load(files[2],allow_pickle=False) as z:
            equal={k:bool(np.array_equal(x[k],y[k])) for k in x.files};prior={k:bool(np.array_equal(y[k],z[k])) for k in z.files}
        assert all(equal.values()) and all(prior.values());parity.append(dict(case_seed=case,all_candidate_baseline_fields_exact=equal,baseline_R157_fields_exact=prior))
    assert all(digest(f)==h for f,h in source_hashes.items()),'Source changed during read-only audit'
    summary=dict(read_only=True,no_MjData_environment_forward_or_integration=True,no_outcome_or_validity_relabeling=True,
        smoke=args.smoke,distinct_actual_programs=len(paths),nonzero_programs=len(nonzero),actual_training_trajectories=25*len(paths),
        generations=4,selected_parameters=terminal['parameters'],candidate_successes=terminal['candidate']['successes'],baseline_successes=terminal['baseline']['successes'],
        candidate_physical_failures=terminal['candidate']['physical_failures'],development_gate=terminal['original_development_gate'],
        best_nonzero=best_any,best_all_valid_nonzero=best_valid,all_valid_nonzero_programs=len(valid),programs=programs,terminal_parity=parity,
        paired_algebra_not_dynamic_counterfactual=True,sign_invariance_not_total_energy_or_recovery_proof=True,
        qpos_qvel_only_read_for_separate_original_field_parity_not_feedback_inputs=True,
        source_hashes_unchanged=True,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False)
    write(args.output/'results.json',summary);write(args.output/'source_hashes.json',source_hashes)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['programs','terminal_parity']}),flush=True)
    for p in programs:print('R166_PROGRAM',Path(p['directory']).relative_to(source),p['successes'],p['physical_failures'],p['rescued'],p['regressed'],flush=True)

if __name__=='__main__':main()
