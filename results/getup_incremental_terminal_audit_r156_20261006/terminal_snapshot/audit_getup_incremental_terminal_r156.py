"""Read-only terminal audit of R155 direct residuals, no integration/relabeling."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_incremental_residual_r155 as run
from diagnostics.getup_independent_native import digest

OUTPUT=run.ROOT/'outputs/getup_incremental_terminal_audit_r156_20261006'

def main():
    assert not OUTPUT.exists()
    terminal=json.loads((run.OUTPUT/'results.json').read_text())
    closed=json.loads((run.OUTPUT/'training_closed.json').read_text())
    assert len(closed['history'])==6 and not np.any(terminal['parameters'])
    assert not terminal['original_development_gate'] and not terminal['independent_qualification_run']
    OUTPUT.mkdir();shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    hashes={};exact=[]
    def load(path):
        for name in ['trajectory.npz','result.json']:hashes[str(path/name)]=digest(path/name)
        with np.load(path/'trajectory.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        return arrays,json.loads((path/'result.json').read_text())
    for a,b in zip(terminal['candidate']['rows'],terminal['baseline']['rows']):
        case=a['case_seed'];assert case==b['case_seed'] and a['initial_hash']==b['initial_hash'] and a['peaks']==b['peaks']
        x,_=load(run.OUTPUT/'development_candidate'/f'case_{case}')
        y,_=load(run.OUTPUT/'development_baseline'/f'case_{case}')
        old,_=load(run.prior.selector.OUTPUT/'candidate'/f'case_{case}')
        equal={k:bool(np.array_equal(x[k],y[k])) for k in x}
        assert all(equal.values()) and all(np.array_equal(y[k],old[k]) for k in old)
        assert not np.any(x['incremental_extra_rad'])
        exact.append(dict(case_seed=case,initial_hash=a['initial_hash'],fields=equal,R134_exact=True,peaks_exact=True))
    locations=sorted({p for h in closed['history'] for p in h['closed_trial_directories']})
    zero=json.loads((run.OUTPUT/'training/generation_0001/candidate_00/results.json').read_text())
    base_rows={r['case_seed']:r for r in zero['rows']};programs=[]
    for location in locations:
        path=Path(location);report=json.loads((path/'results.json').read_text());hashes[str(path/'results.json')]=digest(path/'results.json')
        rescued=[];regressed=[];details=[]
        for row in report['rows']:
            case=row['case_seed'];base=base_rows[case];assert row['initial_hash']==base['initial_hash']
            if case is not None:
                if row['success'] and not base['success']:rescued.append(case)
                if base['success'] and not row['success']:regressed.append(case)
            x,_=load(path/f'case_{case}');act=x['incremental_activation'];p=np.array(row['parameters']);n=min(len(act),529)
            expected=-.18*np.tanh(p[:3]*act[:n,:3]+p[3:]*act[:n,3:])
            np.testing.assert_array_equal(x['incremental_extra_rad'][:n],expected)
            assert not np.any(x['incremental_extra_rad'][529:]) and not np.any(act[529:])
            np.testing.assert_array_equal(act[0],np.zeros(6))
            np.testing.assert_array_equal(x['incremental_extra_rad'][0],np.zeros(3))
            if case is None:assert not np.any(act) and not np.any(x['local_hip_extra_rad'])
            first=np.flatnonzero(np.any(x['incremental_extra_rad']!=0,axis=1))
            details.append(dict(case_seed=case,success=row['success'],valid=row['valid'],controls=row['controls'],
                first_incremental_nonzero_control=int(first[0]) if len(first) else None,
                max_abs_activation_by_channel=np.abs(act).max(0).tolist(),
                max_abs_incremental_by_joint_rad=np.abs(x['incremental_extra_rad']).max(0).tolist(),
                max_abs_frozen_feedback_by_joint_rad=np.abs(x['local_hip_extra_rad']).max(0).tolist(),
                maximum_combined_right_hip_correction_rad=row['maximum_combined_right_hip_correction_rad'],
                original_peaks=row['peaks']))
        programs.append(dict(path=str(path),parameters=report['rows'][0]['parameters'],successes=report['successes'],
            physical_failures=report['physical_failures'],nominal_success=report['nominal_success'],return_sum=report['return_sum'],
            short_rescued=rescued,short_regressed=regressed,actual_feedback=details))
    nonzero=[p for p in programs if np.any(p['parameters'])];valid=[p for p in nonzero if not p['physical_failures']]
    best=max(nonzero,key=lambda p:(p['successes'],-p['physical_failures'],p['return_sum']))
    best_valid=max(valid,key=lambda p:(p['successes'],p['return_sum'])) if valid else None
    run.prior.local.init_worker();ids=np.array([9,10,11]);scalar=[]
    selected_paths={best['path'],str(run.OUTPUT/'nonzero_smoke')}
    if best_valid:selected_paths.add(best_valid['path'])
    for location in sorted(selected_paths):
        path=Path(location);report=json.loads((path/'results.json').read_text())
        for row in report['rows']:
            x,_=load(path/f'case_{row["case_seed"]}');obs=x['observations'];n=min(len(obs),529);activation=[];extra=[];frozen=[]
            for k in range(n):
                e,a=run.incremental_feedback(row['parameters'],obs[k],run.prior.local.NOMINAL[k],obs[0],run.prior.local.NOMINAL[0],ids)
                activation.append(a);extra.append(e)
                frozen.append(run.prior.local.local_feedback(obs[k],run.prior.local.NOMINAL[k],ids,row['base_gains']))
            np.testing.assert_array_equal(activation,x['incremental_activation'][:n]);np.testing.assert_array_equal(extra,x['incremental_extra_rad'][:n])
            np.testing.assert_array_equal(frozen,x['local_hip_extra_rad'][:n])
            scalar.append(dict(path=location,case_seed=row['case_seed'],scalar_activation_residual_frozen_feedback_exact=True,controls=n))
    for p,h in hashes.items():assert digest(p)==h
    result=dict(read_only=True,no_dynamic_integration=True,no_relabeling=True,closed_generations=6,
        terminal_parameters=terminal['parameters'],candidate_successes=terminal['candidate']['successes'],baseline_successes=terminal['baseline']['successes'],
        candidate_physical_failures=terminal['candidate']['physical_failures'],baseline_physical_failures=terminal['baseline']['physical_failures'],
        full_pair_exact=exact,distinct_programs=len(programs),nonzero_programs=len(nonzero),short_rollouts=25*len(programs),
        nonzero_all_valid_programs=len(valid),best_nonzero_short=best,best_all_valid_nonzero_short=best_valid,
        all_programs=programs,scalar_execution_checks=scalar,smoke_probe_in_CEM=any(np.array_equal(p['parameters'],run.PROBE) for p in programs),
        short_labels_not_30s_acceptance=True,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False)
    run.prior.local.write_json(OUTPUT/'results.json',result);run.prior.local.write_json(OUTPUT/'source_hashes.json',hashes)
    print(json.dumps({k:result[k] for k in ['distinct_programs','nonzero_programs','short_rollouts','nonzero_all_valid_programs','smoke_probe_in_CEM']}),flush=True)
    for name,p in [('BEST_NONZERO',best),('BEST_VALID',best_valid)]:
        if p:print(name,p['path'],p['successes'],p['physical_failures'],p['short_rescued'],p['short_regressed'],flush=True)

if __name__=='__main__':main()
