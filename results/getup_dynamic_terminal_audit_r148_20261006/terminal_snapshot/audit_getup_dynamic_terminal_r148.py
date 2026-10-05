"""Read-only terminal and actual activation audit. No integration or relabeling."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_dynamic_gain_r147 as run
from diagnostics.getup_independent_native import digest

OUTPUT=run.ROOT/'outputs/getup_dynamic_terminal_audit_r148_20261006'

def main():
    OUTPUT.mkdir(exist_ok=False)
    terminal=json.loads((run.OUTPUT/'results.json').read_text())
    closed=json.loads((run.OUTPUT/'training_closed.json').read_text())
    assert len(closed['history'])==8 and not np.any(terminal['parameters'])
    assert not terminal['original_development_gate'] and not terminal['independent_qualification_run']
    baseline=terminal['baseline'];candidate=terminal['candidate']
    exact=[];source_hashes={}
    for a,b in zip(candidate['rows'],baseline['rows']):
        assert a['case_seed']==b['case_seed'] and a['initial_hash']==b['initial_hash']
        pa=run.OUTPUT/'development_candidate'/f'case_{a["case_seed"]}'/'trajectory.npz'
        pb=run.OUTPUT/'development_baseline'/f'case_{b["case_seed"]}'/'trajectory.npz'
        with np.load(pa,allow_pickle=False) as x,np.load(pb,allow_pickle=False) as y:
            equal={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
        assert all(equal.values()),equal
        assert all(b['frozen_R134_trace_bitwise_equal'].values())
        exact.append(dict(case_seed=a['case_seed'],initial_hash=a['initial_hash'],fields=equal))
        source_hashes[str(pa)]=digest(pa);source_hashes[str(pb)]=digest(pb)
    histories=closed['history']
    locations=sorted({p for h in histories for p in h['closed_trial_directories']})
    zero=json.loads((run.OUTPUT/'training/generation_0001/candidate_00/results.json').read_text())
    zero_rows={r['case_seed']:r for r in zero['rows']}
    programs=[]
    for location in locations:
        path=Path(location);report=json.loads((path/'results.json').read_text())
        source_hashes[str(path/'results.json')]=digest(path/'results.json')
        rescue=[];regress=[];details=[]
        for row in report['rows']:
            case=row['case_seed'];base=zero_rows[case]
            assert row['initial_hash']==base['initial_hash']
            if case is not None:
                if row['success'] and not base['success']:rescue.append(case)
                if base['success'] and not row['success']:regress.append(case)
            trace=path/f'case_{case}'/'trajectory.npz'
            with np.load(trace,allow_pickle=False) as z:
                act=z['dynamic_activation'];gains=z['dynamic_gains']
                delta=np.abs(gains-np.asarray(row['base_gains']))
                first=np.flatnonzero(np.any(delta!=0,axis=1))
                details.append(dict(case_seed=case,success=row['success'],valid=row['valid'],
                    saved_controls=row['controls'],first_gain_change_control=int(first[0]) if len(first) else None,
                    max_gain_change=float(delta.max()),max_abs_activation=float(np.abs(act).max()),
                    activation_abs_over_095_controls=int(np.sum(np.abs(act)>=.95)),
                    max_combined_correction_rad=row['maximum_combined_right_hip_correction_rad'],
                    original_peaks=row['peaks']))
                assert act[0]==0 and np.array_equal(gains[0],row['base_gains'])
                if case is None:assert not np.any(act)
            source_hashes[str(trace)]=digest(trace)
        programs.append(dict(path=str(path),parameters=report['rows'][0]['parameters'],
            successes=report['successes'],physical_failures=report['physical_failures'],
            nominal_success=report['nominal_success'],return_sum=report['return_sum'],
            short_rescued=rescue,short_regressed=regress,actual_feedback=details))
    nonzero=[p for p in programs if np.any(p['parameters'])]
    valid=[p for p in nonzero if not p['physical_failures']]
    best=max(nonzero,key=lambda p:(p['successes'],-p['physical_failures'],p['return_sum']))
    best_valid=max(valid,key=lambda p:(p['successes'],p['return_sum']))
    probe=run.OUTPUT/'nonzero_smoke'
    probe_report=json.loads((probe/'results.json').read_text())
    proposal=np.r_[np.array([.1,-.1,.1,.05,-.05,.05]),np.full(12,.2)]
    assert all(np.array_equal(r['parameters'],proposal) for r in probe_report['rows'])
    found=any(np.array_equal(p['parameters'],proposal) for p in programs)
    result=dict(read_only=True,no_dynamic_integration=True,no_relabeling=True,
        candidate_successes=candidate['successes'],baseline_successes=baseline['successes'],
        candidate_physical_failures=candidate['physical_failures'],baseline_physical_failures=baseline['physical_failures'],
        candidate_is_exact_zero=True,full_pair_exact=exact,closed_generations=8,
        distinct_programs=len(programs),nonzero_programs=len(nonzero),short_rollouts=25*len(programs),
        best_nonzero_short=best,best_all_valid_nonzero_short=best_valid,
        all_programs=programs,fixed_smoke_probe=probe_report,smoke_probe_in_CEM=found,
        hypothesis_for_next_probe='The previously fixed nonzero smoke program rescued known 769002, but was not uniformly tested. Verify the same frozen numeric function on all 24 known starts before proposing another search.',
        short_labels_not_30s_acceptance=True,independent_qualification_run=False,
        full_task_completed=False,hardware_readiness=False)
    run.local.write_json(OUTPUT/'results.json',result)
    run.local.write_json(OUTPUT/'source_hashes.json',source_hashes)
    shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    print(json.dumps({k:result[k] for k in ('candidate_successes','baseline_successes','distinct_programs','nonzero_programs','short_rollouts','smoke_probe_in_CEM')}),flush=True)
    print('BEST_NONZERO',best['path'],best['successes'],best['physical_failures'],best['short_rescued'],best['short_regressed'],flush=True)
    print('BEST_VALID_NONZERO',best_valid['path'],best_valid['successes'],best_valid['short_rescued'],best_valid['short_regressed'],flush=True)

if __name__=='__main__':main()
