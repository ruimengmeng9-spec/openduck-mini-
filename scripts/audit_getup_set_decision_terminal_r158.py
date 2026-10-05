"""Read-only pairing/decision audit; full labels never changed or inferred anew."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_set_decision_r157 as run
from diagnostics.getup_independent_native import digest

OUTPUT=run.ROOT/'outputs/getup_set_decision_audit_r158_20261006'

def main():
    assert not OUTPUT.exists()
    terminal=json.loads((run.OUTPUT/'results.json').read_text());assert not terminal['smoke']
    assert not terminal['original_development_gate'] and not terminal['independent_qualification_run']
    OUTPUT.mkdir();shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    hashes={};models={};rows=[];rescued=[];regressed=[]
    for name,file in [('candidate',run.OUTPUT/'training/model.npz'),('baseline',run.old.OUTPUT/'training/model.npz')]:
        hashes[str(file)]=digest(file)
        with np.load(file,allow_pickle=False) as z:models[name]={k:z[k].copy() for k in z.files}
    with np.load(run.OUTPUT/'training/offline_training_state.npz',allow_pickle=False) as z:
        sensors=z['sensors'].copy();success=z['complete_success_labels'].astype(bool);valid=z['physical_valid'].copy()
    for index,(a,b) in enumerate(zip(terminal['reports']['candidate']['rows'],terminal['reports']['baseline']['rows'])):
        case=a['case_seed'];assert case==b['case_seed'] and a['initial_hash']==b['initial_hash']
        details={}
        for name,row in [('candidate',a),('baseline',b)]:
            path=run.OUTPUT/name/f'case_{case}';hashes[str(path/'trajectory.npz')]=digest(path/'trajectory.npz');hashes[str(path/'result.json')]=digest(path/'result.json')
            with np.load(path/'trajectory.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
            np.testing.assert_array_equal(arrays['preparation_sensors'][-1],sensors[index])
            gains,choice,logits=run.old.predict(models[name],arrays['preparation_sensors'][-1])
            np.testing.assert_array_equal(gains,row['gains']);np.testing.assert_array_equal(logits,row['initial_sensor_logits']);assert choice==row['global_program_choice']
            fixed=run.old.source.OUTPUT/f'program_{choice:02d}'/f'case_{case}'
            hashes[str(fixed/'trajectory.npz')]=digest(fixed/'trajectory.npz');hashes[str(fixed/'result.json')]=digest(fixed/'result.json')
            original=json.loads((fixed/'result.json').read_text())
            assert row['success']==original['success'] and row['valid']==original['valid'] and row['peaks']==original['peaks']
            with np.load(fixed/'trajectory.npz',allow_pickle=False) as z:exact={k:bool(np.array_equal(arrays[k],z[k])) for k in arrays}
            assert all(exact.values())
            if name=='baseline':
                old=run.old.OUTPUT/'candidate'/f'case_{case}'
                hashes[str(old/'trajectory.npz')]=digest(old/'trajectory.npz')
                with np.load(old/'trajectory.npz',allow_pickle=False) as z:assert all(np.array_equal(arrays[k],z[k]) for k in arrays)
            assert row['success']==bool(success[index,choice]) and row['valid']==bool(valid[index,choice])
            if case is None:assert not np.any(arrays['local_hip_extra_rad'])
            margin=float(logits[success[index]].max()-logits[~success[index]].max()) if np.any(~success[index]) else None
            details[name]=dict(choice=choice,scalar_scores_gains_exact=True,selected_fixed_program_fields_exact=exact,
                success=row['success'],valid=row['valid'],initial_hash=row['initial_hash'],complete_success_programs=np.flatnonzero(success[index]).tolist(),
                best_success_minus_best_failure_score=margin,controls=row['controls'],entry_time_s=row['entry_time_s'],strict_tail_s=row['strict_tail_s'],original_peaks=row['peaks'])
        if case is not None:
            if a['success'] and not b['success']:rescued.append(case)
            if b['success'] and not a['success']:regressed.append(case)
        rows.append(dict(case_seed=case,models=details))
    for p,h in hashes.items():assert digest(p)==h
    result=dict(read_only=True,no_dynamic_integration=True,no_relabeling=True,rows=rows,rescued=rescued,regressed=regressed,
        candidate_successes=terminal['reports']['candidate']['successes'],baseline_successes=terminal['reports']['baseline']['successes'],
        candidate_physical_failures=terminal['reports']['candidate']['physical_failures'],baseline_physical_failures=terminal['reports']['baseline']['physical_failures'],
        known_set_count_equals_full_replay=True,decision_margins_diagnostic_not_acceptance=True,independent_qualification_run=False,hardware_readiness=False,full_task_completed=False)
    run.old.local.write_json(OUTPUT/'results.json',result);run.old.local.write_json(OUTPUT/'source_hashes.json',hashes)
    print(json.dumps({k:result[k] for k in ['candidate_successes','baseline_successes','candidate_physical_failures','baseline_physical_failures','rescued','regressed']}),flush=True)

if __name__=='__main__':main()
