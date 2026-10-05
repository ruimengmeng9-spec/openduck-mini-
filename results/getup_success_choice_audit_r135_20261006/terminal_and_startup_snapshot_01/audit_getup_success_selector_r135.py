"""Read-only consistency and wrong-choice audit; no new policy or relabeling."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_success_selector_r134 as selector
from diagnostics.getup_independent_native import digest

ROOT=selector.ROOT
OUTPUT=ROOT/'outputs/getup_success_choice_audit_r135_20261006'


def main():
    OUTPUT.mkdir(exist_ok=False);shutil.copy2(__file__,OUTPUT/Path(__file__).name)
    candidates=json.loads((selector.OUTPUT/'candidate/results.json').read_text())
    baselines=json.loads((selector.OUTPUT/'baseline/results.json').read_text())
    with np.load(selector.OUTPUT/'training/model.npz',allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
    assert set(model)=={'mean','std','weight','bias','global_gains'}
    rows=[];rescued=[];regressed=[]
    for row,base in zip(candidates['rows'],baselines['rows']):
        case=row['case_seed'];assert case==base['case_seed'] and row['initial_hash']==base['initial_hash']
        path=selector.OUTPUT/'candidate'/f'case_{case}'
        with np.load(path/'trajectory.npz',allow_pickle=False) as z:actual={k:z[k].copy() for k in z.files}
        gains,choice,logits=selector.predict(model,actual['preparation_sensors'][-1])
        np.testing.assert_array_equal(gains,row['gains']);np.testing.assert_array_equal(logits,row['initial_sensor_logits'])
        assert choice==row['global_program_choice']
        fixed=selector.source.OUTPUT/f'program_{choice:02d}'/f'case_{case}'
        original=json.loads((fixed/'result.json').read_text());assert original['initial_hash']==row['initial_hash']
        with np.load(fixed/'trajectory.npz',allow_pickle=False) as z:equal={k:bool(np.array_equal(actual[k],z[k])) for k in actual}
        assert all(equal.values()) and original['success']==row['success'] and original['valid']==row['valid']
        alternatives=[]
        for i in range(4):
            r=json.loads((selector.source.OUTPUT/f'program_{i:02d}'/f'case_{case}'/'result.json').read_text())
            if r['success'] and r['valid']:alternatives.append(i)
        if case is not None and row['success'] and not base['success']:rescued.append(case)
        if case is not None and base['success'] and not row['success']:regressed.append(case)
        order=np.sort(logits)
        rows.append(dict(case_seed=case,choice=choice,selected_success=row['success'],selected_valid=row['valid'],
            complete_success_alternatives=alternatives,wrong_choice=not row['success'],score_margin=float(order[-1]-order[-2]),
            logits=logits.tolist(),selected_fixed_program_trace_bitwise_equal=equal,
            same_observed_context_from_actual_episode=True,source_trajectory_hash=digest(path/'trajectory.npz')))
    result=dict(rows=rows,rescued=rescued,regressed=regressed,selected_program_counts=[sum(r['choice']==i for r in rows if r['case_seed'] is not None) for i in range(4)],
        read_only=True,no_dynamic_integration=True,no_new_success_labels=True,no_unique_root_cause_claim=True,
        no_unseen_qualification=True,model_has_no_context_library=True,
        hashes={str(p):digest(p) for p in [Path(__file__),selector.OUTPUT/'training/model.npz',selector.OUTPUT/'candidate/results.json',selector.OUTPUT/'baseline/results.json']})
    (OUTPUT/'results.json').write_text(json.dumps(result,indent=2))
    print('R135_TERMINAL',json.dumps(dict(rescued=rescued,regressed=regressed,counts=result['selected_program_counts'],wrong_cases=[r['case_seed'] for r in rows if r['wrong_choice']])),flush=True)


if __name__=='__main__':main()
