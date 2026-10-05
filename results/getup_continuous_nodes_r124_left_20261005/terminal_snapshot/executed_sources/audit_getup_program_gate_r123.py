"""Read-only frozen programme gate and failure audit; no dynamic integration."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_history_program_r122 as previous
from diagnostics.getup_independent_native import digest
from diagnostics.train_getup_joint_anchor_r102 import write_json

ROOT=previous.ROOT
OUTPUT=ROOT/'outputs/getup_program_gate_audit_r123_20261005'


def main():
    OUTPUT.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),OUTPUT/Path(__file__).name)
    source=previous.OUTPUT;terminal=json.loads((source/'results.json').read_text())
    with np.load(previous.audit.OUTPUT/'causal_sensor_dataset.npz',allow_pickle=False) as data:histories=data['histories'].copy()
    manifest=json.loads((previous.audit.OUTPUT/'manifest.json').read_text());cases=[r['case_seed'] for r in manifest]
    assert len(cases)==105 and not set(range(3200000,3200040)).intersection(cases)
    report={}
    for mode in ('snapshot','history'):
        file=source/'training'/(mode+'.npz')
        with np.load(file,allow_pickle=False) as data:w={k:data[k].copy() for k in data.files}
        rows=terminal['summaries'][mode]['all']['rows'];assert [r['case_seed'] for r in rows]==cases
        records=[]
        for case,h,row in zip(cases,histories,rows):
            p,knots,info=previous.scalar_program(w,h)
            np.testing.assert_array_equal(knots,np.array(row['predicted_knots']))
            assert p==row['predicted_profile'] and info==row['program_info']
            feature=previous.sensor_features(h,mode)
            raw=np.clip(((feature-w['nominal_feature'])/w['std'])@w['nodes'],-1.,1.).reshape(6,10)
            target_gate=None;target_profile=None
            if case in cases[:65]:
                with np.load(previous.audit.TEACHERS/f'teachers/case_{case}/teacher_parameters.npz',allow_pickle=False) as t:
                    target_gate=bool(np.any(t['knots']!=0.));target_profile=int(t['profile'])
            folder=source/mode/f'case_{case}'
            with np.load(folder/'trajectory.npz',allow_pickle=False) as trace:
                actions=trace['normalized_residual'];obs=trace['observations']
                assert len(actions)==row['controls'] and np.isfinite(actions).all() and np.abs(actions).max()<=1.
                # Diagnostics only; neither frozen action nor success is altered.
                up_change=np.linalg.norm(obs[:,3:6]-obs[0,3:6],axis=1)
                marked=np.flatnonzero(up_change>.15)
                first_sensor_change_s=None if not len(marked) else float(marked[0]*.02)
            record=dict(case_seed=case,initial_hash=row['initial_hash'],success=row['success'],valid=row['valid'],
                profile=p,has_knots=info['has_knots'],raw_node_max_abs=float(np.abs(raw).max()),
                gated_off_nonzero_prediction=bool(not info['has_knots'] and np.any(raw!=0.)),
                offline_teacher_has_knots=target_gate,offline_teacher_profile=target_profile,
                controls=row['controls'],invalid_control_time_s=None if row['valid'] else row['controls']*.02,
                invalid_peaks=None if row['valid'] else row['peaks'],first_up_change_diagnostic_s=first_sensor_change_s,
                action_max_abs=float(np.abs(actions).max()),trajectory_sha256=digest(folder/'trajectory.npz'))
            records.append(record)
        report[mode]=dict(model_sha256=digest(file),records=records,
            gated_off_nonzero_predictions=sum(r['gated_off_nonzero_prediction'] for r in records if r['case_seed'] is not None),
            failed_gated_off_nonzero=sum(r['gated_off_nonzero_prediction'] and not r['success'] for r in records if r['case_seed'] is not None),
            successful_gated_off_nonzero=sum(r['gated_off_nonzero_prediction'] and r['success'] for r in records if r['case_seed'] is not None),
            known_teacher_nonzero_but_gate_off=sum(r['offline_teacher_has_knots'] is True and not r['has_knots'] for r in records),
            known_teacher_zero_but_gate_on=sum(r['offline_teacher_has_knots'] is False and r['has_knots'] for r in records),
            physical_invalid_cases=[r['case_seed'] for r in records if not r['valid']],
            scalar_saved_prediction_bitwise_equal=True)
    write_json(OUTPUT/'results.json',dict(modes=report,read_only=True,no_dynamic_integration=True,
        teacher_labels_not_control_or_causal_proof=True,first_up_change_threshold_diagnostic_only=True,
        reserved_qualification_never_loaded=True,no_success_relabeling=True,
        hashes={str(p):digest(p) for p in [Path(__file__),source/'results.json',previous.audit.OUTPUT/'causal_sensor_dataset.npz']}))
    print(json.dumps({m:{k:v for k,v in r.items() if k!='records'} for m,r in report.items()}),flush=True)


if __name__=='__main__':main()
