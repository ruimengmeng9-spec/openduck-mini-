"""Read-only R110 closed-loop action and observation drift audit.

Teacher identity is offline diagnosis metadata only, never an actor input.
Diagnostic drift markers do not replace the frozen standing/physics gates.
"""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics.train_getup_distill_r110 import TEACHERS, ROOT, TRAIN, RECOVERY, policy_action, normalize, causal_input
from diagnostics.search_getup_case_teachers_r109 import knot_action
from diagnostics.train_getup_joint_anchor_r102 import action_for, write_json
from diagnostics.getup_reference_env_r100 import numpy_action
from diagnostics.probe_getup_anchor_r101 import MODEL
from diagnostics.getup_independent_native import digest


def first_above(values,level):
    where=np.flatnonzero(np.asarray(values)>level)
    return int(where[0]) if len(where) else None


def teacher_action(weights,anchors,gains,profile,knots,obs,k):
    if k>=RECOVERY:return np.zeros(10)
    base=np.zeros(10) if profile==0 else action_for(weights,obs,anchors[k],gains)
    return np.clip(base+knot_action(knots,k),-1.,1.)


def main():
    run=ROOT/'outputs/getup_distill_r110_left_20261005';out=ROOT/'outputs/getup_distill_audit_r111_20261005'
    result=json.loads((run/'results.json').read_text());assert not result['independent_qualification_run']
    out.mkdir(exist_ok=False);shutil.copy2(Path(__file__),out/Path(__file__).name)
    with np.load(TEACHERS/'frozen_profiles.npz') as data:anchors=data['anchors'].copy();gains=data['gains'].copy()
    with np.load(MODEL) as data:expert_weights={k:data[k].copy() for k in data.files}
    groups=[]
    for report in result['candidates']:
        model=Path(report['model']);directory=run/('development_'+model.stem)
        with np.load(model) as data:weights={k:data[k].copy() for k in data.files}
        rows=[]
        for case in TRAIN:
            path=directory/f'case_{case}';teacher=TEACHERS/f'case_{case}'
            with np.load(path/'trajectory.npz') as data:
                obs=data['observations'].copy();actual=data['normalized_residual'].copy();context=data['initial_context'].copy()
            with np.load(teacher/'trajectory.npz') as data:to=data['observations'].copy();ta=data['normalized_residual'].copy()
            with np.load(teacher/'teacher_parameters.npz') as data:profile=int(data['profile']);knots=data['knots'].copy()
            meta=json.loads((teacher/'result.json').read_text());row=json.loads((path/'result.json').read_text())
            assert row['initial_hash']==meta['initial_hash'];np.testing.assert_array_equal(context,to[0])
            n=min(len(obs),RECOVERY)
            fitted=np.stack([policy_action(weights,to[k],to[0],k) for k in range(RECOVERY)])
            replay=np.stack([policy_action(weights,obs[k],context,k) for k in range(n)])
            np.testing.assert_array_equal(replay,actual[:n])
            relabel=np.stack([teacher_action(expert_weights,anchors,gains,profile,knots,obs[k],k) for k in range(n)])
            # Labels on the original teacher states reproduce original control.
            nominal_labels=np.stack([teacher_action(expert_weights,anchors,gains,profile,knots,to[k],k) for k in range(RECOVERY)])
            np.testing.assert_array_equal(nominal_labels,ta[:RECOVERY])
            drift=np.sqrt(np.mean(((obs[:n]-to[:n])/weights['input_std'][:55])**2,axis=1))
            action_error=np.max(np.abs(actual[:n]-ta[:n]),axis=1)
            rows.append(dict(case_seed=case,success=row['success'],valid=row['valid'],controls=row['controls'],
                teacher_state_prediction_mse=float(np.mean((fitted-ta[:RECOVERY])**2)),
                executed_same_time_teacher_mse=float(np.mean((actual[:n]-ta[:n])**2)),
                executed_current_state_teacher_mse=float(np.mean((actual[:n]-relabel)**2)),
                initial_action_error=float(action_error[0]),initial_teacher_action_max=float(np.abs(ta[0]).max()),
                first_action_error_above_002_control=first_above(action_error,.02),
                first_observation_rms_drift_above_01_control=first_above(drift,.1),
                max_observation_rms_drift=float(drift.max()),
                early_50_action_mse=float(np.mean((actual[:min(n,50)]-relabel[:min(n,50)])**2)),
                initial_hash=row['initial_hash'],trajectory_sha256=digest(path/'trajectory.npz')))
            np.savez_compressed(out/f'{model.stem}_case_{case}.npz',observation_rms_drift=drift,
                action_error_same_time=action_error,current_state_expert_labels=relabel,
                teacher_state_predictions=fitted)
        groups.append(dict(model=str(model),model_sha256=digest(model),successes=report['successes'],
            physical_failures=report['physical_failures'],rows=rows))
    write_json(out/'results.json',dict(groups=groups,read_only_audit=True,no_dynamic_integration=True,
        current_state_teacher_relabeling_only_offline=True,not_a_deployed_case_selector=True,
        diagnostic_thresholds_not_acceptance=True,independent_qualification_run=False,
        source_result_sha256=digest(run/'results.json'),source_sha256=digest(Path(__file__))))
    print(json.dumps([dict(model=Path(g['model']).stem,successes=g['successes'],physical_failures=g['physical_failures'],
        median_initial_action_error=float(np.median([r['initial_action_error'] for r in g['rows']])),
        median_drift_start=[r['first_observation_rms_drift_above_01_control'] for r in g['rows']],
        teacher_mse=float(np.mean([r['teacher_state_prediction_mse'] for r in g['rows']])),
        executed_expert_mse=float(np.mean([r['executed_current_state_teacher_mse'] for r in g['rows']]))) for g in groups]),flush=True)


if __name__=='__main__':main()
