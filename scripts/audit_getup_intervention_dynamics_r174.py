"""Read-only first-intervention causal pairs and held-model errors, no environment."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_sensor_dynamics_r172b as math_model

ROOT=math_model.ROOT
SOURCE=ROOT/'outputs/getup_intervention_dynamics_r173_20261009'
OUTPUT=ROOT/'outputs/getup_intervention_dynamics_audit_r174_20261009'
SMOKE=ROOT/'outputs/getup_intervention_dynamics_audit_r174_smoke_20261009'
CASES=[None,*math_model.CASES]

def errors(predicted,actual):
    return {name:dict(rmse=float(np.sqrt(np.mean(((predicted[lo:hi]-actual[lo:hi])*mult)**2))),
                      actual_effect_norm=float(np.linalg.norm(actual[lo:hi]*mult)),
                      prediction_error_norm=float(np.linalg.norm((predicted[lo:hi]-actual[lo:hi])*mult)))
            for name,(lo,hi,mult) in math_model.GROUPS.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    dest=SMOKE if args.smoke else OUTPUT;assert not dest.exists()
    result=json.loads((SOURCE/'results.json').read_text());assert result['terminal_result_saved'] and not result['new_controller']
    paths=[SOURCE/'learner_closed_01/model.npz',math_model.OUTPUT/'model.npz',SOURCE/'frozen/nominal_sensor_trajectory.npz',
           SOURCE/'learner_closed_01/evaluation.json',Path(__file__),Path(math_model.__file__)]
    with np.load(paths[0],allow_pickle=False) as z:w=z['weights'].copy();blind=z['action_blind_weights'].copy()
    with np.load(paths[1],allow_pickle=False) as z:old=z['weights'].copy()
    with np.load(paths[2],allow_pickle=False) as z:nom=z['observations'][:,:50].copy();nt=z['applied'].copy()
    assert all(a.shape==(342,34) and np.isfinite(a).all() for a in (w,blind,old))
    zero=math_model.features(np.zeros(50),np.zeros(50),np.zeros(14),.2)
    for a in (w,blind,old):np.testing.assert_array_equal(zero@a,np.zeros(34))
    before={str(f):math_model.sha(f) for f in paths}
    cases=[None,769000,773007] if args.smoke else CASES
    dest.mkdir();sources=dest/'executed_sources';sources.mkdir();shutil.copy2(Path(__file__),sources/Path(__file__).name)
    shutil.copy2(Path(math_model.__file__),sources/Path(math_model.__file__).name)
    signals=dest/'signals';signals.mkdir();rows=[]
    for case in cases:
        for axis in range(3):
            pair=[]
            for program in (1+2*axis,2+2*axis):
                path=SOURCE/'episodes'/f'program_{program:02d}'/f'case_{case}'/'trajectory.npz'
                before[str(path)]=math_model.sha(path)
                with np.load(path,allow_pickle=False) as z:
                    # Whitelist only; never load root qpos/qvel or future substeps.
                    pair.append(dict(obs=z['observations'][:3,:50].astype(float),planned=z['planned_before_integration_rad'][:2].copy(),
                                     base=z['same_state_unperturbed_planned_rad'][:2].copy(),extra=z['intervention_requested_extra_rad'][:2].copy()))
            a,b=pair
            np.testing.assert_array_equal(a['obs'][:2],b['obs'][:2]);np.testing.assert_array_equal(a['planned'][0],b['planned'][0])
            np.testing.assert_array_equal(a['base'][1],b['base'][1]);np.testing.assert_array_equal(a['extra'][1],-b['extra'][1])
            fa=math_model.features(a['obs'][1]-nom[1],a['obs'][0]-nom[0],a['planned'][1]-nt[1],1/529.)
            fb=math_model.features(b['obs'][1]-nom[1],b['obs'][0]-nom[0],b['planned'][1]-nt[1],1/529.)
            actual=a['obs'][2,:34]-b['obs'][2,:34]
            predictions=[(fa-fb)@model*math_model.SCALE[:34] for model in (w,old,blind)]
            np.testing.assert_allclose(predictions[2],np.zeros(34),rtol=0,atol=1e-14)
            if case is None:
                np.testing.assert_array_equal(actual,np.zeros(34))
                for pred in predictions:np.testing.assert_array_equal(pred,np.zeros(34))
            row=dict(case_seed=case,axis=axis,held_case=case in math_model.HELD_CASES,held_axis=axis==2,
                     common_control1_state_bitwise_equal=True,planned_pair_separation_rad=float(np.abs(a['planned'][1]-b['planned'][1]).max()),
                     actual_next_native34_effect=actual.tolist(),new_model=errors(predictions[0],actual),old_R172b_model=errors(predictions[1],actual),
                     action_blind_model=errors(predictions[2],actual),first20ms_endpoint_not_physical_substep_peak=True)
            rows.append(row)
            np.savez_compressed(signals/f'case_{case}_axis_{axis}.npz',actual_next_effect=actual,new_model_effect=predictions[0],
                                old_model_effect=predictions[1],action_blind_effect=predictions[2],planned_target_pair_difference=a['planned'][1]-b['planned'][1])
    groups={}
    for split,selected in [('all',[r for r in rows if r['case_seed'] is not None]),
                           ('both_held',[r for r in rows if r['held_case'] and r['held_axis']])]:
        groups[split]=dict(pairs=len(selected),nonzero_applied_pairs=sum(r['planned_pair_separation_rad']>0 for r in selected),metrics={})
        for name in math_model.GROUPS:
            groups[split]['metrics'][name]={key:float(np.mean([r[key][name]['rmse'] for r in selected])) for key in ('new_model','old_R172b_model','action_blind_model')} if selected else {}
    closed=json.loads((SOURCE/'learner_closed_01/evaluation.json').read_text());held_summary={}
    for split in ('case_held','axis_held','both_held'):
        selected=[r for r in closed if r['split']==split and r['case_seed'] is not None]
        held_summary[split]=dict(trajectories=len(selected),metrics={})
        if selected:
            for stage in selected[0]['metrics']:
                held_summary[split]['metrics'][stage]={name:{key:float(np.mean([r['metrics'][stage][name][key] for r in selected]))
                    for key in ('action_model_rmse','action_blind_model_rmse','nominal_deviation_persistence_rmse')} for name in math_model.GROUPS}
                for name in math_model.GROUPS:
                    held_summary[split]['metrics'][stage][name]['max_abs_error_across_held_trajectories']=max(r['metrics'][stage][name]['action_model_max_abs_error'] for r in selected)
    math_model.write(dest/'pairs.json',rows);math_model.write(dest/'source_hashes.json',before)
    assert before=={f:math_model.sha(f) for f in before}
    if not args.smoke:
        previous=json.loads((SMOKE/'pairs.json').read_text())
        for oldrow in previous:assert oldrow==next(r for r in rows if r['case_seed']==oldrow['case_seed'] and r['axis']==oldrow['axis'])
    report=dict(smoke=args.smoke,read_only=True,terminal_result_saved=True,pairs=len(rows),pair_summary=groups,
                held_one_step_summary=held_summary,source_hashes_unchanged=True,new_dynamic_trajectories=0,
                scalar_fields_whitelist_no_root_reads=True,smoke_rows_exact=not args.smoke,new_controller=False,
                current_best_unified_successes=18,full_task_completed=False,hardware_readiness=False,qualification_run=False)
    math_model.write(dest/'results.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
