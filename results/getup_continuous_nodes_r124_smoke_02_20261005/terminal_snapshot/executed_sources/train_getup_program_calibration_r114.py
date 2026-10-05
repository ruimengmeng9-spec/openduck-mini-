"""R114 frozen encoder, least-norm supervised node decoder calibration.

No context lookup: one fixed parametric model, no case data in inference.
Physics, original feedback, entry deadline and standing audit are unchanged.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_program_r113 as program
from diagnostics.getup_reference_env_r100 import TRAIN
from diagnostics.probe_getup_anchor_r101 import ROOT
from diagnostics.train_getup_joint_anchor_r102 import write_json,aggregate
from diagnostics.getup_independent_native import digest

OUTPUT=ROOT/'outputs/getup_program_calibration_r114_left_20261005'
SOURCE=ROOT/'outputs/getup_program_r113_left_20261005'


def hidden_design(w,contexts):
    rows=[]
    for context in contexts:
        x=program.normalize(program.checked_context(context),w['context_mean'],w['context_std'])
        for name in ('hidden0','hidden1'):x=np.tanh(x@w[name+'_kernel']+w[name+'_bias'])
        rows.append(np.r_[x.astype(np.float64),1.])
    return np.stack(rows)


def calibrate(w,contexts,teacher_nodes):
    labels=np.asarray(teacher_nodes,dtype=np.float64)
    assert labels.shape==(25,6,10) and np.isfinite(labels).all() and np.abs(labels).max()<1.
    assert np.array_equal(labels[0],np.zeros((6,10)))
    design=hidden_design(w,contexts)
    prior=np.vstack((w['knots_kernel'],w['knots_bias'])).astype(np.float64)
    # All known examples supervise one common decoder. No observation/library is
    # retained in its payload; the resulting matrix also accepts unseen sensors.
    targets=np.arctanh(labels.reshape(25,60))
    delta,residuals,rank,singular=np.linalg.lstsq(design,targets-design@prior,rcond=None)
    assert rank==25 and np.isfinite(delta).all()
    trained={k:v.copy() for k,v in w.items()}
    trained['knots_kernel']=(prior+delta)[:-1];trained['knots_bias']=(prior+delta)[-1]
    for name in w:
        if name not in ('knots_kernel','knots_bias'):np.testing.assert_array_equal(trained[name],w[name])
    before=[program.predict_program(w,c) for c in contexts]
    after=[program.predict_program(trained,c) for c in contexts]
    assert all(a[0]==b[0] and a[2]==b[2] for a,b in zip(before,after))
    np.testing.assert_array_equal(after[0][1],np.zeros((6,10)))
    fitted=np.stack([r[1] for r in after]);error=float(np.abs(fitted-labels).max())
    assert error<1e-10
    return trained,dict(design_shape=list(design.shape),rank=int(rank),singular_values=singular.tolist(),
        condition_number=float(singular[0]/singular[-1]),max_coefficient_change=float(np.abs(delta).max()),
        effective_node_max_error=error,effective_node_mse=float(np.mean((fitted-labels)**2)),
        all_other_arrays_bitwise_unchanged=True,flag_logits_bitwise_unchanged=True,nominal_nodes_exact_zero=True,
        closed_form_minimum_norm_decoder_update=True,parameter_fit_not_recovery_success=True)


def audit_source(result,contexts,labels):
    rows=[]
    for report in result['candidates']:
        with np.load(report['model'],allow_pickle=False) as data:w={k:data[k].copy() for k in data.files}
        for case,context,label,trial in zip((None,*TRAIN),contexts,labels,report['rows']):
            with np.load(program.TEACHERS/f'case_{case}/teacher_parameters.npz') as data:profile=int(data['profile'])
            predicted,nodes,meta=program.predict_program(w,context)
            rows.append(dict(model=report['model'],case_seed=case,success=trial['success'],valid=trial['valid'],
                teacher_has_nodes=bool(np.any(label!=0)),profile_matches=profile==predicted,
                gate_matches=meta['has_knots']==bool(np.any(label!=0)),max_node_error=float(np.abs(nodes-label).max())))
    return dict(rows=rows,read_only=True,no_dynamic_relabeling=True,association_not_unique_cause=True)


def main():
    OUTPUT.mkdir(exist_ok=False);sources=OUTPUT/'executed_sources';sources.mkdir()
    names=(Path(__file__).name,'test_getup_program_calibration_r114.py','launch_getup_program_calibration_r114.py',
        'train_getup_program_r113.py','test_getup_program_r113.py','getup_reference_env_r100.py','probe_getup_anchor_r101.py',
        'train_getup_joint_anchor_r102.py','getup_independent_native.py','validate_getup_fullpath_r27.py',
        'train_getup_fullpath_r27.py','getup_fullfallen_env_r32.py','getup_fullfallen_contract_r32.py',
        'search_getup_reference_feedback_r64.py','search_getup_sustained_bridge_r42.py')
    for name in names:shutil.copy2(Path(__file__).with_name(name),sources/name)
    previous=json.loads((SOURCE/'results.json').read_text());assert not previous['independent_qualification_run']
    selected=max(previous['candidates'],key=lambda r:(r['nominal_success'],r['successes'],-r['physical_failures'],r['return_sum']))
    assert selected['successes']==21 and selected['physical_failures']==0 and selected['nominal_success']
    x,flags,ignored,frozen,records=program.load_program_data()
    labels=[]
    for case in (None,*TRAIN):
        with np.load(program.TEACHERS/f'case_{case}/teacher_parameters.npz') as data:labels.append(data['knots'].copy())
    labels=np.stack(labels)
    write_json(OUTPUT/'source_failure_audit.json',audit_source(previous,x,labels))
    with np.load(selected['model'],allow_pickle=False) as data:w={k:data[k].copy() for k in data.files}
    trained,fit=calibrate(w,x,labels);model=OUTPUT/'calibrated_model.npz';np.savez_compressed(model,**trained)
    np.savez_compressed(OUTPUT/'closed_supervised_learner.npz',design=hidden_design(w,x),targets=np.arctanh(labels.reshape(25,60)),
        prior_decoder=np.vstack((w['knots_kernel'],w['knots_bias'])),trained_decoder=np.vstack((trained['knots_kernel'],trained['knots_bias'])))
    write_json(OUTPUT/'fit_report.json',fit)
    write_json(OUTPUT/'rng.json',dict(deterministic_closed_form=True,no_new_sampling=True,frozen_encoder_seed=213))
    write_json(OUTPUT/'contract.json',dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        hypothesis='R113 failures are confined to node teachers despite correct flags; calibrating only its continuous decoder tests sensitivity to parameter fitting error',
        source_model=selected['model'],source_sha256=digest(selected['model']),model_sha256=digest(model),
        initial_actual_context_dim=55,no_case_id_or_lookup_at_inference=True,teacher_parameters_supervised_labels_only=True,
        inference_no_design_matrix_or_targets=True,encoder_and_flags_frozen=True,node_decoder_precision='float64',
        one_least_norm_supervised_decoder_fit=True,original_physics_rewards_acceptance_unchanged=True,
        control_hz=50,physics_hz=500,full_path_controls=2279,entry_deadline_s=12,strict_tail_s=30,
        correction_limit_rad=.18,root_edits_after_initialization=0,teachers=records,
        hashes={str(f):digest(f) for f in [program.SCENE,program.STAND,program.REFERENCE,program.MODEL,*sources.iterdir()]}))
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        # Independent scalar nominal regression before full learned development.
        nominal=program.evaluate(pool,str(model),[None],OUTPUT/'independent_nominal_smoke')
        assert nominal['nominal_success'] and nominal['rows'][0]['original_full_path_bitwise_parity']
        development=program.evaluate(pool,str(model),[None,*TRAIN],OUTPUT/'full_development')
        assert all(c['initial_hash']==b['initial_hash'] for c,b in zip(development['rows'],selected['rows']))
        write_json(OUTPUT/'progress.json',dict(development=development))
        print('R114_DEVELOPMENT',development['successes'],development['physical_failures'],flush=True)
        independent=None
        if development['nominal_success'] and development['successes']>=22 and development['physical_failures']==0:
            candidate=OUTPUT/'frozen_candidate.npz';shutil.copy2(model,candidate)
            write_json(OUTPUT/'frozen_candidate.json',dict(sha256=digest(candidate),before_independent_evaluation=True))
            seeds=list(range(3160000,3160040))
            trials=program.evaluate(pool,str(candidate),seeds,OUTPUT/'qualification_candidate')
            jobs=[(str(candidate),s,str(OUTPUT/'qualification_baseline'/f'case_{s}'),None,
                dict(profile=0,knots=np.zeros((6,10)).tolist())) for s in seeds]
            baseline=aggregate(list(pool.map(program.full_trial,jobs)))
            write_json(OUTPUT/'qualification_baseline/results.json',baseline)
            assert all(c['initial_hash']==b['initial_hash'] for c,b in zip(trials['rows'],baseline['rows']))
            counts=[sum(r['success'] for r in trials['rows'][i:i+20]) for i in (0,20)]
            independent=dict(candidate=trials,baseline=baseline,group_successes=counts,
                left_stage_passed=min(counts)>=18 and trials['physical_failures']==0)
            print('R114_INDEPENDENT',counts,trials['physical_failures'],flush=True)
        write_json(OUTPUT/'results.json',dict(source_selected=selected,fit=fit,development=development,independent=independent,
            independent_qualification_run=independent is not None,left_stage_passed=bool(independent and independent['left_stage_passed']),
            full_task_completed=False,hardware_readiness=False,simulation_only=True))
    print('R114_TERMINAL',flush=True)


if __name__=='__main__':main()
