"""R116 unified causal program predictor trained on 65 complete teachers.

Expanded former qualification states are development. New qualification starts
remain unseen until a frozen candidate passes full paired development gates.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_program_r113 as program
from diagnostics.train_getup_joint_anchor_r102 import write_json,aggregate
from diagnostics.getup_independent_native import digest
from diagnostics.search_getup_expanded_teachers_r115 import OUTPUT as TEACHER_RUN,EXPANDED
from diagnostics.train_getup_program_calibration_r114 import hidden_design

ROOT=program.ROOT
OUTPUT=ROOT/'outputs/getup_expanded_program_r116_left_20261005'
CASES=[None,*program.TRAIN,*EXPANDED]


def full_trial(job):
    row=program.full_trial(job)
    row['independent_qualification_trial']=job[1] in range(3180000,3180040)
    row['former_qualification_now_development']=job[1] in EXPANDED
    write_json(Path(job[2])/'result.json',row)
    return row


def evaluate(pool,model,seeds,dest):
    jobs=[(model,s,str(dest/f'case_{s}'),str(program.TEACHERS/'case_None') if s is None else None,None) for s in seeds]
    rows=list(pool.map(full_trial,jobs));result=aggregate(rows);write_json(dest/'results.json',result)
    return result


def load_data():
    terminal=json.loads((TEACHER_RUN/'results.json').read_text())
    assert terminal['verified_teacher_count']==65 and not terminal['uncovered']
    contexts=[];flags=[];nodes=[];records=[]
    for case in CASES:
        folder=TEACHER_RUN/'teachers'/f'case_{case}'
        result=json.loads((folder/'result.json').read_text())
        assert result['case_seed']==case and result['qualified_teacher'] and result['valid'] and result['controls']==2279
        assert result['strict_tail_s']>=30.-1e-8 and result['entry_time_s']<=12.
        with np.load(folder/'trajectory.npz',allow_pickle=False) as data:context=program.checked_context(data['observations'][0]).copy()
        with np.load(folder/'teacher_parameters.npz',allow_pickle=False) as data:profile=int(data['profile']);knots=data['knots'].copy()
        assert profile in (0,1) and knots.shape==(6,10) and np.abs(knots).max()<1.
        contexts.append(context);flags.append([profile,bool(np.any(knots!=0))]);nodes.append(knots)
        records.append(dict(case_seed=case,initial_hash=result['initial_hash'],trajectory_sha256=digest(folder/'trajectory.npz'),
            parameter_sha256=digest(folder/'teacher_parameters.npz')))
    x=np.stack(contexts);y=np.array(flags,dtype=np.float32);nodes=np.stack(nodes).astype(np.float64)
    _,_,_,frozen,_=program.load_program_data()
    frozen.update(context_mean=x.mean(0,dtype=np.float64).astype(np.float32),
        context_std=np.maximum(x.std(0,dtype=np.float64),1e-4).astype(np.float32),nominal_context=x[0].copy())
    assert np.array_equal(nodes[0],np.zeros((6,10))) and np.array_equal(y[0],np.zeros(2))
    return x,y,nodes,frozen,records


def calibrate_decoder(w,x,nodes):
    design=hidden_design(w,x);prior=np.vstack((w['knots_kernel'],w['knots_bias'])).astype(np.float64)
    target=np.arctanh(nodes.reshape(len(x),60))
    delta,_,rank,singular=np.linalg.lstsq(design,target-design@prior,rcond=None)
    assert rank==len(x) and np.isfinite(delta).all()
    calibrated={k:v.copy() for k,v in w.items()}
    calibrated['knots_kernel']=(prior+delta)[:-1];calibrated['knots_bias']=(prior+delta)[-1]
    for key in w:
        if key not in ('knots_kernel','knots_bias'):np.testing.assert_array_equal(calibrated[key],w[key])
    np.testing.assert_array_equal(program.predict_program(calibrated,x[0])[1],np.zeros((6,10)))
    raw=np.stack([program.scalar_predict(calibrated,c)[1] for c in x])
    error=float(np.abs(raw-raw[0]-nodes).max());assert error<1e-10
    return calibrated,dict(rank=int(rank),design_shape=list(design.shape),condition_number=float(singular[0]/singular[-1]),
        max_coefficient_change=float(np.abs(delta).max()),ungated_node_max_error=error,nominal_nodes_exact_zero=True,
        other_arrays_bitwise_unchanged=True,supervised_fit_not_recovery_success=True),design,target


def fit(output):
    import jax
    import jax.numpy as jnp
    from flax import linen as nn,serialization
    import optax
    x,y,nodes,frozen,records=load_data()
    nx=program.normalize(x,frozen['context_mean'],frozen['context_std'])
    class Encoder(nn.Module):
        @nn.compact
        def __call__(self,x):
            x=nn.tanh(nn.Dense(128,name='hidden0')(x));x=nn.tanh(nn.Dense(128,name='hidden1')(x))
            return nn.Dense(2,name='flags')(x),nn.tanh(nn.Dense(60,name='knots')(x)).reshape(x.shape[:-1]+(6,10))
    encoder=Encoder();key=jax.random.PRNGKey(216);key,initkey=jax.random.split(key)
    params=encoder.init(initkey,jnp.zeros(55))['params']
    opt=optax.chain(optax.clip_by_global_norm(1.),optax.adam(optax.linear_schedule(1e-3,1e-4,3000)))
    state=opt.init(params)
    @jax.jit
    def update(params,state):
        def loss(p):
            logits,raw=encoder.apply({'params':p},jnp.array(nx));anchored=jnp.clip(raw-raw[0],-1.,1.)
            return jnp.mean(optax.sigmoid_binary_cross_entropy(logits,jnp.array(y)))+100.*jnp.mean((anchored-jnp.array(nodes,dtype=jnp.float32))**2)
        value,grad=jax.value_and_grad(loss)(params);changes,state=opt.update(grad,state,params)
        return optax.apply_updates(params,changes),state,value
    def export():
        w={name+'_'+kind:np.array(params[name][kind]) for name in ('hidden0','hidden1','flags','knots') for kind in ('kernel','bias')}
        w.update(frozen);return w
    output.mkdir(exist_ok=False);np.savez_compressed(output/'initial_encoder.npz',**export())
    write_json(output/'data_manifest.json',dict(teachers=records,examples=65,actual_context_dim=55,
        teacher_programs_labels_only=True,no_case_identifiers_or_context_library_in_inference=True))
    history=[]
    for step in range(1,3001):
        params,state,value=update(params,state);value=float(value);assert np.isfinite(value)
        if step%100==0:history.append(dict(update=step,objective=value));write_json(output/'history.json',history)
        if step in (1000,3000):
            w=export();path=output/f'encoder_{step:05d}.npz';np.savez_compressed(path,**w)
            (output/f'learner_{step:05d}.msgpack').write_bytes(serialization.to_bytes(dict(params=params,opt_state=state,key=key)))
            predicted=[program.predict_program(w,c) for c in x]
            report=dict(update=step,profile_correct=sum(p[0]==int(f[0]) for p,f in zip(predicted,y)),
                node_gate_correct=sum(p[2]['has_knots']==bool(f[1]) for p,f in zip(predicted,y)),model_sha256=digest(path),fit_only=True)
            write_json(output/f'report_{step:05d}.json',report);print('R116_CHECKPOINT',json.dumps(report),flush=True)
    w=export();calibrated,calibration,design,target=calibrate_decoder(w,x,nodes)
    model=output/'calibrated_model.npz';np.savez_compressed(model,**calibrated)
    np.savez_compressed(output/'closed_decoder_learner.npz',design=design,target=target,prior_decoder=np.vstack((w['knots_kernel'],w['knots_bias'])),
        trained_decoder=np.vstack((calibrated['knots_kernel'],calibrated['knots_bias'])))
    write_json(output/'calibration_report.json',calibration)
    write_json(output/'rng.json',dict(jax_key=np.array(key).tolist(),seed=216,full_batch_no_sampling=True,decoder_deterministic_closed_form=True))
    write_json(output/'results.json',dict(completed_updates=3000,model=str(model),sha256=digest(model),fit_only=True))
    return str(model)


def main():
    OUTPUT.mkdir(exist_ok=False);sources=OUTPUT/'executed_sources';sources.mkdir()
    names=(Path(__file__).name,'test_getup_expanded_program_r116.py','launch_getup_expanded_program_r116.py',
        'train_getup_program_r113.py','train_getup_program_calibration_r114.py','search_getup_expanded_teachers_r115.py',
        'getup_reference_env_r100.py','probe_getup_anchor_r101.py','train_getup_joint_anchor_r102.py','getup_independent_native.py',
        'validate_getup_fullpath_r27.py','train_getup_fullpath_r27.py','getup_fullfallen_env_r32.py',
        'getup_fullfallen_contract_r32.py','search_getup_reference_feedback_r64.py','search_getup_sustained_bridge_r42.py')
    for name in names:shutil.copy2(Path(__file__).with_name(name),sources/name)
    write_json(OUTPUT/'contract.json',dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        hypothesis='65 complete verified program teachers broaden initial sensor coverage relative to R114 25, while retaining its frozen-feedback execution interface',
        seed=216,input_dim=55,encoder_hidden_sizes=[128,128],updates=3000,learning_rate_start=1e-3,learning_rate_end=1e-4,
        classification_loss_weight=1,node_mse_weight=100,gradient_norm_cap=1,full_batch_examples=65,
        new_encoder_optimizer_rng=True,not_exact_checkpoint_resume=True,one_final_supervised_decoder_calibration=True,
        actual_initial_context_only=True,no_case_lookup=True,normalized_action_limit=1,combined_correction_limit_rad=.18,
        original_physics_rewards_acceptance_unchanged=True,control_hz=50,physics_hz=500,full_path_controls=2279,
        entry_deadline_s=12,strict_tail_s=30,root_edits_after_initialization=0,
        original_development_gate=22,expanded_development_gate=36,new_independent_seeds=list(range(3180000,3180040)),
        hashes={str(f):digest(f) for f in [program.SCENE,program.STAND,program.REFERENCE,program.MODEL,TEACHER_RUN/'results.json',*sources.iterdir()]}))
    model=fit(OUTPUT/'training')
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn')) as pool:
        nominal=evaluate(pool,model,[None],OUTPUT/'independent_nominal_smoke')
        assert nominal['nominal_success'] and nominal['rows'][0]['original_full_path_bitwise_parity']
        original=evaluate(pool,model,[None,*program.TRAIN],OUTPUT/'development_original')
        expanded=evaluate(pool,model,EXPANDED,OUTPUT/'development_expanded')
        x,y,nodes,frozen,records=load_data();initial={r['case_seed']:r['initial_hash'] for r in records}
        assert all(r['initial_hash']==initial[r['case_seed']] for r in [*original['rows'],*expanded['rows']])
        write_json(OUTPUT/'progress.json',dict(original=original,expanded=expanded))
        print('R116_FULL_DEVELOPMENT',original['successes'],expanded['successes'],original['physical_failures']+expanded['physical_failures'],flush=True)
        independent=None
        if original['nominal_success'] and original['successes']>=22 and expanded['successes']>=36 and original['physical_failures']+expanded['physical_failures']==0:
            frozen_model=OUTPUT/'frozen_candidate.npz';shutil.copy2(model,frozen_model)
            write_json(OUTPUT/'frozen_candidate.json',dict(sha256=digest(frozen_model),before_independent_evaluation=True))
            seeds=list(range(3180000,3180040))
            candidate=evaluate(pool,str(frozen_model),seeds,OUTPUT/'qualification_candidate')
            jobs=[(str(frozen_model),s,str(OUTPUT/'qualification_baseline'/f'case_{s}'),None,dict(profile=0,knots=np.zeros((6,10)).tolist())) for s in seeds]
            baseline=aggregate(list(pool.map(full_trial,jobs)))
            write_json(OUTPUT/'qualification_baseline/results.json',baseline)
            assert all(c['initial_hash']==b['initial_hash'] for c,b in zip(candidate['rows'],baseline['rows']))
            counts=[sum(r['success'] for r in candidate['rows'][i:i+20]) for i in (0,20)]
            independent=dict(candidate=candidate,baseline=baseline,group_successes=counts,left_stage_passed=min(counts)>=18 and candidate['physical_failures']==0)
            print('R116_INDEPENDENT',counts,candidate['physical_failures'],flush=True)
        write_json(OUTPUT/'results.json',dict(original=original,expanded=expanded,independent=independent,
            independent_qualification_run=independent is not None,left_stage_passed=bool(independent and independent['left_stage_passed']),
            simulation_only=True,full_task_completed=False,hardware_readiness=False))
    print('R116_TERMINAL',flush=True)


if __name__=='__main__':main()
