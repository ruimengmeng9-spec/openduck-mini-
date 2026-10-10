"""Bounded success-set likelihood objective, unchanged linear controller capacity.

Offline complete-success labels supervise coefficients, never inference inputs.
Inference uses only actual initial native50 and original current feedback.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_success_selector_r134 as old
from diagnostics.getup_independent_native import digest

ROOT=old.ROOT
OUTPUT=ROOT/'outputs/getup_set_decision_r157_left_20261006'
SMOKE=ROOT/'outputs/getup_set_decision_r157_smoke_20261006'
PENALTIES=(.01,.1,1.)
UPDATES=1000
MODELS=None

def set_loss_gradient(logits,success):
    logits=np.asarray(logits,dtype=float);success=np.asarray(success,dtype=bool)
    if logits.shape!=success.shape or logits.ndim!=2 or not np.isfinite(logits).all() or not np.all(success.any(1)):
        raise ValueError('Finite logits and nonempty complete-valid success sets required')
    exp=np.exp(logits-logits.max(1,keepdims=True));total=exp.sum(1,keepdims=True)
    good=exp*success;mass=good.sum(1,keepdims=True)
    return float(np.mean(np.log(total)-np.log(mass))), (exp/total-good/mass)/len(logits)

def solve(x,y,penalty,save=None):
    # Same fold-local normalization and linear initialization as the R134 ridge100
    # comparison; no hidden layer, context library or nearest-neighbor inference.
    model=old.fit(x,y.astype(float),100.)
    z=(x-model['mean'])/model['std'];w=model['weight'].copy();b=model['bias'].copy()
    mw=np.zeros_like(w);vw=np.zeros_like(w);mb=np.zeros_like(b);vb=np.zeros_like(b)
    history=[];rng=np.random.default_rng(257)
    if save:save.mkdir(parents=True,exist_ok=False)
    for t in range(1,UPDATES+1):
        likelihood,d=set_loss_gradient(z@w+b,y)
        dw=z.T@d+penalty*w;db=d.sum(0)+penalty*.1*b
        mw=.9*mw+.1*dw;vw=.999*vw+.001*dw*dw
        mb=.9*mb+.1*db;vb=.999*vb+.001*db*db
        w-=.01*(mw/(1-.9**t))/(np.sqrt(vw/(1-.999**t))+1e-8)
        b-=.01*(mb/(1-.9**t))/(np.sqrt(vb/(1-.999**t))+1e-8)
        assert np.isfinite(w).all() and np.isfinite(b).all()
        if t%250==0:
            loss,_=set_loss_gradient(z@w+b,y)
            item=dict(update=t,set_likelihood_loss=loss,regularized_loss=loss+penalty*.5*(np.sum(w*w)+.1*np.sum(b*b)))
            history.append(item)
            if save:
                dest=save/f'update_{t:05d}';dest.mkdir(exist_ok=False)
                np.savez_compressed(dest/'learner.npz',weight=w,bias=b,mean=model['mean'],std=model['std'],
                    adam_weight_m=mw,adam_weight_v=vw,adam_bias_m=mb,adam_bias_v=vb,
                    updates=np.array(t),penalty=np.array(penalty),learning_rate=np.array(.01),beta1=np.array(.9),beta2=np.array(.999),epsilon=np.array(1e-8))
                old.local.write_json(dest/'rng.json',rng.bit_generator.state)
                old.local.write_json(dest/'history.json',history)
    return dict(mean=model['mean'],std=model['std'],weight=w,bias=b),history

def train(path):
    with np.load(old.OUTPUT/'training/offline_training_state.npz',allow_pickle=False) as f:
        x=f['sensors'].copy();y=f['complete_success_labels'].astype(bool);valid=f['physical_valid'].copy();gains=f['global_gains'].copy()
    assert x.shape==(25,50) and y.shape==(25,4) and np.all(y<=valid)
    path.mkdir(exist_ok=False);grid=[]
    for penalty in PENALTIES:
        choices=[];fold_states=[]
        for i in range(1,25):
            keep=np.arange(25)!=i;model,_=solve(x[keep],y[keep],penalty)
            _,choice,_=old.predict({**model,'global_gains':gains},x[i]);choices.append(choice)
            fold_states.append(model)
        folder=path/f'cv_penalty_{penalty:g}';folder.mkdir()
        np.savez_compressed(folder/'closed_fold_models.npz',**{k:np.stack([m[k] for m in fold_states]) for k in fold_states[0]})
        grid.append(dict(penalty=penalty,leave_one_out_complete_successes=int(sum(y[i,choices[i-1]] for i in range(1,25))),
            leave_one_out_invalid=int(sum(not valid[i,choices[i-1]] for i in range(1,25))),choices=choices))
    selected=max(grid,key=lambda v:(v['leave_one_out_invalid']==0,v['leave_one_out_complete_successes'],-v['leave_one_out_invalid'],v['penalty']))
    model,history=solve(x,y,selected['penalty'],path/'checkpoints');model['global_gains']=gains
    np.savez_compressed(path/'model.npz',**model)
    np.savez_compressed(path/'offline_training_state.npz',sensors=x,complete_success_labels=y,physical_valid=valid,**model)
    choices=[old.predict(model,v)[1] for v in x]
    report=dict(seed=257,updates=UPDATES,learning_rate=.01,grid=grid,chosen_penalty=selected['penalty'],history=history,
        original_R134_ridge100_leave_one_out_successes=14,leave_one_out_not_qualification=True,
        known_success_set_selection_count=int(sum(y[i,c] for i,c in enumerate(choices))),known_choices=choices,
        objective='-log probability mass of any complete valid success program + fixed L2; not squared four-label regression',
        unchanged_linear_50x4_capacity=True,no_inference_training_context_library=True,
        deterministic_full_batch_no_sampling=True,new_adam_complete_state_saved=True,training_cases=[None,*old.local.prior.program.TRAIN])
    old.local.write_json(path/'training_closed.json',report);return path/'model.npz',report

def init_worker(file):
    global MODELS
    old.local.init_worker();MODELS={}
    for name,path in [('candidate',Path(file)),('baseline',old.OUTPUT/'training/model.npz')]:
        with np.load(path,allow_pickle=False) as z:MODELS[name]={k:z[k].copy() for k in z.files}

def evaluate(job):
    case,name,directory=job
    old.MODEL=MODELS[name]
    row=old.evaluate((case,False,directory))
    dest=Path(directory);choice=row['global_program_choice']
    # References below are audit-only, after the full controller episode ended.
    source=old.source.OUTPUT/f'program_{choice:02d}'/f'case_{case}'
    saved=json.loads((source/'result.json').read_text())
    assert row['initial_hash']==saved['initial_hash'] and row['peaks']==saved['peaks']
    with np.load(dest/'trajectory.npz',allow_pickle=False) as a,np.load(source/'trajectory.npz',allow_pickle=False) as b:
        exact={k:bool(np.array_equal(a[k],b[k])) for k in a.files};assert all(exact.values()),exact
    if name=='baseline':
        with np.load(old.OUTPUT/'candidate'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as b,np.load(dest/'trajectory.npz',allow_pickle=False) as a:
            assert all(np.array_equal(a[k],b[k]) for k in a.files)
    row.update(model_group=name,selected_fixed_R133_program_bitwise_equal=exact,R134_baseline_exact=name=='baseline')
    old.local.write_json(dest/'result.json',row);return row

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true');p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for name in [Path(__file__).name,'test_getup_set_decision_r157.py','train_getup_success_selector_r134.py','train_getup_local_hip_r130.py',
        'train_getup_history_program_r122.py','probe_getup_sensor_history_r121.py','train_getup_program_r113.py',
        'getup_reference_env_r100.py','getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py']:
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    file,training=train(args.output/'training')
    old.local.write_json(args.output/'contract.json',dict(seed=257,original_episode_initialization_seed=234,smoke=args.smoke,workers=args.workers,
        method='Success-set likelihood over unchanged four global programs and linear native50 coefficients',
        hypothesis='R134 seven complete failures all select program zero while other complete valid fixed programs exist. Test a set-decision objective rather than increasing capacity or the old MSE budget. This does not prove the loss is the only cause or that unseen performance improves.',
        no_case_metadata_no_lookup_no_root_truth_no_future_sensors=True,controller_inputs='Actual initial native50; original current feedback after selection',
        nominal_actual_scalar_extra_exact_zero=True,home_original=True,full_controls=2279,control_hz=50,physics_hz=500,
        correction_limit_rad=.18,entry_deadline_s=12,strict_tail_s=30,physics_rewards_acceptance_unchanged=True,
        automatic_expanded_or_qualification=False,independent_qualification_never_loaded=True,no_mid_episode_root_reset=True,
        hashes={str(q):digest(q) for q in [file,old.OUTPUT/'training/model.npz',old.OUTPUT/'training/offline_training_state.npz',
            old.source.OUTPUT/'frozen_programs.npz',old.local.prior.program.SCENE,old.local.prior.program.STAND,old.local.prior.program.REFERENCE,*sources.iterdir()]}))
    cases=[None,769002,773004] if args.smoke else [None,*old.local.prior.program.TRAIN];reports={}
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(file,)) as pool:
        for name in ['baseline','candidate']:
            rows=list(pool.map(evaluate,[(case,name,str(args.output/name/f'case_{case}')) for case in cases]))
            reports[name]=old.local.aggregate(rows);old.local.write_json(args.output/name/'results.json',reports[name])
            print('R157_FULL',name,reports[name]['successes'],reports[name]['physical_failures'],flush=True)
    a=reports['candidate'];b=reports['baseline'];assert [v['initial_hash'] for v in a['rows']]==[v['initial_hash'] for v in b['rows']]
    gate=not args.smoke and a['nominal_success'] and a['successes']>=22 and a['successes']>b['successes'] and not a['physical_failures']
    old.local.write_json(args.output/'results.json',dict(smoke=args.smoke,reports=reports,training=training,original_development_gate=bool(gate),
        expanded_development_run=False,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False))
    print('R157_TERMINAL',bool(gate),flush=True)

if __name__=='__main__':main()
