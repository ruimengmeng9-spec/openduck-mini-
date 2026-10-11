"""Set-valued complete-success supervision over four global controllers.

Inference takes actual native50 only, with no stored context library. It emits
one of four fixed global gain vectors, not a case-specific teacher recipe.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import probe_getup_complementary_feedback_r133 as source
from diagnostics import train_getup_local_hip_r130 as local
from diagnostics.getup_independent_native import digest

ROOT=local.ROOT
OUTPUT=ROOT/'outputs/getup_success_selector_r134_left_20261006'
RIDGES=(.1,1.,10.,100.)
MODEL=None


def predict(model,sensors):
    sensors=np.asarray(sensors,dtype=float)
    if sensors.shape!=(50,) or not np.isfinite(sensors).all():raise ValueError('Only current initial native50 sensors allowed')
    logits=(sensors-model['mean'])/model['std']@model['weight']+model['bias']
    choice=int(np.argmax(logits));gains=model['global_gains'][choice].copy()
    assert gains.shape==(6,) and np.isfinite(gains).all() and np.abs(gains).max()<=2.
    return gains,choice,logits


def fit(x,y,ridge):
    mean=x.mean(0);std=np.maximum(x.std(0),1e-4);z=(x-mean)/std;bias=y.mean(0)
    weights=z.T@np.linalg.solve(z@z.T+ridge*np.eye(len(x)),y-bias)
    return dict(mean=mean,std=std,weight=weights,bias=bias)


def train(output):
    terminal=json.loads((source.OUTPUT/'results.json').read_text())
    assert not terminal['smoke'] and terminal['complete_union_count']>=22
    manifest=json.loads((source.OUTPUT/'manifest.json').read_text())
    cases=[r['case_seed'] for r in manifest];assert cases==[None,*local.prior.program.TRAIN]
    with np.load(source.OUTPUT/'offline_success_sets.npz',allow_pickle=False) as z:y=z['success'].astype(float);valid=z['valid'].copy()
    with np.load(source.OUTPUT/'frozen_programs.npz',allow_pickle=False) as z:gains=z['gains'].copy()
    assert y.shape==(25,4) and np.all(np.any(y.astype(bool)&valid,axis=1))
    # Actual histories are used only offline to fit fixed coefficient matrices.
    x=[]
    for case in cases:
        with np.load(source.OUTPUT/'program_00'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as z:x.append(z['preparation_sensors'][-1])
    x=np.stack(x).astype(float);grid=[]
    for ridge in RIDGES:
        chosen=[]
        for index in range(1,25):
            keep=np.arange(25)!=index;model={**fit(x[keep],y[keep],ridge),'global_gains':gains}
            _,choice,_=predict(model,x[index]);chosen.append(choice)
        successes=int(sum(y[i,chosen[i-1]] for i in range(1,25)))
        failures=int(sum(not valid[i,chosen[i-1]] for i in range(1,25)))
        grid.append(dict(ridge=ridge,leave_one_out_complete_successes=successes,leave_one_out_invalid=failures,choices=chosen))
    best=max(grid,key=lambda r:(r['leave_one_out_invalid']==0,r['leave_one_out_complete_successes'],-r['leave_one_out_invalid'],r['ridge']))
    model={**fit(x,y,best['ridge']),'global_gains':gains};output.mkdir(exist_ok=False)
    np.savez_compressed(output/'model.npz',**model)
    np.savez_compressed(output/'offline_training_state.npz',sensors=x,complete_success_labels=y,physical_valid=valid,**model)
    report=dict(grid=grid,chosen_ridge=best['ridge'],deterministic_closed_form=True,no_optimizer_or_rng_sampling=True,
        initialization_seed=234,inference_model_has_no_training_contexts=True,
        objective='Set-valued complete valid success supervision, not single teacher recipe MSE',
        leave_one_out_not_qualification=True,training_cases=cases)
    local.write_json(output/'training.json',report);return output/'model.npz',report


def init_worker(file):
    global MODEL
    local.init_worker()
    with np.load(file,allow_pickle=False) as z:MODEL={k:z[k].copy() for k in z.files}


def evaluate(job):
    case,baseline,directory=job
    assert case in {None,*local.prior.program.TRAIN}
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,234,True)
    env.reset(case);sim=env.sim
    gains,choice,logits=predict(MODEL,env.preparation_sensors[-1])
    if baseline:gains=np.zeros(6)
    actuator_ids=np.array([sim.model.actuator(n).id for n in local.JOINTS])
    sensor_ids=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in local.JOINTS])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    original=sim.step_target;extra=[]
    def controlled(target):
        k=env.controls;feedback=local.local_feedback(env.observe(),local.NOMINAL[k],sensor_ids,gains) if k<529 else np.zeros(3)
        adjusted=local.merge_target(target,env.targets[k],feedback,actuator_ids,sim.lower,sim.upper)
        assert np.abs(adjusted[actuator_ids]-env.targets[k][actuator_ids]).max()<=.18+1e-12
        extra.append(feedback.copy());return original(adjusted)
    sim.step_target=controlled;records=[]
    try:
        while True:
            obs=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    finally:sim.step_target=original
    row=step[5];row.update(success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),
        baseline=baseline,global_program_choice=choice,initial_sensor_logits=logits.tolist(),gains=gains.tolist(),no_case_metadata_in_controller=True)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],
        qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],
        local_hip_extra_rad=extra,preparation_sensors=env.preparation_sensors)
    old=local.prior.OUTPUT/'snapshot'/f'case_{case}';assert row['initial_hash']==json.loads((old/'result.json').read_text())['initial_hash']
    if baseline or case is None:
        with np.load(old/'trajectory.npz',allow_pickle=False) as z:equal={k:bool(np.array_equal(v,z[k])) for k,v in arrays.items() if k not in ('local_hip_extra_rad','preparation_sensors')}
        assert all(equal.values());row['original_complete_trace_bitwise_equal']=equal
    path=Path(directory);path.mkdir(parents=True,exist_ok=False);np.savez_compressed(path/'trajectory.npz',**arrays);local.write_json(path/'result.json',row);return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve() and 1<=args.workers<=6
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_success_selector_r134.py','launch_getup_success_selector_r134.py','probe_getup_complementary_feedback_r133.py','train_getup_local_hip_r130.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    file,training=train(args.output/'training')
    local.write_json(args.output/'contract.json',dict(smoke=args.smoke,workers=args.workers,seed=234,
        controller_input='Actual initial native50 sensor vector; then original current sensor feedback',
        no_case_lookup_or_privileged_truth=True,no_unseen_qualification=True,no_mid_episode_root_reset=True,
        physics_rewards_acceptance_unchanged=True,combined_correction_rad=.18,full_controls=2279,control_hz=50,physics_hz=500,
        entry_deadline_s=12,strict_tail_s=30,hardware_readiness=False,training_method='Regularized linear multi-success predictor over four globally fixed gain vectors',
        hashes={str(p):digest(p) for p in [file,source.OUTPUT/'results.json',source.OUTPUT/'frozen_programs.npz',local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE,*sources.iterdir()]}))
    cases=[None,769000] if args.smoke else [None,*local.prior.program.TRAIN];reports={}
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(file,)) as pool:
        for baseline,name in ((True,'baseline'),(False,'candidate')):
            rows=list(pool.map(evaluate,[(s,baseline,str(args.output/name/f'case_{s}')) for s in cases]))
            reports[name]=local.aggregate(rows);local.write_json(args.output/name/'results.json',reports[name]);print('R134_COMPLETE',name,reports[name]['successes'],reports[name]['physical_failures'],flush=True)
    a=reports['candidate'];b=reports['baseline'];assert [r['initial_hash'] for r in a['rows']]==[r['initial_hash'] for r in b['rows']]
    gate=not args.smoke and a['nominal_success'] and a['successes']>=22 and a['successes']>b['successes'] and a['physical_failures']==0
    local.write_json(args.output/'results.json',dict(smoke=args.smoke,reports=reports,training=training,original_development_gate=bool(gate),expanded_run=False,
        independent_qualification_run=False,hardware_readiness=False,full_task_completed=False));print('R134_TERMINAL',bool(gate),flush=True)


if __name__=='__main__':main()
