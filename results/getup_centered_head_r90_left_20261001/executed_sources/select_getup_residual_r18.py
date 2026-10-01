"""Select bounded residual checkpoints by actual success, not shaping return."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil

from flax import serialization
import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode
from diagnostics.getup_macro_curriculum_r15 import advance_macro
from diagnostics.train_getup_residual_r18 import export_actor
from diagnostics.getup_independent_native import digest


def evaluate(job):
    model,contract_path,seed=job
    c=json.loads(Path(contract_path).read_text())
    e=NativeEpisode(c['scene_path'],
                    '/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                    c['starts_rad'],seed,c['tilt_max_rad'])
    initial_hash=hashlib.sha256(e.sim.data.qpos.tobytes()+e.sim.data.qvel.tobytes()).hexdigest()
    import onnxruntime as ort
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    actor=ort.InferenceSession(model,sess_options=options,providers=['CPUExecutionProvider'])
    repeat=c['policy_action_repeat']
    for _ in range(200):
        action=actor.run(None,{'obs':e.sim.observation()[None]})[0][0]
        _,_,done,_,_,info,_=advance_macro(e,action,repeat,.99**(1/repeat))
        if done:return dict(model=Path(model).stem,seed=seed,initial_hash=initial_hash,**info)
    raise RuntimeError('episode exceeded declared length')


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=8);p.add_argument('--stride',type=int,default=16)
    args=p.parse_args();out=args.experiment/'checkpoint_selection';out.mkdir(exist_ok=False)
    contract_path=args.experiment/'controller_contract.json';c=json.loads(contract_path.read_text())
    candidates=[('initial',args.experiment/'initial.msgpack',args.experiment/'initial.onnx')]
    for checkpoint in sorted(args.experiment.glob('checkpoint_*.msgpack')):
        if int(checkpoint.stem.split('_')[-1])%args.stride:continue
        model=out/f'{checkpoint.stem}.onnx'
        params=serialization.msgpack_restore(checkpoint.read_bytes())['actor']
        export_actor(params,model,c['home_rad'],c['lower_rad'],c['upper_rad'],c['residual_scale_rad'])
        candidates.append((checkpoint.stem,checkpoint,model))
    jobs=[(str(model),str(contract_path),220000+s) for _,_,model in candidates for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(evaluate,jobs,chunksize=1))
    for seed in range(220000,220000+args.seeds):
        if len({r['initial_hash'] for r in rows if r['seed']==seed})!=1:raise RuntimeError('paired state mismatch')
    scores=[]
    for name,params,model in candidates:
        group=[r for r in rows if r['model']==model.stem]
        scores.append(dict(name=name,successes=sum(r['success'] for r in group),runs=len(group),
                           full_duration_runs=sum(r['steps']==200 for r in group),safe_runs=sum(r['safe'] for r in group),
                           mean_return=float(np.mean([r['return_sum'] for r in group])),
                           params_sha256=digest(params),onnx_sha256=digest(model)))
    # Stable candidate ordering breaks ties: initial first, then earliest checkpoint.
    # Do not select a different motion merely for its higher shaping return.
    winner=max(scores,key=lambda r:(r['successes'],r['full_duration_runs'],r['safe_runs']))
    name,params,model=next(row for row in candidates if row[0]==winner['name'])
    selected=args.experiment/'selected_candidate';selected.mkdir(exist_ok=False)
    for source,destination in ((params,'final.msgpack'),(model,'final.onnx'),
                               (args.experiment/'initial.msgpack','initial.msgpack'),
                               (args.experiment/'initial.onnx','initial.onnx'),(contract_path,'controller_contract.json')):
        shutil.copyfile(source,selected/destination)
    report=dict(winner=winner,scores=scores,results=rows,validation_seed_base=220000,
                final_test_seed_base=230000,policy_action_repeat=c['policy_action_repeat'],
                selection_rule='success, full duration, physical checks; ties baseline/earliest',
                simulation_only=True,hardware_readiness=False,paired_initial_states_verified=True,
                script_sha256=digest(__file__),warning='Validation choice is not independent test performance.')
    (out/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (selected/'selection_contract.json').write_text(json.dumps(dict(winner=winner,
        source_experiment=str(args.experiment),validation_seed_base=220000,final_test_seed_base=230000,
        selection_sha256=digest(out/'selection.json')),indent=2),encoding='utf-8')
    print('RESIDUAL CHECKPOINTS:',json.dumps(scores),flush=True)
    print('SELECTED:',json.dumps(winner),flush=True)


if __name__=='__main__':main()
