"""Select a stage-1 checkpoint on validation seeds, separate from final test."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

from flax import serialization
import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode
from diagnostics.getup_independent_native import digest
from diagnostics.train_getup_native_ppo_v2 import export_actor


def evaluate(job):
    model,contract_path,seed,tilt=job
    contract=json.loads(Path(contract_path).read_text())
    e=NativeEpisode(contract['scene_path'],'/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
                    contract['starts_rad'],seed,tilt)
    import onnxruntime as ort
    options=ort.SessionOptions(); options.intra_op_num_threads=options.inter_op_num_threads=1
    session=ort.InferenceSession(model,sess_options=options,providers=['CPUExecutionProvider'])
    for _ in range(200):
        obs=e.sim.observation()
        action=session.run(None,{'obs':obs[None]})[0][0]
        _,_,done,_,_,info=e.step(action)
        if done: return dict(model=Path(model).stem,seed=seed,**info)
    raise RuntimeError('declared episode length exceeded')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=8)
    p.add_argument('--stride',type=int,default=32)
    args=p.parse_args()
    out=args.experiment/'checkpoint_selection'; out.mkdir(exist_ok=False)
    contract_path=args.experiment/'controller_contract.json'
    contract=json.loads(contract_path.read_text())
    candidates=[('initial',args.experiment/'initial.msgpack',args.experiment/'initial.onnx')]
    for checkpoint in sorted(args.experiment.glob('checkpoint_*.msgpack')):
        iteration=int(checkpoint.stem.split('_')[-1])
        if iteration%args.stride: continue
        model=out/f'{checkpoint.stem}.onnx'
        params=serialization.msgpack_restore(checkpoint.read_bytes())['actor']
        export_actor(params,model)
        candidates.append((checkpoint.stem,checkpoint,model))
    jobs=[(str(model),str(contract_path),88000+seed,contract['tilt_max_rad'])
          for _,_,model in candidates for seed in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(evaluate,jobs,chunksize=1))
    scores=[]
    for name,params,model in candidates:
        group=[r for r in rows if r['model']==model.stem]
        scores.append(dict(name=name,successes=sum(r['success'] for r in group),runs=len(group),
                           full_duration_runs=sum(r['steps']==200 for r in group),
                           safe_runs=sum(r['safe'] for r in group),
                           mean_return=float(np.mean([r['return_sum'] for r in group])),
                           params_sha256=digest(params),onnx_sha256=digest(model)))
    # Success is primary; survival precedes shaping reward. Baseline is eligible.
    winner=max(scores,key=lambda r:(r['successes'],r['full_duration_runs'],r['safe_runs'],r['mean_return']))
    selected=args.experiment/'selected_candidate'; selected.mkdir(exist_ok=False)
    name,params,model=next(c for c in candidates if c[0]==winner['name'])
    shutil.copyfile(params,selected/'final.msgpack'); shutil.copyfile(model,selected/'final.onnx')
    shutil.copyfile(args.experiment/'initial.msgpack',selected/'initial.msgpack')
    shutil.copyfile(args.experiment/'initial.onnx',selected/'initial.onnx')
    shutil.copyfile(contract_path,selected/'controller_contract.json')
    report=dict(winner=winner,scores=scores,results=rows,validation_seed_base=88000,
                final_test_seed_base=95000,stage=contract['stage'],simulation_only=True,
                hardware_readiness=False,script_sha256=digest(__file__),
                warning='Selection success is not held-out performance or full fallen recovery.')
    (out/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (selected/'selection_contract.json').write_text(json.dumps(dict(winner=winner,
        source_experiment=str(args.experiment),validation_seed_base=88000,
        final_test_seed_base=95000,selection_sha256=digest(out/'selection.json')),indent=2),encoding='utf-8')
    print('VALIDATION CHECKPOINTS:',json.dumps(scores),flush=True)
    print('SELECTED:',json.dumps(winner),flush=True)


if __name__=='__main__': main()
