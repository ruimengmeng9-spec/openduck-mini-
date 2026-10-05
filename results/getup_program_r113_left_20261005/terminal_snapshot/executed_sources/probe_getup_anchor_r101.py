"""Development-only causal nominal anchoring, never inject reference states."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np

from diagnostics.getup_reference_env_r100 import ReferenceEpisode,TRAIN,numpy_action
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck')
SCENE=ROOT/'training/getup_decomposed_r4/model/scene.xml'
STAND=ROOT/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
REFERENCE=ROOT/'outputs/getup_path_audit_r99_20261005/frozen_path.npz'
MODEL=ROOT/'outputs/getup_reference_residual_r100_left_20261005/training/final.npz'


def nominal_observations():
    env=ReferenceEpisode(str(SCENE),str(STAND),REFERENCE,200,True);env.reset(None)
    observations=[]
    while True:
        observations.append(env.observe())
        row=env.step(np.zeros(10),auto_reset=False)
        if row[2]:break
    assert row[5]['valid'] and env.tail*.02>=30.
    return np.stack(observations)


def trial(job):
    seed,mode,scale,anchors,output=job
    with np.load(MODEL) as data:weights={k:data[k].copy() for k in data.files}
    env=ReferenceEpisode(str(SCENE),str(STAND),REFERENCE,200,True);env.reset(seed)
    trace=[];actions=[]
    while True:
        observation=env.observe();action=numpy_action(weights,observation)
        if mode=='anchored':action=action-anchors[env.controls]
        action=np.clip(scale*action,-1.,1.)
        actions.append(action)
        row=env.step(action,auto_reset=False)
        trace.append((env.sim.data.time,env.sim.data.qpos.copy(),env.sim.data.qvel.copy(),env.sim.prev.copy(),env.tail>0))
        if row[2]:break
    result=row[5]
    result.update(mode=mode,scale=scale,success=bool(result['valid'] and result['controls']==len(env.targets)
        and result['entry_time_s'] is not None and result['entry_time_s']<=12. and result['strict_tail_s']>=30.-1e-8),
        reference_states_injected=False)
    np.savez_compressed(Path(output)/f'case_{seed}.npz',time=[r[0] for r in trace],qpos=[r[1] for r in trace],
        qvel=[r[2] for r in trace],applied=[r[3] for r in trace],strict=[r[4] for r in trace],normalized_residual=actions)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    observations=nominal_observations()
    with np.load(MODEL) as data:weights={k:data[k].copy() for k in data.files}
    anchors=numpy_action(weights,observations)
    np.savez_compressed(args.output/'nominal_anchor.npz',observations=observations,actions=anchors)
    contract=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,development_only=True,
        hypothesis='Subtracting nominal network output removes trajectory-independent drift; counterfactual gain tests, not a success claim',
        model_sha256=digest(MODEL),reference_sha256=digest(REFERENCE),source_sha256=digest(Path(__file__)),
        cases=[None,*TRAIN],controller_only=True,reference_states_never_injected=True,
        combined_residual_cap_rad=.18,physics_and_acceptance_unchanged=True,
        frozen_settings=[['anchored',1.],['anchored',4.],['anchored',-1.],['raw',.25]])
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'getup_reference_env_r100.py','search_getup_reference_feedback_r64.py',
                 'getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    reports={}
    # One persistent pool avoids four separate backend shutdown waits.
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for mode,scale in contract['frozen_settings']:
            name=mode+'_'+str(scale);directory=args.output/name;directory.mkdir()
            jobs=[(seed,mode,scale,anchors,str(directory)) for seed in [None,*TRAIN]]
            rows=[]
            for result in pool.map(trial,jobs):
                rows.append(result);print('R101_CASE',name,result['case_seed'],result['success'],result['valid'],flush=True)
            report=dict(rows=rows,successes=sum(r['success'] for r in rows if r['case_seed'] is not None),
                nominal_success=rows[0]['success'],physical_failures=sum(not r['valid'] for r in rows),
                mode=mode,scale=scale)
            if mode=='anchored':
                with np.load(directory/'case_None.npz') as d:
                    np.testing.assert_array_equal(d['normalized_residual'],np.zeros((len(observations),10)))
            (directory/'results.json').write_text(json.dumps(report,indent=2));reports[name]=report
            print('R101_ANCHOR_RESULT',name,report['successes'],report['nominal_success'],report['physical_failures'],flush=True)
            (args.output/'partial_results.json').write_text(json.dumps(reports,indent=2))
    (args.output/'results.json').write_text(json.dumps(dict(reports=reports,full_task_completed=False,
        simulation_only=True,hardware_readiness=False,independent_qualification_not_run=True),indent=2))
    print('R101_TERMINAL',flush=True)


if __name__=='__main__':main()
