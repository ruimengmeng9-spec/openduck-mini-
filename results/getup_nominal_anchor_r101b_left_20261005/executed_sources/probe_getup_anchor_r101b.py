"""R101b fixes scalar/batch numerical parity, unchanged physical constraints."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics.probe_getup_anchor_r101 import nominal_observations,trial,ROOT,MODEL,REFERENCE
from diagnostics.getup_reference_env_r100 import TRAIN,numpy_action
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);args=p.parse_args()
    args.output.mkdir(exist_ok=False)
    observations=nominal_observations()
    with np.load(MODEL) as data:weights={k:data[k].copy() for k in data.files}
    # Same matrix-vector calculation as the runtime trial, not matrix-matrix.
    anchors=np.stack([numpy_action(weights,row) for row in observations])
    np.savez_compressed(args.output/'nominal_anchor.npz',observations=observations,actions=anchors)
    settings=[['anchored',1.],['anchored',4.],['anchored',-1.],['raw',.25]]
    contract=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        development_only=True,hypothesis='Remove nominal output drift before judging state correction',
        original_failed_run_preserved=True,anchor_inference='scalar vector matmul identical to runtime',
        model_sha256=digest(MODEL),reference_sha256=digest(REFERENCE),cases=[None,*TRAIN],
        reference_states_never_injected=True,combined_residual_cap_rad=.18,
        physics_and_acceptance_unchanged=True,frozen_settings=settings)
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'probe_getup_anchor_r101.py','getup_reference_env_r100.py',
                 'inspect_anchor_numerics_r101.py','search_getup_reference_feedback_r64.py',
                 'getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    reports={}
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn')) as pool:
        for mode,scale in settings:
            name=mode+'_'+str(scale);directory=args.output/name;directory.mkdir()
            rows=[]
            for result in pool.map(trial,[(seed,mode,scale,anchors,str(directory)) for seed in [None,*TRAIN]]):
                rows.append(result);print('R101B_CASE',name,result['case_seed'],result['success'],result['valid'],flush=True)
            report=dict(rows=rows,successes=sum(r['success'] for r in rows if r['case_seed'] is not None),
                nominal_success=rows[0]['success'],physical_failures=sum(not r['valid'] for r in rows),mode=mode,scale=scale)
            if mode=='anchored':
                with np.load(directory/'case_None.npz') as d:
                    np.testing.assert_array_equal(d['normalized_residual'],np.zeros((len(observations),10)))
                    report['nominal_residual_exact_zero']=True
            (directory/'results.json').write_text(json.dumps(report,indent=2));reports[name]=report
            print('R101B_ANCHOR_RESULT',name,report['successes'],report['nominal_success'],report['physical_failures'],flush=True)
            (args.output/'partial_results.json').write_text(json.dumps(reports,indent=2))
    (args.output/'results.json').write_text(json.dumps(dict(reports=reports,full_task_completed=False,
        simulation_only=True,hardware_readiness=False,independent_qualification_not_run=True),indent=2))
    print('R101B_TERMINAL',flush=True)


if __name__=='__main__':main()
