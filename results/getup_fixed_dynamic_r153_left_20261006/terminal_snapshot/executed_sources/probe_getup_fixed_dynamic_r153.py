"""Bounded full-path evaluation of the already fixed R151 smoke program.

No new parameter fitting, scaling, thresholds, seed lookup, or extra settling.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_jointwise_dynamic_r151 as run
from diagnostics.getup_independent_native import digest

OUTPUT=run.ROOT/'outputs/getup_fixed_dynamic_r153_left_20261006'
PROBE=run.PROBE.copy()

def exact(path,reference):
    with np.load(path/'trajectory.npz',allow_pickle=False) as x,np.load(reference/'trajectory.npz',allow_pickle=False) as y:
        equal={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
    assert all(equal.values()),equal
    a=json.loads((path/'result.json').read_text());b=json.loads((reference/'result.json').read_text())
    assert a['initial_hash']==b['initial_hash'] and a['peaks']==b['peaks']
    return dict(case_seed=a['case_seed'],fields=equal,initial_hash=a['initial_hash'],original_peaks_equal=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true')
    args=p.parse_args();assert args.output.parent.resolve()==(run.ROOT/'outputs').resolve()
    audit=json.loads((run.ROOT/'outputs/getup_jointwise_terminal_audit_r152_20261006/results.json').read_text())
    assert not np.any(audit['terminal_parameters']) and not audit['smoke_probe_in_CEM']
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for name in [Path(__file__).name,'launch_getup_fixed_dynamic_r153.py','train_getup_jointwise_dynamic_r151.py','test_getup_jointwise_dynamic_r151.py','train_getup_dynamic_gain_r147.py',
                 'train_getup_success_selector_r134.py','train_getup_local_hip_r130.py','train_getup_history_program_r122.py',
                 'probe_getup_sensor_history_r121.py','train_getup_program_r113.py','getup_reference_env_r100.py',
                 'getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py']:
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    shutil.copytree(run.OUTPUT/'frozen',args.output/'frozen')
    run.prior.local.write_json(args.output/'contract.json',dict(seed=251,smoke=args.smoke,parameters=PROBE.tolist(),
        method='One fixed prior diagonal smoke program versus frozen zero R151/R134; no new training or parameter selection',
        controls=2279,control_hz=50,physics_hz=500,entry_deadline_s=12,strict_tail_s=30,
        physical_limits_rewards_acceptance_unchanged=True,no_mid_episode_root_edits=True,
        no_new_qualification=True,no_case_metadata_controller=True,
        source_hashes={str(f):digest(f) for f in [*sources.iterdir(),*(args.output/'frozen').iterdir(),
            run.prior.local.prior.program.SCENE,run.prior.local.prior.program.STAND,run.prior.local.prior.program.REFERENCE]}))
    checks=[]
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=run.prior.init_worker,
        initargs=(args.output/'frozen/initial_selector.npz',)) as pool:
        zero=run.group(pool,np.zeros(6),[None,769000,773004],True,args.output/'zero_parity',True)
        candidate=run.group(pool,PROBE,[None,769002,773004],True,args.output/'probe_parity')
        for case in [None,769000,773004]:checks.append(exact(args.output/'zero_parity'/f'case_{case}',run.OUTPUT/'zero_parity'/f'case_{case}'))
        for case in [None,769002,773004]:checks.append(exact(args.output/'probe_parity'/f'case_{case}',run.OUTPUT/'nonzero_smoke'/f'case_{case}'))
        run.prior.local.write_json(args.output/'parity.json',checks)
        print('R153_SIX_COMPLETE_PARITY_PASS',flush=True)
        if args.smoke:
            run.prior.local.write_json(args.output/'results.json',dict(smoke=True,zero=zero,candidate=candidate,checks=checks,
                full_task_completed=False,hardware_readiness=False));return
        cases=[None,*run.prior.local.prior.program.TRAIN]
        candidate=run.group(pool,PROBE,cases,True,args.output/'candidate')
        baseline=run.group(pool,np.zeros(6),cases,True,args.output/'baseline',True)
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        baseline_checks=[exact(args.output/'baseline'/f'case_{c}',run.OUTPUT/'development_baseline'/f'case_{c}') for c in cases]
        rescued=[a['case_seed'] for a,b in zip(candidate['rows'],baseline['rows']) if a['case_seed'] is not None and a['success'] and not b['success']]
        regressed=[a['case_seed'] for a,b in zip(candidate['rows'],baseline['rows']) if a['case_seed'] is not None and b['success'] and not a['success']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
        run.prior.local.write_json(args.output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,
            baseline_checks=baseline_checks,rescued=rescued,regressed=regressed,parameters=PROBE.tolist(),
            original_development_gate=gate,expanded_development_run=False,independent_qualification_run=False,
            full_task_completed=False,hardware_readiness=False))
        print('R153_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],rescued,regressed,flush=True)

if __name__=='__main__':main()
