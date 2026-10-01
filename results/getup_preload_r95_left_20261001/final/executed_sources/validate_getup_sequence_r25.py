"""Audit exact atlas labels then independently validate the frozen radius.

No hyperparameter changes after R25's 40-start selection. Original scene,
50 Hz targets, physical limits and success conditions remain unchanged.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics import train_getup_sequence_r24 as seq
from diagnostics.getup_independent_native import digest
from diagnostics.search_getup_rescue_r22 import BOUNDS


def audit_atlas(experiment):
    r=json.loads((experiment/'results.json').read_text())
    if not r.get('complete'):raise RuntimeError('atlas fitting/evaluation incomplete')
    if r['selected_radius']!=seq.select_radius(r['selection_40']['summary']):
        raise RuntimeError('radius not selected by original validation rule')
    with np.load(experiment/'trajectory_library.npz',allow_pickle=False) as a:library={k:a[k] for k in a.files}
    if library['obs'].shape!=(len(library['seed']),50) or library['knots'].shape!=(len(library['seed']),2,8):
        raise RuntimeError('invalid atlas shape')
    if not np.isfinite(library['obs']).all() or not np.isfinite(library['knots']).all():
        raise RuntimeError('nonfinite atlas')
    if np.any(np.abs(library['knots'])>BOUNDS+1e-8):raise RuntimeError('out of range teacher')
    if len(np.unique(library['seed']))!=len(library['seed']):raise RuntimeError('duplicate atlas seed')
    rows=json.loads((experiment/'transfer_dataset.json').read_text())
    for row in rows:
        ix=int(np.flatnonzero(library['seed']==row['seed'])[0])
        np.testing.assert_array_equal(library['obs'][ix],row['baseline']['initial_obs'])
        if row['anchor_kind']=='verified_rescue':
            if row['baseline']['success'] or not row['selected_long']['success'] or not row['selected_long']['success_4s']:
                raise RuntimeError('incorrect rescue label')
            if row['selected_long']['initial_hash']!=row['baseline']['initial_hash']:raise RuntimeError('teacher initial state mismatch')
            if not library['rescue'][ix]:raise RuntimeError('lost rescue label')
            np.testing.assert_array_equal(library['knots'][ix],row['selected_teacher']['knots'])
        else:
            if library['rescue'][ix] or np.any(library['knots'][ix]):raise RuntimeError('abstention/home mislabeled as rescue')
    report=dict(anchor_count=len(library['seed']),rescue_anchors=int(library['rescue'].sum()),
        selected_radius=r['selected_radius'],library_sha256=digest(experiment/'trajectory_library.npz'),
        finite_and_bounded=True,transfer_labels_verified=True,simulation_only=True,hardware_readiness=False)
    if report['library_sha256']!=r['library_sha256']:raise RuntimeError('library changed since selection')
    return report


def init_worker(contract,library,tilt):
    seq.init_worker(contract,library)
    seq._context[0].tilt_max=tilt


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    audit=audit_atlas(args.experiment)
    (args.output/'atlas_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('ATLAS AUDIT:',json.dumps(audit),flush=True)
    c=json.loads((args.experiment/'controller_contract.json').read_text());radius=audit['selected_radius']
    report=dict(audit=audit,selected_radius_frozen=True,physics_unchanged=True,
        success_gate_unchanged=True,paired_initial_states_verified=True,auto_resets=0,
        simulation_only=True,hardware_readiness=False,default_controller_replaced=False,
        stage='near-standing only, NOT fallen get-up',
        hashes={str(f):digest(f) for f in (Path(__file__),args.experiment/'results.json',args.experiment/'trajectory_library.npz')})
    output=args.output/'results.json'
    for name,tilt,seed_base,count,steps in [('broad_4s',.55,440000,200,200),
            ('broad_30s',.55,450000,50,1500),('mild_4s',.25,460000,100,200)]:
        seeds=list(range(seed_base,seed_base+count))
        if set(seeds)&set(c['training_seed_list']):raise RuntimeError('evaluation seed overlap')
        with ProcessPoolExecutor(max_workers=4,initializer=init_worker,
                initargs=(str(args.experiment/'controller_contract.json'),str(args.experiment/'trajectory_library.npz'),tilt)) as pool:
            block=seq.paired_block(pool,sorted(set([0.,radius])),seeds,steps)
        block.update(tilt_max_rad=tilt,seed_base=seed_base);report[name]=block
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print('FROZEN ATLAS:',name,json.dumps(block['summary']),flush=True)
    report['complete']=True
    report['actual_control_steps']=sum(v['actual_control_steps'] for v in report.values() if isinstance(v,dict) and 'actual_control_steps' in v)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
