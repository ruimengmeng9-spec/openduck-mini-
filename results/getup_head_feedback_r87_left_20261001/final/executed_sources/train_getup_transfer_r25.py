"""Expand a trajectory atlas by native counterfactual teacher transfer.

Every new training start tests home and five pre-existing verified sequences.
Only new rescues with an independent 30-second replay become rescue anchors.
Unsupported starts are abstention anchors, never mislabeled successful home.
Selection and held-out seeds remain disjoint from this counterfactual dataset.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from diagnostics import search_getup_rescue_r22 as rescue
from diagnostics import train_getup_sequence_r24 as seq
from diagnostics.getup_independent_native import digest
from diagnostics.distill_getup_rescue_r23 import physical_contract


_teachers=None


def init_worker(contract,teachers):
    global _teachers
    rescue.init_worker(contract,.55);_teachers=teachers


def transfer(seed):
    baseline,_=rescue.replay(seed,np.zeros((2,8)))
    controls=baseline['steps'];trials=[];selected=None;long=None
    if not baseline['success']:
        for teacher in _teachers:
            row,_=rescue.replay(seed,np.asarray(teacher['knots']))
            controls+=row['steps']
            if row['initial_hash']!=baseline['initial_hash']:raise RuntimeError('transfer start mismatch')
            trials.append(dict(teacher_seed=teacher['seed'],row=row))
        successful=sorted([r for r in trials if r['row']['success']],key=lambda r:r['row']['return_sum'],reverse=True)
        for r in successful:
            teacher=next(t for t in _teachers if t['seed']==r['teacher_seed'])
            checked,_=rescue.replay(seed,np.asarray(teacher['knots']),steps=1500)
            controls+=checked['steps']
            if checked['initial_hash']!=baseline['initial_hash']:raise RuntimeError('long transfer start mismatch')
            if checked['success'] and checked['success_4s']:
                selected=teacher;long=checked;break
    label='home_success' if baseline['success'] else 'verified_rescue' if selected else 'unsupported_abstention'
    return dict(seed=seed,baseline=baseline,trials=trials,selected_teacher=selected,
        selected_long=long,anchor_kind=label,actual_control_steps=controls)


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--previous',type=Path,required=True);p.add_argument('--search',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--training-seeds',type=int,default=100)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    teachers=[]
    for folder in args.search:
        for r in json.loads((folder/'results.json').read_text())['results']:
            if r['selected']['success'] and r['selected_long']['success']:
                teachers.append(dict(seed=r['seed'],knots=r['knots']))
    if len(teachers)!=5:raise RuntimeError('expected frozen five-teacher transfer set')
    report=dict(teachers=teachers,training_seed_base=400000,training_seed_count=args.training_seeds,
        method='counterfactual whole-sequence teacher transfer; NOT PPO or neural actor training',
        default_controller_replaced=False,simulation_only=True,hardware_readiness=False,
        stage='near-standing ONLY, NOT full fallen get-up',physics_unchanged=True,success_gate_unchanged=True,
        auto_resets=0,paired_initial_states_verified=True,
        hashes={str(f):digest(f) for f in (Path(__file__),args.contract,args.previous/'trajectory_library.npz')})
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(str(args.contract),teachers)) as pool:
        rows=[]
        for row in pool.map(transfer,range(400000,400000+args.training_seeds)):
            rows.append(row)
            (args.output/'transfer_dataset.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print('TRANSFER:',row['seed'],row['anchor_kind'],'count',len(rows),flush=True)
    with np.load(args.previous/'trajectory_library.npz',allow_pickle=False) as a:
        old={k:a[k] for k in a.files}
    extra=dict(seed=np.array([r['seed'] for r in rows]),obs=np.array([r['baseline']['initial_obs'] for r in rows]),
        rescue=np.array([r['anchor_kind']=='verified_rescue' for r in rows],dtype=bool),
        knots=np.array([r['selected_teacher']['knots'] if r['selected_teacher'] else np.zeros((2,8)) for r in rows]))
    library={k:np.concatenate([old[k],extra[k]]) for k in old}
    if len(np.unique(library['seed']))!=len(library['seed']):raise RuntimeError('duplicate atlas seed')
    np.savez_compressed(args.output/'trajectory_library.npz',**library)
    c=physical_contract(json.loads(args.contract.read_text()))
    c.update(stage=report['stage'],training_method=report['method'],feature_indices=seq.FEATURES.tolist(),
        feature_scales=seq.SCALES.tolist(),rescue_vs_home_distance_margin=.8,
        knots_s=[.2,.6,1.2],training_seed_list=library['seed'].tolist(),
        abstention_anchors_are_success_labels=False,default_controller_replaced=False)
    contract=args.output/'controller_contract.json';contract.write_text(json.dumps(c,indent=2),encoding='utf-8')
    report['transfer_summary']={kind:sum(r['anchor_kind']==kind for r in rows) for kind in ('home_success','verified_rescue','unsupported_abstention')}
    report['transfer_control_steps']=sum(r['actual_control_steps'] for r in rows)
    print('TRANSFER SUMMARY:',json.dumps(report['transfer_summary']),flush=True)
    output=args.output/'results.json';output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    training=library['seed'].tolist();selection=list(range(410000,410040));short=list(range(420000,420100));long=list(range(430000,430020))
    if set(training)&set(selection+short+long):raise RuntimeError('validation seed contamination')
    with ProcessPoolExecutor(max_workers=4,initializer=seq.init_worker,
            initargs=(str(contract),str(args.output/'trajectory_library.npz'))) as pool:
        report['selection_40']=seq.paired_block(pool,[0.,.15,.3,.6,1.2],selection,200)
        selected=seq.select_radius(report['selection_40']['summary']);report['selected_radius']=selected
        print('ATLAS SELECTION:',json.dumps(report['selection_40']['summary']),'CHOSEN',selected,flush=True)
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        candidates=sorted(set([0.,selected,.3]));report['predeclared_exploratory_radius']=.3
        for name,seeds,steps in [('demonstration_4s',training,200),('independent_100',short,200),('independent_30s',long,1500)]:
            report[name]=seq.paired_block(pool,candidates,seeds,steps)
            output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('ATLAS POLICY:',name,json.dumps(report[name]['summary']),flush=True)
    report['complete']=True
    report['actual_control_steps']=report['transfer_control_steps']+sum(v['actual_control_steps'] for v in report.values() if isinstance(v,dict) and 'actual_control_steps' in v)
    report['library_sha256']=digest(args.output/'trajectory_library.npz')
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
