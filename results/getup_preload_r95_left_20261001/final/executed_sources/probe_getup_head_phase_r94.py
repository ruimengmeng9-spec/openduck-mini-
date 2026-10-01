"""R94 early head/leg relative phase fitting, not a shared whole-body clock.

R92 observed head loading differences before feedback differences. Test whether
relative command timing before that transition matters. No new target poses,
physical budget, online seed/outcome input or intermediate state restoration.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.getup_independent_native import DT,digest
from diagnostics.probe_getup_terminal_timing_r72 import timed_targets
from diagnostics.probe_getup_wait_library_r77 import TRAIN,save_trace
from diagnostics.train_getup_head_feedback_r87 import path
from diagnostics.train_getup_gyro_attenuation_r93 import run,BIAS
from diagnostics.search_getup_reference_feedback_r64 import rollout
from diagnostics.train_getup_prefix_pose_r73 import success

HELDOUT=tuple(range(794000,794040))
LEVELS=(0.,-.06,.06,-.12,.12)
WINDOW=(0.,.3,.9,1.2)


def warped_head_prefix(ck,head_ids,lead):
    lead=np.asarray(lead,dtype=float);head_ids=np.asarray(head_ids)
    if lead.shape!=(2,) or not np.isfinite(lead).all() or abs(lead).max()>.12+1e-10:
        raise ValueError('Need two finite head timing leads within 0.12 seconds')
    if head_ids.shape!=(2,) or len(set(head_ids.tolist()))!=2:
        raise ValueError('Need distinct neck and head-pitch actuator IDs')
    source=ck['prefix_targets'];phases=ck['prefix_phases']
    count=round(WINDOW[-1]/DT)+1
    if len(source)<count or np.any(phases[:count]!=0):
        raise ValueError('Intervention must stay in the original initial phase')
    value=source.copy()
    if not np.any(lead):return {**ck,'prefix_targets':value}
    elapsed=np.arange(count)*DT
    envelope=np.interp(elapsed,WINDOW,(0.,1.,1.,0.))
    for joint,shift in zip(head_ids,lead):
        cursor=(elapsed+shift*envelope)/DT
        if np.any(np.diff(cursor)<=0) or cursor.min()<0 or cursor.max()>count-1+1e-9:
            raise RuntimeError('Head cursor must be monotonic and remain in early path')
        value[:count,joint]=np.interp(cursor,np.arange(len(source)),source[:,joint])
    return {**ck,'prefix_targets':value}


def changed_path(lead,hold):
    sim,cases,ck,ids=base._CTX
    if not np.any(lead):return (*path(hold),None)
    changed=warped_head_prefix(ck,ids[8:10],lead)
    targets,phases=timed_targets(sim,changed,(.4,.6,.8),hold)
    gains=np.concatenate([ck['prefix_gains'],ck['terminal_gains']])
    # Physical nominal reference only. The zero-feedback replay also retains
    # an audit-failed reference trace instead of losing it to an exception.
    reference_row,trace=rollout(sim,*cases[None][:2],True,targets,phases,ids,
                               np.zeros((len(targets),4)),np.zeros_like(gains),True)
    ref=np.asarray([row[6] for row in trace])
    if not reference_row['valid'] or len(ref)!=len(targets):
        return targets,phases,ref,gains,(reference_row,trace)
    return targets,phases,ref,gains,None


def evaluate(job):
    lead,bias,seeds,hold,required,directory=job
    sim,cases,ck,ids=base._CTX;dest=Path(directory);dest.mkdir(exist_ok=False)
    targets,phases,ref,gains,rejected=changed_path(lead,hold)
    row={'lead_s':list(lead),'bias_rad':list(bias),'cases':[],
         'nominal_success':False,'successes':0,'target_steps':len(targets)}
    if rejected is not None:
        value,trace=rejected;save_trace(dest/'failed_reference.npz',trace)
        row.update(rejected_reference=value,rejection='Physical nominal reference audit failed')
    else:
        for seed in (None,*seeds):
            source,peaks,initial,initial_hash=cases[seed]
            value,trace=run(sim,source,peaks,targets,phases,ids,ref,gains,(1.,1.),bias,True)
            value.update(seed=seed,initial=initial,initial_state_sha256=initial_hash,
                         success=success(value,len(targets),required))
            save_trace(dest/f'case_{seed}.npz',trace)
            if seed is None:
                row['nominal']=value;row['nominal_success']=value['success']
                if not value['success']:break
            else:row['cases'].append(value);row['successes']+=int(value['success'])
    (dest/'summary.json').write_text(json.dumps(row,indent=2))
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=6:p.error('Fresh output and bounded CPU workers required')
    root=Path('/data/shijinsheng/open_duck');prior=root/'outputs/getup_gyro_attenuation_r93_left_20261001/results.json'
    r93=json.loads(prior.read_text());scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    for key,file in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if digest(file)!=r93[key]:raise RuntimeError('Frozen baseline input changed')
    if r93['qualification_run'] or r93['training_best']['grid_index']!=0:
        raise RuntimeError('R94 requires the rejected R93 intervention, not a combined candidate')
    assert not set(TRAIN)&set(HELDOUT)
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for file in Path(__file__).parent.glob('*.py'):shutil.copy2(file,sources/file.name)
    grid=list(itertools.product(LEVELS,repeat=2))
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
        'method':'head and neck relative timing grid fitting before early contact transition',
        'hypothesis':'early head-to-leg timing may alter fragile head-ground load transfer',
        'not_a_proven_root_cause':True,'training_seeds':list(TRAIN),'heldout_seeds':list(HELDOUT),
        'grid_lead_seconds':grid,'window_seconds':WINDOW,'head_reference_clock_rate_bounds':[.6,1.4],
        'nominal_reference_regenerated_physically':True,'no_midpath_state_reset':True,
        'required_strict_seconds':30.,'hold_seconds':35.,'training_gate_seconds':1.,
        'noise_development_only':True,'robustness_bias_rad':BIAS,'workers':args.workers,
        'frozen':['all_nonhead_targets','targets_after_1.2s','phase_timing','feedback_gains',
                  'target_value_path','collision','mass','friction','torque','joint_range','target_slew',
                  'combined_residual_cap','strict_standing_gate','actual_initial_fall'],
        'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),
        'scene_sha256':digest(scene),'stand_sha256':digest(stand),'r93_result_sha256':digest(prior),
        'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')}}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    rows=[];robust=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker,
                             initargs=(str(scene),str(stand),ck,TRAIN)) as pool:
        jobs=[(lead,BIAS[0],TRAIN,2.,1.,str(args.output/f'grid_{i:02d}')) for i,lead in enumerate(grid)]
        for i,row in enumerate(pool.map(evaluate,jobs,chunksize=1)):
            row['grid_index']=i
            if i==0 and (not row['nominal_success'] or row['successes']!=13):
                raise RuntimeError('Identity must reproduce nominal and 13/24')
            rows.append(row);(args.output/'grid_progress.json').write_text(json.dumps(rows,indent=2))
            print('GRID',i,row['lead_s'],'NOMINAL',int(row['nominal_success']),'SUCCESS',row['successes'],
                  'EVALUATED',len(row['cases']),flush=True)
        improved=sorted([r for r in rows[1:] if r['nominal_success'] and r['successes']>13],
                        key=lambda r:r['successes'],reverse=True)[:3]
        if improved:
            for setting in [rows[0],*improved]:
                jobs=[(setting['lead_s'],bias,TRAIN,2.,1.,
                       str(args.output/f'robust_grid_{setting["grid_index"]:02d}_bias{i}'))
                      for i,bias in enumerate(BIAS)]
                noise=list(pool.map(evaluate,jobs,chunksize=1))
                entry={'setting':setting,'noise':noise,'all_nominal_success':all(n['nominal_success'] for n in noise),
                       'mean_count':float(np.mean([n['successes'] for n in noise])),
                       'worst_count':min(n['successes'] for n in noise)}
                robust.append(entry);(args.output/'robust_progress.json').write_text(json.dumps(robust,indent=2))
                print('ROBUST',setting['grid_index'],entry['mean_count'],entry['worst_count'],flush=True)
    eligible=[r for r in robust[1:] if r['all_nominal_success'] and r['mean_count']>=robust[0]['mean_count']
              and r['worst_count']>=robust[0]['worst_count']] if robust else []
    chosen=max(eligible,key=lambda r:(r['worst_count'],r['mean_count'],r['setting']['successes'])) if eligible else None
    selected=chosen['setting'] if chosen else rows[0]
    np.savez_compressed(args.output/'checkpoint.npz',**ck,r94_head_lead_s=selected['lead_s'],
                        r94_grid_completed=len(rows),r94_promoted=chosen is not None)
    report={**contract,'grid_results':rows,'robustness_results':robust,'training_best':selected,
            'qualification_run':False}
    if chosen:
        report['qualification_run']=True
        with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker,
                                 initargs=(str(scene),str(stand),ck,HELDOUT)) as pool:
            jobs=[(lead,BIAS[0],HELDOUT,35.,30.,str(args.output/f'qualification_{name}'))
                  for name,lead in [('baseline',(0.,0.)),('candidate',selected['lead_s'])]]
            a,b=list(pool.map(evaluate,jobs,chunksize=1));report['qualification']={'baseline':a,'candidate':b}
        if len(a['cases'])!=40 or len(b['cases'])!=40:
            report['qualification_rejected']='Nominal did not pass longer qualification; no stage acceptance'
        else:
            paired=[]
            for x,y in zip(a['cases'],b['cases']):
                if (x['seed'],x['initial_state_sha256'])!=(y['seed'],y['initial_state_sha256']):
                    raise RuntimeError('Fresh full-fall trials must be exactly paired')
                paired.append({'seed':x['seed'],'baseline_success':x['success'],'candidate_success':y['success']})
            report.update(heldout_pairs=paired,heldout_successes=b['successes'],baseline_successes=a['successes'],
                          paired_rescues=[v['seed'] for v in paired if v['candidate_success'] and not v['baseline_success']],
                          paired_regressions=[v['seed'] for v in paired if v['baseline_success'] and not v['candidate_success']])
    else:report['qualification_skipped_reason']='No improved setting passing nominal and micro-command robustness screen'
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('RESULTS_SAVED',args.output,flush=True)


if __name__=='__main__':main()
