"""R95 coordinated early preload and full-cohort search through nominal failures.

Changes precede the first R92 contact divergence. Intermediate optimization
may use candidates failing nominal standing, but never waive physical audit
or the nominal/fresh continuous-standing gates for the selected candidate.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import time
import numpy as np
from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.getup_independent_native import DT,digest
from diagnostics.probe_getup_terminal_timing_r72 import timed_targets
from diagnostics.probe_getup_wait_library_r77 import TRAIN,save_trace
from diagnostics.train_getup_gyro_attenuation_r93 import run,BIAS
from diagnostics.search_getup_reference_feedback_r64 import rollout
from diagnostics.train_getup_prefix_pose_r73 import success
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS

BOUND=.12
WINDOW=(0.,.3,.6,.9)
HELDOUT=tuple(range(795000,795040))


def edited_preload(sim,ck,ids,flat):
    flat=np.asarray(flat,dtype=float)
    if flat.shape!=(10,) or not np.isfinite(flat).all() or abs(flat).max()>BOUND+1e-10:
        raise ValueError('Need ten finite bounded preload offsets')
    count=round(WINDOW[-1]/DT)+1
    if len(ck['prefix_targets'])<count or np.any(ck['prefix_phases'][:count]!=0):
        raise ValueError('Preload must remain in original early phase')
    target=ck['prefix_targets'].copy()
    if np.any(flat):
        weight=np.interp(np.arange(count)*DT,WINDOW,(0.,1.,1.,0.))
        target[np.ix_(np.arange(count),ids)]=np.clip(target[np.ix_(np.arange(count),ids)]+weight[:,None]*flat,
                                                     sim.lower[ids],sim.upper[ids])
    return {**ck,'prefix_targets':target}


def fitness(nominal,cases,flat):
    # Search may traverse a non-standing nominal path. Physical failures are
    # still scored as failures; this does not change final acceptance.
    scores=np.sort([r['score'] for r in cases])
    passed=sum(r['success'] for r in cases)
    return float(14*passed+4*int(nominal['success'])+.2*nominal['score']+
                 scores[:6].mean()+.2*scores.mean()-.1*np.dot(flat,flat))


def feasible_rank(row):
    return (int(row['nominal_success']),row['successes'],row['fitness'])


def evaluate(job):
    flat,bias,seeds,hold,required,directory=job
    sim,cases,ck,ids=base._CTX;dest=Path(directory);dest.mkdir(exist_ok=False)
    changed=edited_preload(sim,ck,ids,flat)
    targets,phases=timed_targets(sim,changed,(.4,.6,.8),hold)
    gains=np.concatenate([ck['prefix_gains'],ck['terminal_gains']])
    value,trace=rollout(sim,*cases[None][:2],True,targets,phases,ids,
                        np.zeros((len(targets),4)),np.zeros_like(gains),True)
    row={'params':np.asarray(flat).tolist(),'bias_rad':list(bias),'target_steps':len(targets),
         'nominal_success':False,'successes':0,'cases':[],'fitness':-1e5,
         'nominal_reference_peaks':value['peaks']}
    if not value['valid'] or len(trace)!=len(targets):
        save_trace(dest/'failed_reference.npz',trace)
        row.update(rejected_reference=value,rejection='Physical nominal reference failed')
    else:
        ref=np.asarray([r[6] for r in trace]);values=[]
        for seed in (None,*seeds):
            source,peaks,initial,initial_hash=cases[seed]
            value,trace=run(sim,source,peaks,targets,phases,ids,ref,gains,(1.,1.),bias,True)
            value.update(seed=seed,initial=initial,initial_state_sha256=initial_hash,
                         success=success(value,len(targets),required))
            save_trace(dest/f'case_{seed}.npz',trace);values.append(value)
        row.update(nominal=values[0],nominal_success=values[0]['success'],cases=values[1:],
                   successes=sum(r['success'] for r in values[1:]))
        row['fitness']=fitness(values[0],values[1:],flat)
    (dest/'summary.json').write_text(json.dumps(row,indent=2))
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seed',type=int,default=195);p.add_argument('--generations',type=int,default=32)
    p.add_argument('--population',type=int,default=24);args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=6 or not 1<=args.generations<=32 or args.population!=24:
        p.error('Need fresh output and bounded search budget')
    root=Path('/data/shijinsheng/open_duck');prior=root/'outputs/getup_head_phase_r94_left_20261001/results.json'
    r94=json.loads(prior.read_text());scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    for key,file in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if digest(file)!=r94[key]:raise RuntimeError('Frozen inputs changed')
    if r94['qualification_run'] or r94['training_best']['grid_index']!=0:
        raise RuntimeError('R95 does not combine a promoted R94 intervention')
    assert not set(TRAIN)&set(HELDOUT)
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,sources/f.name)
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
        'method':'coordinated ten-joint early preload CEM with full-cohort intermediate scoring',
        'hypothesis':'shared pre-contact body preparation may avoid fragile head loading; mandatory nominal pruning may hide search gradients',
        'not_a_proven_root_cause':True,'training_seeds':list(TRAIN),'heldout_seeds':list(HELDOUT),
        'window_seconds':WINDOW,'preload_bound_rad':BOUND,'learned_joints':list(JOINTS),
        'search_nominal_short_gate_failure_allowed':True,'selected_nominal_success_required':True,
        'no_midpath_state_reset':True,'training_gate_seconds':1.,'required_strict_seconds':30.,'hold_seconds':35.,
        'robustness_bias_rad':BIAS,'noise_development_only':True,'seed':args.seed,'workers':args.workers,
        'generation_budget':args.generations,'population':args.population,
        'frozen':['targets_after_0.9s','feedback_gains','phase_timing','actual_fallen_starts','collision','mass',
                  'friction','torque','joint_range','target_slew','combined_residual_cap','strict_standing_gate'],
        'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),'scene_sha256':digest(scene),
        'stand_sha256':digest(stand),'r94_result_sha256':digest(prior),
        'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')}}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    rng=np.random.default_rng(args.seed);mean=np.zeros(10);std=np.full(10,.025)
    best=None;stale=0;history=[];robust=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker,
                             initargs=(str(scene),str(stand),ck,TRAIN)) as pool:
        for gen in range(args.generations):
            candidates=[np.zeros(10),mean.copy()]
            if best is not None:candidates.append(np.asarray(best['params']))
            if gen==0:
                for j in range(10):
                    for sign in (-1.,1.):
                        flat=np.zeros(10);flat[j]=sign*.02;candidates.append(flat)
            while len(candidates)<args.population:
                candidates.append(np.clip(mean+rng.normal(0.,std),-BOUND,BOUND))
            rows=list(pool.map(evaluate,[(flat,BIAS[0],TRAIN,2.,1.,str(args.output/f'gen_{gen+1:03d}_candidate_{i:03d}'))
                                         for i,flat in enumerate(candidates)],chunksize=1))
            if gen==0 and (not rows[0]['nominal_success'] or rows[0]['successes']!=13):
                raise RuntimeError('Literal unchanged preload must reproduce nominal and 13/24')
            feasible=[r for r in rows if r['nominal_success']]
            top=max(feasible,key=feasible_rank)
            if best is None or feasible_rank(top)>feasible_rank(best):best=top;stale=0
            else:stale+=1
            # Intermediate distribution update includes physically valid
            # nominal paths which did not finish standing. Final best never does.
            ranked=sorted(rows,key=lambda r:r['fitness'],reverse=True)
            elite=np.asarray([r['params'] for r in ranked[:5]])
            mean=.5*mean+.5*elite.mean(axis=0);std=np.maximum(.01,.7*std+.3*elite.std(axis=0))
            history.append({'generation':gen+1,'best':best,'search_best':ranked[0],
                            'nominal_eligible_candidates':len(feasible),'stale':stale,
                            'wall_seconds':time.monotonic()-started})
            np.savez_compressed(args.output/'checkpoint.tmp.npz',**ck,r95_preload_rad=best['params'],
                                r95_mean=mean,r95_std=std,r95_generation=gen+1)
            (args.output/'checkpoint.tmp.npz').replace(args.output/'checkpoint.npz')
            (args.output/'search_history.json').write_text(json.dumps(history,indent=2))
            (args.output/'rng_state.json').write_text(json.dumps(rng.bit_generator.state))
            print('GEN',gen+1,'FEASIBLE_SUCCESS',best['successes'],'/24','SEARCH_SUCCESS',ranked[0]['successes'],
                  'SEARCH_NOMINAL',int(ranked[0]['nominal_success']),'ELIGIBLE',len(feasible),'STALE',stale,flush=True)
            if best['successes']==24 or stale>=16:break
        review=list(pool.map(evaluate,[(flat,BIAS[0],TRAIN,2.,1.,str(args.output/f'review_{name}'))
                              for name,flat in [('baseline',np.zeros(10)),('candidate',best['params'])]],chunksize=1))
        if best['successes']>13:
            for name,flat in [('baseline',np.zeros(10)),('candidate',best['params'])]:
                noise=list(pool.map(evaluate,[(flat,bias,TRAIN,2.,1.,str(args.output/f'robust_{name}_bias{i}'))
                                              for i,bias in enumerate(BIAS)],chunksize=1))
                row={'name':name,'noise':noise,'all_nominal_success':all(r['nominal_success'] for r in noise),
                     'mean_count':float(np.mean([r['successes'] for r in noise])),
                     'worst_count':min(r['successes'] for r in noise)}
                robust.append(row);(args.output/'robust_progress.json').write_text(json.dumps(robust,indent=2))
    report={**contract,'generations_completed':len(history),'training_best':best,'training_review':review,
            'robustness_results':robust,'qualification_run':False}
    promoted=bool(robust and robust[1]['all_nominal_success'] and robust[1]['mean_count']>=robust[0]['mean_count']
                  and robust[1]['worst_count']>=robust[0]['worst_count'])
    if promoted:
        report['qualification_run']=True
        with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker,
                                 initargs=(str(scene),str(stand),ck,HELDOUT)) as pool:
            a,b=list(pool.map(evaluate,[(flat,BIAS[0],HELDOUT,35.,30.,str(args.output/f'qualification_{name}'))
                               for name,flat in [('baseline',np.zeros(10)),('candidate',best['params'])]],chunksize=1))
        if len(a['cases'])!=40 or len(b['cases'])!=40:raise RuntimeError('Missing independent cases')
        paired=[]
        for x,y in zip(a['cases'],b['cases']):
            if (x['seed'],x['initial_state_sha256'])!=(y['seed'],y['initial_state_sha256']):
                raise RuntimeError('Unpaired full-fall starts')
            paired.append({'seed':x['seed'],'baseline_success':x['success'],'candidate_success':y['success']})
        report.update(qualification={'baseline':a,'candidate':b},heldout_pairs=paired,
                      baseline_successes=a['successes'],heldout_successes=b['successes'],
                      paired_rescues=[v['seed'] for v in paired if v['candidate_success'] and not v['baseline_success']],
                      paired_regressions=[v['seed'] for v in paired if v['baseline_success'] and not v['candidate_success']])
    else:report['qualification_skipped_reason']='No improved feasible preload passing micro-command development robustness'
    report['promoted']=promoted
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('RESULTS_SAVED',args.output,flush=True)


if __name__=='__main__':main()
