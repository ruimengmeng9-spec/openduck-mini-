"""Early head/neck coordination search with leg references held fixed.

Earlier R80/R81 early interventions deliberately excluded head joints.
This tests their moving-mass/support contribution, not a proven cause.
Every complete trial begins from its physical fall and retains StrictSim.
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
from diagnostics.probe_getup_initial_settling_r84 import audited_reference
from diagnostics.probe_getup_wait_library_r77 import TRAIN,save_trace
from diagnostics.search_getup_reference_feedback_r64 import rollout
from diagnostics.train_getup_prefix_pose_r73 import success

BOUND=.35
KNOT_TIMES=np.array([.5,1.5,2.5,3.5])
NAMES=('neck_pitch','head_pitch')
HELDOUT=tuple(range(786000,786040))
_OUT=None


def edited_head(sim,ck,flat):
    delta=np.asarray(flat,dtype=float)
    if delta.shape!=(4,) or not np.isfinite(delta).all() or abs(delta).max()>BOUND+1e-10:
        raise ValueError('Need four finite bounded head-knot offsets')
    phases=ck['prefix_phases'];ix=np.flatnonzero(phases==0)
    if len(ix)<176 or not np.array_equal(ix,np.arange(len(ix))):
        raise ValueError('Need full contiguous phase 0')
    targets=ck['prefix_targets'].copy();knots=np.vstack([np.zeros(2),delta.reshape(2,2),np.zeros(2)])
    offset=np.stack([np.interp(ix*DT,KNOT_TIMES,knots[:,j],left=0.,right=0.) for j in range(2)],axis=1)
    ids=np.array([sim.model.actuator(name).id for name in NAMES])
    targets[np.ix_(ix,ids)]=np.clip(targets[np.ix_(ix,ids)]+offset,sim.lower[ids],sim.upper[ids])
    return {**ck,'prefix_targets':targets}


def init_worker(scene,stand,ck,seeds,output):
    global _OUT
    base.init_worker(scene,stand,ck,seeds);_OUT=Path(output)


def run_path(flat,hold,dest):
    sim,cases,ck,ids=base._CTX
    targets,phases=timed_targets(sim,edited_head(sim,ck,flat),(.4,.6,.8),hold)
    ref,peaks=audited_reference(sim,cases[None],targets,dest)
    gains=np.concatenate([ck['prefix_gains'],ck['terminal_gains']])
    return targets,phases,ref,gains,peaks


def objective(job):
    tag,flat=job
    sim,cases,ck,ids=base._CTX
    dest=_OUT/tag;dest.mkdir(exist_ok=False)
    targets,phases,ref,gains,peaks=run_path(flat,2.,dest)
    row={'params':np.asarray(flat).tolist(),'successes':0,'nominal_success':False,
         'quality':-1e4,'nominal_reference_peaks':peaks}
    if ref is None:row['rejected']='Nominal reference failed physical audit'
    else:
        nominal,trace=rollout(sim,*cases[None][:2],True,targets,phases,ids,ref,gains,True)
        row['nominal']=nominal;row['nominal_success']=success(nominal,len(targets),1.)
        if not row['nominal_success']:
            save_trace(dest/'failed_nominal.npz',trace)
        else:
            rs=[]
            for seed,case in cases.items():
                if seed is None:continue
                value,_=rollout(sim,*case[:2],True,targets,phases,ids,ref,gains)
                rs.append({'seed':seed,'success':success(value,len(targets),1.),**value})
            row['cases']=rs;row['successes']=sum(r['success'] for r in rs)
            scores=np.sort([r['score'] for r in rs])
            row['quality']=float(scores[:6].mean()+.2*scores.mean()-.1*np.dot(flat,flat))
    (dest/'summary.json').write_text(json.dumps(row,indent=2))
    return row


def rank(row):
    # Actual success count dominates reward; preserve mandatory nominal gate.
    return (int(row['nominal_success']),row['successes'],row['quality'])


def qualify(job):
    seed,flat,directory,hold,required=job
    sim,cases,ck,ids=base._CTX;dest=Path(directory)/f'case_{seed}';dest.mkdir(exist_ok=False)
    source,peaks,initial,initial_hash=cases[seed]
    row={'seed':seed,'initial_fallen':True,'initial':initial,'initial_state_sha256':initial_hash}
    for name,params in [('baseline',np.zeros(4)),('candidate',flat)]:
        d=dest/name;d.mkdir()
        targets,phases,ref,gains,reference_peaks=run_path(params,hold,d)
        if ref is None:raise RuntimeError('Selected reference failed physical audit on long hold')
        value,trace=rollout(sim,source,peaks,True,targets,phases,ids,ref,gains,True)
        value['success']=success(value,len(targets),required)
        row[name]=value;save_trace(d/'trace.npz',trace)
    (dest/'summary.json').write_text(json.dumps(row,indent=2))
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--generations',type=int,default=24)
    p.add_argument('--population',type=int,default=24)
    p.add_argument('--seed',type=int,default=186)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=6 or args.population<24 or args.generations<1:
        p.error('Need fresh output and bounded search settings')
    root=Path('/data/shijinsheng/open_duck');scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    prior=root/'outputs/getup_initial_settling_r84_left_20261001/results.json'
    old=json.loads(prior.read_text())
    for key,path in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if old[key]!=digest(path):raise RuntimeError('Frozen input changed')
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    assert not set(TRAIN)&set(HELDOUT)
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,sources/f.name)
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
              'method':'early head/neck two-knot trajectory CEM, leg commands frozen',
              'hypothesis':'head/neck motion can improve early body rotation without changing leg trajectory',
              'no_midpath_state_reset':True,'training_seeds':list(TRAIN),'heldout_seeds':list(HELDOUT),
              'nominal_gate_required':True,'learned_joint_order':list(NAMES),'knot_times_s':KNOT_TIMES.tolist(),
              'delta_bound_rad':BOUND,'training_gate_seconds':1.,'required_strict_seconds':30.,'hold_seconds':35.,
              'qualification_only_after_training_improvement':True,'ranking':'nominal, actual success count, quality',
              'frozen':['leg_reference_targets','targets_outside_window','phase_timing','initial_settling',
                        'feedback_gains','collision','torque','joint_range','target_slew','strict_gate'],
              'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),
              'scene_sha256':digest(scene),'stand_sha256':digest(stand),'r84_result_sha256':digest(prior),
              'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')},
              'seed':args.seed,'generation_budget':args.generations,'population':args.population,'workers':args.workers}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    candidates=args.output/'candidates';candidates.mkdir()
    rng=np.random.default_rng(args.seed);mean=np.zeros(4);std=np.full(4,.06)
    best=None;history=[];stale=0;started=time.monotonic();baseline_count=None
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                             initargs=(str(scene),str(stand),ck,TRAIN,str(candidates))) as pool:
        for gen in range(args.generations):
            population=[np.zeros(4),mean.copy()]
            if best is not None:population.append(best[0].copy())
            if gen==0:
                for size in (.025,.075):
                    for j in range(4):
                        for sign in (-1.,1.):
                            delta=np.zeros(4);delta[j]=sign*size;population.append(delta)
            while len(population)<args.population:
                center=best[0] if best is not None and len(population)%3==0 else mean
                population.append(np.clip(center+rng.normal(0,std),-BOUND,BOUND))
            jobs=[(f'gen_{gen+1:03d}_candidate_{i:03d}',flat) for i,flat in enumerate(population)]
            evaluated=list(pool.map(objective,jobs,chunksize=1))
            if gen==0:
                if not evaluated[0]['nominal_success'] or evaluated[0]['successes']!=13:
                    raise RuntimeError('Zero-head-offset baseline failed reproduction')
                baseline_count=evaluated[0]['successes']
            ranked=sorted(zip(population,evaluated),key=lambda x:rank(x[1]),reverse=True)
            if best is None or rank(ranked[0][1])>rank(best[1]):best=ranked[0];stale=0
            else:stale+=1
            elite=np.stack([x[0] for x in ranked[:max(4,args.population//5)]])
            mean=.5*mean+.5*elite.mean(axis=0);std=np.maximum(.015,.7*std+.3*elite.std(axis=0))
            history.append({'generation':gen+1,'best':best[1],'stale':stale,
                            'nominal_eligible_candidates':sum(x['nominal_success'] for x in evaluated),
                            'wall_seconds':time.monotonic()-started})
            np.savez_compressed(args.output/'checkpoint.tmp.npz',**ck,r86_head_delta=best[0],
                                r86_mean=mean,r86_std=std,r86_generation=gen+1)
            (args.output/'checkpoint.tmp.npz').replace(args.output/'checkpoint.npz')
            (args.output/'search_history.json').write_text(json.dumps(history,indent=2))
            (args.output/'rng_state.json').write_text(json.dumps(rng.bit_generator.state))
            print('GEN',gen+1,'SUCCESS',best[1]['successes'],'/24','NOMINAL',int(best[1]['nominal_success']),
                  'ELIGIBLE',history[-1]['nominal_eligible_candidates'],'STALE',stale,flush=True)
            if best[1]['successes']==24 or stale>=12:break
        review=args.output/'training_review';review.mkdir()
        trainrows=list(pool.map(qualify,[(s,best[0],str(review),2.,1.) for s in (None,*TRAIN)],chunksize=1))
    report={**contract,'generations_completed':len(history),'training_best':best[1],
            'training_review':trainrows,'qualification_run':False}
    if best[1]['successes']>baseline_count:
        report['qualification_run']=True;qualification=args.output/'qualification';qualification.mkdir();rows=[]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                                 initargs=(str(scene),str(stand),ck,HELDOUT,str(candidates))) as pool:
            for row in pool.map(qualify,[(s,best[0],str(qualification),35.,30.) for s in (None,*HELDOUT)],chunksize=1):
                rows.append(row);(args.output/'qualification_progress.json').write_text(json.dumps(rows,indent=2))
                print('HELDOUT',row['seed'],'BASE',int(row['baseline']['success']),
                      'CANDIDATE',int(row['candidate']['success']),flush=True)
        rs=rows[1:];report.update(canonical=rows[0],heldout=rs,heldout_trials=len(rs),
                                baseline_successes=sum(r['baseline']['success'] for r in rs),
                                heldout_successes=sum(r['candidate']['success'] for r in rs),
                                paired_rescues=[r['seed'] for r in rs if r['candidate']['success'] and not r['baseline']['success']],
                                paired_regressions=[r['seed'] for r in rs if r['baseline']['success'] and not r['candidate']['success']])
        print('QUALIFIED',report['heldout_successes'],'/40 BASELINE',report['baseline_successes'],flush=True)
    else:report['qualification_skipped_reason']='No training-count improvement; fresh seeds unused'
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print('RESULTS_SAVED',args.output,flush=True)


if __name__=='__main__':main()
