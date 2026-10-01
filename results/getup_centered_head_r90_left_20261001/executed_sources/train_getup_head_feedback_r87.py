"""R87: causal early head/neck reference-error feedback.

R86 fixed head offsets rescued different training failures but never improved
the common policy. Test state-dependent head response with all reference
targets, existing leg feedback, physics and strict gates unchanged.
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
from diagnostics.search_getup_reference_feedback_r64 import features,residual,record_reference,rollout
from diagnostics.search_getup_support_margin_r49 import foot_support,support_quality,supported_entry
from diagnostics.search_getup_sustained_bridge_r42 import JOINTS
from diagnostics.train_getup_prefix_pose_r73 import success
from diagnostics.train_getup_early_feedback_r81 import early_envelope
from diagnostics.train_getup_early_head_r86 import rank

BOUND=2.
CAP=.18
HELDOUT=tuple(range(787000,787040))
_OUT=None


def matrix(flat):
    x=np.asarray(flat,dtype=float)
    if x.shape!=(8,) or not np.isfinite(x).all() or abs(x).max()>BOUND+1e-10:
        raise ValueError('Need eight finite bounded head feedback gains')
    return x.reshape(2,4)


def correction(error,original_gains,head_gains,elapsed):
    original=residual(error,original_gains)
    weight=early_envelope(elapsed)
    if not weight:return original
    out=original.copy()
    out[8:10]+=weight*(head_gains@error)
    return np.clip(out,-CAP,CAP)


def controlled_rollout(sim,source,peaks,targets,phases,ids,ref,gains,flat,record=False):
    head=matrix(flat)
    if not np.any(head):return rollout(sim,source,peaks,True,targets,phases,ids,ref,gains,record)
    sim.restore(source);sim.peaks,sim.finite=peaks.copy(),True
    best=-np.inf;tail=longest=run=0;trace=[]
    for i,(target_base,phase) in enumerate(zip(targets,phases)):
        error=features(sim)-ref[i]  # Pre-control observation; no future trial state.
        offset=correction(error,gains[phase],head,i*DT)
        target=target_base.copy();target[ids]=np.clip(target[ids]+offset,sim.lower[ids],sim.upper[ids])
        sim.step_target(target)
        valid=sim.physical_valid();m=sim.measure();s=foot_support(sim)
        best=max(best,support_quality(m,s));run=run+1 if supported_entry(m,s) else 0
        longest=max(longest,run);tail=tail+1 if m['stable'] else 0
        if record:trace.append((sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),int(phase),
                               int(m['stable']),s['margin_m'],error.copy(),offset.copy()))
        if not valid:break
    score=-100. if not valid else best+10*min(longest*DT,1.)+20*min(tail*DT,1.)
    return {'score':float(score),'valid':bool(valid),'strict_tail_s':tail*DT,
            'supported_longest_s':longest*DT,'completed_steps':i+1,'final':m,
            'final_support':s,'peaks':sim.peaks.copy()},trace


def init_worker(scene,stand,ck,seeds,out):
    global _OUT
    base.init_worker(scene,stand,ck,seeds);_OUT=Path(out)
    if tuple(JOINTS[8:10])!=('neck_pitch','head_pitch'):
        raise RuntimeError('Frozen actuator ordering changed')


def path(hold):
    sim,cases,ck,ids=base._CTX
    targets,phases=timed_targets(sim,ck,(.4,.6,.8),hold)
    ref=record_reference(sim,*cases[None][:2],True,targets)
    gains=np.concatenate([ck['prefix_gains'],ck['terminal_gains']])
    return targets,phases,ref,gains


def objective(job):
    tag,flat=job;sim,cases,ck,ids=base._CTX
    targets,phases,ref,gains=path(2.)
    dest=_OUT/tag;dest.mkdir(exist_ok=False)
    nominal,trace=controlled_rollout(sim,*cases[None][:2],targets,phases,ids,ref,gains,flat,True)
    row={'params':np.asarray(flat).tolist(),'nominal_success':success(nominal,len(targets),1.),
         'nominal':nominal,'successes':0,'quality':-1e4}
    if not row['nominal_success']:save_trace(dest/'failed_nominal.npz',trace)
    else:
        rows=[]
        for seed,case in cases.items():
            if seed is None:continue
            value,_=controlled_rollout(sim,*case[:2],targets,phases,ids,ref,gains,flat)
            rows.append({'seed':seed,'success':success(value,len(targets),1.),**value})
        row['cases']=rows;row['successes']=sum(r['success'] for r in rows)
        scores=np.sort([r['score'] for r in rows])
        row['quality']=float(scores[:6].mean()+.2*scores.mean()-.03*np.dot(flat,flat))
    (dest/'summary.json').write_text(json.dumps(row,indent=2))
    return row


def qualify(job):
    seed,flat,directory,hold,required=job;sim,cases,ck,ids=base._CTX
    source,peaks,initial,initial_hash=cases[seed]
    targets,phases,ref,gains=path(hold)
    dest=Path(directory)/f'case_{seed}';dest.mkdir(exist_ok=False)
    row={'seed':seed,'initial_fallen':True,'initial':initial,'initial_state_sha256':initial_hash}
    for name,params in [('baseline',np.zeros(8)),('candidate',flat)]:
        value,trace=controlled_rollout(sim,source,peaks,targets,phases,ids,ref,gains,params,True)
        value['success']=success(value,len(targets),required);row[name]=value
        save_trace(dest/f'{name}.npz',trace)
    (dest/'summary.json').write_text(json.dumps(row,indent=2))
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--generations',type=int,default=24)
    p.add_argument('--population',type=int,default=24);p.add_argument('--seed',type=int,default=187)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=6 or args.population<24 or args.generations<1:
        p.error('Need fresh output and bounded search settings')
    root=Path('/data/shijinsheng/open_duck');scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    prior=root/'outputs/getup_early_head_r86_left_20261001/results.json';r86=json.loads(prior.read_text())
    for key,f in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if r86[key]!=digest(f):raise RuntimeError('Frozen R86 inputs changed')
    assert not set(TRAIN)&set(HELDOUT)
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,sources/f.name)
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
              'method':'causal head/neck IMU reference-error feedback CEM',
              'hypothesis':'R86 opposing rescues/regressions require case-dependent head response, not fixed offsets',
              'no_midpath_state_reset':True,'nominal_gate_required':True,'training_seeds':list(TRAIN),
              'heldout_seeds':list(HELDOUT),'learned_joints':['neck_pitch','head_pitch'],
              'features':['upvector_X_error','upvector_Y_error','0.15_gyro_Y_error','0.15_gyro_X_error'],
              'window_seconds':[.5,1.5,2.5,3.5],'feedback_gain_bound':BOUND,'combined_residual_cap_rad':CAP,
              'training_gate_seconds':1.,'required_strict_seconds':30.,'hold_seconds':35.,
              'qualification_only_after_training_improvement':True,'ranking':'nominal, success count, quality',
              'frozen':['all_reference_targets','phase_timing','existing_leg_feedback','initial_settling',
                        'collision','torque','joint_limits','slew','strict_gate'],
              'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),'scene_sha256':digest(scene),
              'stand_sha256':digest(stand),'r86_result_sha256':digest(prior),
              'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')},
              'seed':args.seed,'generation_budget':args.generations,'population':args.population,'workers':args.workers}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    candidate_dir=args.output/'candidates';candidate_dir.mkdir()
    rng=np.random.default_rng(args.seed);mean=np.zeros(8);std=np.full(8,.3)
    best=None;stale=0;history=[];started=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                             initargs=(str(scene),str(stand),ck,TRAIN,str(candidate_dir))) as pool:
        for gen in range(args.generations):
            population=[np.zeros(8),mean.copy()]
            if best is not None:population.append(best[0].copy())
            if gen==0:
                for j in range(8):
                    for sign in (-1.,1.):
                        flat=np.zeros(8);flat[j]=sign*.4;population.append(flat)
            while len(population)<args.population:
                center=best[0] if best is not None and len(population)%3==0 else mean
                population.append(np.clip(center+rng.normal(0,std),-BOUND,BOUND))
            evaluated=list(pool.map(objective,[(f'gen_{gen+1:03d}_candidate_{i:03d}',flat)
                                               for i,flat in enumerate(population)],chunksize=1))
            if gen==0 and (not evaluated[0]['nominal_success'] or
                           evaluated[0]['successes']!=r86['training_best']['successes']):
                raise RuntimeError('Zero-gain baseline does not reproduce R86')
            ranked=sorted(zip(population,evaluated),key=lambda v:rank(v[1]),reverse=True)
            if best is None or rank(ranked[0][1])>rank(best[1]):best=ranked[0];stale=0
            else:stale+=1
            elite=np.stack([x[0] for x in ranked[:max(4,args.population//5)]])
            mean=.5*mean+.5*elite.mean(axis=0);std=np.maximum(.06,.7*std+.3*elite.std(axis=0))
            history.append({'generation':gen+1,'best':best[1],'stale':stale,
                            'nominal_eligible_candidates':sum(r['nominal_success'] for r in evaluated),
                            'wall_seconds':time.monotonic()-started})
            np.savez_compressed(args.output/'checkpoint.tmp.npz',**ck,r87_head_gains=best[0].reshape(2,4),
                                r87_mean=mean,r87_std=std,r87_generation=gen+1)
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
    if best[1]['successes']>r86['training_best']['successes']:
        report['qualification_run']=True;qualification=args.output/'qualification';qualification.mkdir();rows=[]
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                                 initargs=(str(scene),str(stand),ck,HELDOUT,str(candidate_dir))) as pool:
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
