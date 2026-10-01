"""Test the omitted vertical up-vector channel before/through head loading.

Existing four-feature feedback uses up X/Y and pitch/roll rates, not up Z.
This is a conditioning/observation hypothesis, not a proven root cause.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_wait_pose_r74 as base
from diagnostics.getup_independent_native import DT,digest
from diagnostics.probe_getup_wait_library_r77 import TRAIN,save_trace
from diagnostics.train_getup_head_feedback_r87 import path,CAP
from diagnostics.train_getup_gyro_attenuation_r93 import BIAS
from diagnostics.train_getup_early_feedback_r81 import early_envelope
from diagnostics.search_getup_reference_feedback_r64 import features,residual,rollout
from diagnostics.search_getup_support_margin_r49 import foot_support,support_quality,supported_entry
from diagnostics.train_getup_prefix_pose_r73 import success

HELDOUT=tuple(range(797000,797040))


def vertical_offset(error,flat,elapsed):
    flat=np.asarray(flat,dtype=float)
    if flat.shape!=(3,) or not np.isfinite(flat).all() or np.abs(flat).max()>2:
        raise ValueError('Three finite vertical IMU gains in [-2,2] required')
    if not np.isfinite(error) or not np.isfinite(elapsed) or elapsed<0:
        raise ValueError('Finite observed error and nonnegative elapsed required')
    weight=float(np.interp(elapsed,[0.,1.2,3.5],[1.,1.,0.],left=0.,right=0.))
    out=np.zeros(10);out[0]=flat[0]*error;out[4]=-out[0]
    out[8:10]=flat[1:]*error
    return np.clip(weight*out,-CAP,CAP)


def reference_z(sim,source,peaks,targets):
    # Separate physical reference rollout. Never reset the evaluated live trial.
    sim.restore(source);sim.peaks,sim.finite=peaks.copy(),True;out=[]
    for target in targets:
        out.append(float(sim.sensor('upvector')[2]));sim.step_target(target)
        if not sim.physical_valid():raise RuntimeError('Physical nominal vertical reference failed audit')
    return np.asarray(out)


def run(sim,source,peaks,targets,phases,ids,ref,zref,gains,flat,bias):
    vertical_offset(0.,flat,0.)
    if not np.any(flat) and not np.any(bias):
        result,trace=rollout(sim,source,peaks,True,targets,phases,ids,ref,gains,True)
        return result,trace,None
    sim.restore(source);sim.peaks,sim.finite=peaks.copy(),True
    best=-np.inf;tail=longest=length=0;trace=[];zerrors=[]
    for i,(command,phase) in enumerate(zip(targets,phases)):
        error=features(sim)-ref[i];zerror=float(sim.sensor('upvector')[2])-zref[i]
        offset=residual(error,gains[phase])+vertical_offset(zerror,flat,i*DT)
        offset[8:10]+=early_envelope(i*DT)*np.asarray(bias);offset=np.clip(offset,-CAP,CAP)
        target=command.copy();target[ids]=np.clip(target[ids]+offset,sim.lower[ids],sim.upper[ids])
        sim.step_target(target);valid=sim.physical_valid();m=sim.measure();s=foot_support(sim)
        best=max(best,support_quality(m,s));length=length+1 if supported_entry(m,s) else 0
        longest=max(longest,length);tail=tail+1 if m['stable'] else 0
        trace.append((sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),int(phase),
                      int(m['stable']),s['margin_m'],error.copy(),offset.copy()));zerrors.append(zerror)
        if not valid:break
    score=-100. if not valid else best+10*min(longest*DT,1.)+20*min(tail*DT,1.)
    return {'score':float(score),'valid':bool(valid),'strict_tail_s':tail*DT,
            'supported_longest_s':longest*DT,'completed_steps':i+1,'final':m,
            'final_support':s,'peaks':sim.peaks.copy()},trace,np.asarray(zerrors)


def evaluate(job):
    flat,bias,seeds,hold,required,directory=job
    sim,cases,ck,ids=base._CTX;targets,phases,ref,gains=path(hold)
    zref=reference_z(sim,*cases[None][:2],targets)
    dest=Path(directory);dest.mkdir(exist_ok=False);rows=[]
    np.savez_compressed(dest/'vertical_reference.npz',up_z=zref)
    for seed in (None,*seeds):
        source,peaks,initial,initial_hash=cases[seed]
        value,trace,zerrors=run(sim,source,peaks,targets,phases,ids,ref,zref,gains,flat,bias)
        value.update(seed=seed,initial=initial,initial_state_sha256=initial_hash,
                     success=success(value,len(targets),required))
        save_trace(dest/f'case_{seed}.npz',trace)
        if zerrors is not None:np.savez_compressed(dest/f'vertical_error_{seed}.npz',error=zerrors)
        rows.append(value)
    row={'params':list(flat),'bias_rad':list(bias),'nominal_success':rows[0]['success'],
         'nominal':rows[0],'successes':sum(r['success'] for r in rows[1:]),'cases':rows[1:]}
    scores=np.sort([r['score'] for r in rows[1:]])
    row['fitness']=float(14*row['successes']+scores[:6].mean()+.2*scores.mean()-.01*np.dot(flat,flat))
    (dest/'summary.json').write_text(json.dumps(row,indent=2));return row


def rank(row):return (row['successes'],row['fitness'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=6:p.error('Fresh output and bounded workers required')
    root=Path('/data/shijinsheng/open_duck');prior=root/'outputs/getup_initial_encoder_r96_left_20261001/results.json'
    old=json.loads(prior.read_text());scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    for key,file in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if digest(file)!=old[key]:raise RuntimeError('Frozen baseline inputs changed')
    if old['promoted'] or old['qualification_run']:raise RuntimeError('Do not combine a promoted intervention')
    assert not set(TRAIN)&set(HELDOUT)
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,sources/f.name)
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
        'hypothesis':'the omitted up-vector Z channel may condition early roll/head correction better than XY alone',
        'not_a_proven_root_cause':True,'training_seeds':list(TRAIN),'heldout_seeds':list(HELDOUT),
        'input':'current pre-control IMU up-vector Z minus separately recorded physical nominal reference',
        'new_outputs':['opposite hip roll corrections','neck pitch correction','head pitch correction'],
        'policy_uses_seed_or_root_state_or_future_or_contact_force':False,'no_midpath_state_reset':True,
        'window_seconds':[0.,1.2,3.5],'gain_bounds':[-2.,2.],'combined_residual_cap_rad':CAP,
        'workers':args.workers,'seed':197,'population':18,'generation_budget':24,'stagnation_budget':12,
        'training_gate_seconds':1.,'hold_seconds':35.,'required_strict_seconds':30.,
        'noise_development_only':True,'robustness_bias_rad':BIAS,
        'bias_nominal_screen':'retain every baseline nominal success plus no-worse mean/worst development counts',
        'frozen':['reference_targets','original_four_feature_feedback','phase_timing','collision','mass','friction',
                  'torque','joint_range','target_slew','strict_standing_gate','actual_initial_fall'],
        'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),'scene_sha256':digest(scene),
        'stand_sha256':digest(stand),'r96_result_sha256':digest(prior),
        'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')}}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    rng=np.random.default_rng(197);mean=np.zeros(3);std=np.full(3,.25);best=None;stale=0;history=[];robust=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker,
                             initargs=(str(scene),str(stand),ck,TRAIN)) as pool:
        for gen in range(24):
            candidates=[np.zeros(3),mean.copy()]
            if best is not None:candidates.append(np.asarray(best['params']))
            if gen==0:
                for j in range(3):
                    for magnitude in (.05,.25):
                        for sign in (-1,1):
                            flat=np.zeros(3);flat[j]=sign*magnitude;candidates.append(flat)
            while len(candidates)<18:candidates.append(np.clip(mean+rng.normal(0,std),-2,2))
            rows=list(pool.map(evaluate,[(flat,BIAS[0],TRAIN,2.,1.,str(args.output/f'gen_{gen+1:03d}_candidate_{i:03d}'))
                                          for i,flat in enumerate(candidates)],chunksize=1))
            if gen==0 and (not rows[0]['nominal_success'] or rows[0]['successes']!=13):
                raise RuntimeError('Literal baseline must reproduce nominal and 13/24')
            feasible=[r for r in rows if r['nominal_success']]
            if not feasible:raise RuntimeError('Lost literal identity')
            top=max(feasible,key=rank)
            if best is None or rank(top)>rank(best):best=top;stale=0
            else:stale+=1
            elite=np.asarray([r['params'] for r in sorted(rows,key=lambda r:r['fitness'],reverse=True)[:4]])
            mean=.5*mean+.5*elite.mean(axis=0);std=np.maximum(.03,.7*std+.3*elite.std(axis=0))
            history.append({'generation':gen+1,'best':best,'stale':stale,'nominal_eligible':len(feasible)})
            np.savez_compressed(args.output/'checkpoint.tmp.npz',**ck,r97_gains=best['params'],r97_mean=mean,
                                r97_std=std,r97_generation=gen+1)
            (args.output/'checkpoint.tmp.npz').replace(args.output/'checkpoint.npz')
            (args.output/'search_history.json').write_text(json.dumps(history,indent=2))
            (args.output/'rng_state.json').write_text(json.dumps(rng.bit_generator.state))
            print('GEN',gen+1,'BEST',best['successes'],'/24','NOMINAL',int(best['nominal_success']),'STALE',stale,flush=True)
            if best['successes']==24 or stale>=12:break
        if best['successes']>13:
            for name,flat in [('baseline',np.zeros(3)),('candidate',best['params'])]:
                noise=list(pool.map(evaluate,[(flat,bias,TRAIN,2.,1.,str(args.output/f'robust_{name}_bias{i}'))
                                              for i,bias in enumerate(BIAS)],chunksize=1))
                robust.append({'name':name,'noise':noise,'mean_count':float(np.mean([r['successes'] for r in noise])),
                               'worst_count':min(r['successes'] for r in noise)})
                (args.output/'robust_progress.json').write_text(json.dumps(robust,indent=2))
    promoted=bool(robust and robust[1]['mean_count']>=robust[0]['mean_count']
                  and robust[1]['worst_count']>=robust[0]['worst_count']
                  and all(not a['nominal_success'] or b['nominal_success'] for a,b in zip(robust[0]['noise'],robust[1]['noise'])))
    report={**contract,'generations_completed':len(history),'training_best':best,'robustness_results':robust,
            'promoted':promoted,'qualification_run':False}
    if promoted:
        report['qualification_run']=True
        with ProcessPoolExecutor(max_workers=args.workers,initializer=base.init_worker,
                                 initargs=(str(scene),str(stand),ck,HELDOUT)) as pool:
            a,b=list(pool.map(evaluate,[(flat,BIAS[0],HELDOUT,35.,30.,str(args.output/f'qualification_{name}'))
                             for name,flat in [('baseline',np.zeros(3)),('candidate',best['params'])]],chunksize=1))
        if len(a['cases'])!=40 or len(b['cases'])!=40:raise RuntimeError('Missing independent starts')
        pairs=[]
        for x,y in zip(a['cases'],b['cases']):
            if (x['seed'],x['initial_state_sha256'])!=(y['seed'],y['initial_state_sha256']):raise RuntimeError('Unpaired starts')
            pairs.append({'seed':x['seed'],'baseline_success':x['success'],'candidate_success':y['success']})
        report.update(qualification={'baseline':a,'candidate':b},heldout_pairs=pairs,
                      baseline_successes=a['successes'],heldout_successes=b['successes'],
                      paired_rescues=[v['seed'] for v in pairs if v['candidate_success'] and not v['baseline_success']],
                      paired_regressions=[v['seed'] for v in pairs if v['baseline_success'] and not v['candidate_success']])
    else:report['qualification_skipped_reason']='No improved nominal-preserving vertical IMU feedback passing development bias comparison'
    (args.output/'results.json').write_text(json.dumps(report,indent=2));print('RESULTS_SAVED',args.output,flush=True)


if __name__=='__main__':main()
