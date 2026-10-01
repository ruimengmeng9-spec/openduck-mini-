"""Causal initial-encoder compensation before the contact-sensitive transition.

No seed/outcome/root-state inputs. Read joint encoders once at initial full
fall; fade compensation through unchanged feedback cap and physical limits.
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
from diagnostics.probe_getup_wait_library_r77 import TRAIN,save_trace
from diagnostics.train_getup_head_feedback_r87 import path,CAP
from diagnostics.train_getup_gyro_attenuation_r93 import BIAS
from diagnostics.train_getup_early_feedback_r81 import early_envelope
from diagnostics.search_getup_reference_feedback_r64 import features,residual,rollout
from diagnostics.search_getup_support_margin_r49 import foot_support,support_quality,supported_entry
from diagnostics.train_getup_prefix_pose_r73 import success

HELDOUT=tuple(range(796000,796040))
LEVELS=(0.,1.,.5,-1.,2.)
FADES=(.4,.8,1.2)


def encoder_offset(delta,scales,fade,elapsed):
    delta=np.asarray(delta,dtype=float);scales=np.asarray(scales,dtype=float)
    if delta.shape!=(10,) or not np.isfinite(delta).all():raise ValueError('Ten finite initial encoder errors required')
    if scales.shape!=(2,) or not np.isfinite(scales).all() or (scales<-1).any() or (scales>2).any():
        raise ValueError('Two bounded encoder compensation gains required')
    if fade not in FADES or not np.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid fade time')
    weight=.5*(1+np.cos(np.pi*elapsed/fade)) if elapsed<fade else 0.
    factor=np.r_[np.full(8,scales[0]),np.full(2,scales[1])]
    return np.clip(delta*factor*weight,-CAP,CAP)


def init_worker(scene,stand,ck,seeds):
    base.init_worker(scene,stand,ck,seeds)


def run(sim,source,peaks,targets,phases,ids,ref,gains,delta,scales,fade,bias,record):
    if (not np.any(scales) or not np.any(delta)) and not np.any(bias):
        return rollout(sim,source,peaks,True,targets,phases,ids,ref,gains,record)
    sim.restore(source);sim.peaks,sim.finite=peaks.copy(),True
    best=-np.inf;tail=longest=run_length=0;trace=[]
    for i,(command,phase) in enumerate(zip(targets,phases)):
        error=features(sim)-ref[i]
        offset=residual(error,gains[phase])+encoder_offset(delta,scales,fade,i*DT)
        offset[8:10]+=early_envelope(i*DT)*np.asarray(bias)
        offset=np.clip(offset,-CAP,CAP)
        target=command.copy();target[ids]=np.clip(target[ids]+offset,sim.lower[ids],sim.upper[ids])
        sim.step_target(target);valid=sim.physical_valid();m=sim.measure();s=foot_support(sim)
        best=max(best,support_quality(m,s));run_length=run_length+1 if supported_entry(m,s) else 0
        longest=max(longest,run_length);tail=tail+1 if m['stable'] else 0
        if record:trace.append((sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),int(phase),
                               int(m['stable']),s['margin_m'],error.copy(),offset.copy()))
        if not valid:break
    score=-100. if not valid else best+10*min(longest*DT,1.)+20*min(tail*DT,1.)
    return {'score':float(score),'valid':bool(valid),'strict_tail_s':tail*DT,
            'supported_longest_s':longest*DT,'completed_steps':i+1,'final':m,
            'final_support':s,'peaks':sim.peaks.copy()},trace


def evaluate(job):
    scales,fade,bias,seeds,hold,required,directory=job
    sim,cases,ck,ids=base._CTX;targets,phases,ref,gains=path(hold)
    qadr=sim.qadr[ids];nominal_q=cases[None][0]['qpos'][qadr].copy()
    dest=Path(directory);dest.mkdir(exist_ok=False);rows=[]
    for seed in (None,*seeds):
        source,peaks,initial,initial_hash=cases[seed]
        delta=source['qpos'][qadr]-nominal_q
        value,trace=run(sim,source,peaks,targets,phases,ids,ref,gains,delta,scales,fade,bias,True)
        value.update(seed=seed,initial=initial,initial_state_sha256=initial_hash,
                     success=success(value,len(targets),required),initial_encoder_delta_rad=delta.tolist())
        save_trace(dest/f'case_{seed}.npz',trace);rows.append(value)
    row={'scales':list(scales),'fade_s':fade,'bias_rad':list(bias),'nominal':rows[0],
         'nominal_success':rows[0]['success'],'successes':sum(r['success'] for r in rows[1:]),
         'cases':rows[1:],'nominal_encoder_reference_rad':nominal_q.tolist()}
    (dest/'summary.json').write_text(json.dumps(row,indent=2));return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args()
    if args.output.exists() or not 1<=args.workers<=6:p.error('Fresh output and bounded workers required')
    root=Path('/data/shijinsheng/open_duck');prior=root/'outputs/getup_preload_r95_left_20261001/results.json'
    r95=json.loads(prior.read_text());scene=root/'training/getup_decomposed_r4/model/scene.xml'
    stand=root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    for key,file in [('checkpoint_sha256',args.checkpoint),('scene_sha256',scene),('stand_sha256',stand)]:
        if digest(file)!=r95[key]:raise RuntimeError('Frozen baseline input changed')
    if r95['qualification_run'] or np.any(r95['training_best']['params']):
        raise RuntimeError('R96 requires the rejected zero-preload baseline, not combined interventions')
    assert not set(TRAIN)&set(HELDOUT)
    with np.load(args.checkpoint,allow_pickle=False) as z:ck={k:z[k] for k in z.files}
    args.output.mkdir();sources=args.output/'executed_sources';sources.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copy2(f,sources/f.name)
    grid=[((0.,0.),FADES[0])]+[(s,f) for s in itertools.product(LEVELS,repeat=2) if any(s) for f in FADES]
    contract={'simulation_only':True,'hardware_readiness':False,'full_task_completed':False,
        'method':'initial encoder error feedforward compensation grid fitting',
        'hypothesis':'settled joint microdifferences may contribute to contact-sensitive startup; shared fixed preloads did not help',
        'not_a_proven_root_cause':True,'training_seeds':list(TRAIN),'heldout_seeds':list(HELDOUT),
        'grid':grid,'gain_bounds':[-1.,2.],'input':'initial ten joint encoders minus frozen nominal encoder vector',
        'policy_uses_seed_or_outcome_or_root_state':False,'no_midpath_state_reset':True,
        'fade':'half cosine from 1 at start to 0 at fade_s','combined_residual_cap_rad':CAP,
        'training_gate_seconds':1.,'required_strict_seconds':30.,'hold_seconds':35.,
        'noise_development_only':True,'robustness_bias_rad':BIAS,'workers':args.workers,
        'bias_nominal_screen':'candidate must retain each baseline nominal success; same initial-encoder-zero canonical path',
        'frozen':['all_reference_targets','nominal_imu_reference','feedback_gains','phase_timing','collision','mass',
                  'friction','torque','joint_range','target_slew','strict_standing_gate','actual_initial_fall'],
        'source_sha256':digest(__file__),'checkpoint_sha256':digest(args.checkpoint),'scene_sha256':digest(scene),
        'stand_sha256':digest(stand),'r95_result_sha256':digest(prior),
        'executed_source_hashes':{f.name:digest(f) for f in sources.glob('*.py')}}
    (args.output/'contract.json').write_text(json.dumps(contract,indent=2))
    rows=[];robust=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                             initargs=(str(scene),str(stand),ck,TRAIN)) as pool:
        jobs=[(scales,fade,BIAS[0],TRAIN,2.,1.,str(args.output/f'grid_{i:02d}')) for i,(scales,fade) in enumerate(grid)]
        for i,row in enumerate(pool.map(evaluate,jobs,chunksize=1)):
            row['grid_index']=i
            if i==0 and (not row['nominal_success'] or row['successes']!=13):raise RuntimeError('Identity must reproduce nominal and 13/24')
            rows.append(row);(args.output/'grid_progress.json').write_text(json.dumps(rows,indent=2))
            print('GRID',i,row['scales'],row['fade_s'],'NOMINAL',int(row['nominal_success']),'SUCCESS',row['successes'],'/24',flush=True)
        improved=sorted([r for r in rows[1:] if r['nominal_success'] and r['successes']>13],key=lambda r:r['successes'],reverse=True)[:3]
        if improved:
            for setting in [rows[0],*improved]:
                noise=list(pool.map(evaluate,[(setting['scales'],setting['fade_s'],bias,TRAIN,2.,1.,
                                      str(args.output/f'robust_grid{setting["grid_index"]:02d}_bias{i}'))
                                     for i,bias in enumerate(BIAS)],chunksize=1))
                entry={'setting':setting,'noise':noise,'mean_count':float(np.mean([r['successes'] for r in noise])),
                       'worst_count':min(r['successes'] for r in noise)}
                robust.append(entry);(args.output/'robust_progress.json').write_text(json.dumps(robust,indent=2))
                print('ROBUST',setting['grid_index'],entry['mean_count'],entry['worst_count'],flush=True)
    eligible=[r for r in robust[1:] if r['mean_count']>=robust[0]['mean_count'] and r['worst_count']>=robust[0]['worst_count']
              and all(not a['nominal_success'] or b['nominal_success'] for a,b in zip(robust[0]['noise'],r['noise']))] if robust else []
    selected=max(eligible,key=lambda r:(r['worst_count'],r['mean_count'],r['setting']['successes']))['setting'] if eligible else rows[0]
    report={**contract,'grid_results':rows,'robustness_results':robust,'training_best':selected,
            'promoted':bool(eligible),'qualification_run':False}
    np.savez_compressed(args.output/'checkpoint.npz',**ck,r96_scales=selected['scales'],r96_fade_s=selected['fade_s'],r96_promoted=bool(eligible))
    if eligible:
        report['qualification_run']=True
        with ProcessPoolExecutor(max_workers=args.workers,initializer=init_worker,
                                 initargs=(str(scene),str(stand),ck,HELDOUT)) as pool:
            a,b=list(pool.map(evaluate,[(scales,fade,BIAS[0],HELDOUT,35.,30.,str(args.output/f'qualification_{name}'))
                            for name,scales,fade in [('baseline',(0.,0.),FADES[0]),('candidate',selected['scales'],selected['fade_s'])]],chunksize=1))
        pairs=[]
        if len(a['cases'])!=len(HELDOUT) or len(b['cases'])!=len(HELDOUT):
            raise RuntimeError('Missing independent full-fall cases')
        for x,y in zip(a['cases'],b['cases']):
            if (x['seed'],x['initial_state_sha256'])!=(y['seed'],y['initial_state_sha256']):raise RuntimeError('Unpaired full-fall starts')
            pairs.append({'seed':x['seed'],'baseline_success':x['success'],'candidate_success':y['success']})
        report.update(qualification={'baseline':a,'candidate':b},heldout_pairs=pairs,baseline_successes=a['successes'],
                      heldout_successes=b['successes'],paired_rescues=[v['seed'] for v in pairs if v['candidate_success'] and not v['baseline_success']],
                      paired_regressions=[v['seed'] for v in pairs if v['baseline_success'] and not v['candidate_success']])
    else:report['qualification_skipped_reason']='No nominal-preserving encoder compensation passing development bias comparison'
    (args.output/'results.json').write_text(json.dumps(report,indent=2));print('RESULTS_SAVED',args.output,flush=True)


if __name__=='__main__':main()
