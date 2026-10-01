"""Fit a conservative initial-state trajectory library, not framewise BC.

Whole verified R22 two-knot sequences are retrieved without averaging actions.
Unknown states and states closer to a successful home anchor use home. This
nonparametric research policy is near-standing only, not fallen recovery.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnostics.distill_getup_rescue_r23 import physical_contract
from diagnostics.getup_native_curriculum import NativeEpisode, at_goal, physical_safe
from diagnostics.getup_independent_native import DT, SLEW, digest
from diagnostics.search_getup_rescue_r22 import target_at, BOUNDS


FEATURES=np.r_[np.arange(20),np.arange(34,48)]
SCALES=np.r_[np.full(3,2.),np.full(3,.15),np.full(14,.25),np.full(14,.25)]
_context=None


def sequence_choice(observation, library, radius):
    if not np.isfinite(radius) or radius<0:
        raise ValueError('invalid retrieval radius')
    o=np.asarray(observation,dtype=float)
    if o.shape!=(50,) or not np.isfinite(o).all():
        raise ValueError('invalid initial observation')
    if radius==0:return np.zeros((2,8)),dict(mode='home',prototype_seed=None)
    distance=np.linalg.norm((library['obs'][:,FEATURES]-o[FEATURES])/SCALES,axis=1)
    rescue=np.flatnonzero(library['rescue']);home=np.flatnonzero(~library['rescue'])
    if not len(rescue) or not len(home):raise ValueError('both rescue and home anchors required')
    index=int(rescue[np.argmin(distance[rescue])]);d=float(distance[index]);h=float(distance[home].min())
    # Fixed safety margin, not optimized on evaluation outcomes. No joint-target
    # interpolation across incompatible teachers, and no root-state access.
    use=d<=radius and d<.8*h
    return (library['knots'][index].copy() if use else np.zeros((2,8))),dict(
        mode='rescue' if use else 'home',prototype_seed=int(library['seed'][index]) if use else None,
        rescue_distance=d,home_distance=h)


def fit_library(folders):
    anchors=[];seen=set()
    for folder in folders:
        report=json.loads((folder/'results.json').read_text())
        for row in report['results']:
            if row['selected']['success'] and row['selected_long']['success']:
                anchors.append((row['seed'],row['baseline']['initial_obs'],row['knots'],True))
        for row in json.loads((folder/'baseline_scan.json').read_text()):
            if row['success']:
                anchors.append((row['seed'],row['initial_obs'],np.zeros((2,8)).tolist(),False))
    for seed,_,knots,_ in anchors:
        if seed in seen:raise ValueError('duplicate demonstration seed')
        seen.add(seed)
        if np.asarray(knots).shape!=(2,8) or np.any(np.abs(knots)>BOUNDS+1e-8):
            raise ValueError('teacher target outside declared range')
    anchors.sort(key=lambda r:r[0])
    library=dict(seed=np.array([r[0] for r in anchors]),obs=np.array([r[1] for r in anchors]),
        knots=np.array([r[2] for r in anchors]),rescue=np.array([r[3] for r in anchors],dtype=bool))
    if not np.isfinite(library['obs']).all() or not library['rescue'].any() or library['rescue'].all():
        raise ValueError('invalid or unbalanced library')
    return library


def init_worker(contract_path,library_path):
    global _context
    c=json.loads(Path(contract_path).read_text())
    e=NativeEpisode(c['scene_path'],'/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',c['starts_rad'],1,.55)
    with np.load(library_path,allow_pickle=False) as a:library={k:a[k] for k in a.files}
    _context=e,library


def evaluate(job):
    radius,seed,steps=job;e,library=_context
    e.rng=np.random.default_rng(seed);e.reset();sim=e.sim
    initial_hash=hashlib.sha256(sim.data.qpos.tobytes()+sim.data.qvel.tobytes()).hexdigest()
    knots,choice=sequence_choice(sim.observation(),library,radius)
    safe=True;tail=0;success4=False;continuous=True;before=sim.prev.copy();max_slew=0.;trace=[]
    for step in range(steps):
        applied=sim.step_target(target_at(sim,knots,step*DT))
        max_slew=max(max_slew,float(np.abs(applied-before).max()/DT));before=applied.copy()
        if max_slew>SLEW+1e-6:raise RuntimeError('target slew violated')
        m=sim.measure();safe=safe and physical_safe(sim,m);goal=at_goal(m)
        tail=tail+1 if goal else 0
        if step==199:success4=bool(tail>=100 and safe)
        if step>=200:continuous=continuous and goal
        if step%25==0:trace.append(dict(time_s=(step+1)*DT,height_m=m['height_m'],up_z=m['up_z'],goal=goal))
        if not physical_safe(sim,m) or m['up_z']<.45:break
    completed=step+1
    return dict(radius=radius,seed=int(seed),initial_hash=initial_hash,choice=choice,steps=completed,
        success=bool(completed==steps and safe and tail>=100),success_4s=success4,
        continuous_goal_after4s=bool(steps>200 and completed==steps and success4 and continuous),
        safe=safe,max_target_slew_rad_s=max_slew,final=m,trace=trace)


def paired_block(pool,radii,seeds,steps):
    rows=list(pool.map(evaluate,[(r,s,steps) for r in radii for s in seeds]))
    for seed in seeds:
        if len({r['initial_hash'] for r in rows if r['seed']==seed})!=1:raise RuntimeError('paired state mismatch')
    baseline={r['seed']:r['success'] for r in rows if r['radius']==0}
    summary={str(radius):dict(runs=len(seeds),successes=sum(r['success'] for r in rows if r['radius']==radius),
        full_duration_runs=sum(r['steps']==steps for r in rows if r['radius']==radius),
        rescue_selections=sum(r['choice']['mode']=='rescue' for r in rows if r['radius']==radius),
        new_successes=sum(r['success'] and not baseline[r['seed']] for r in rows if r['radius']==radius),
        new_failures=sum(not r['success'] and baseline[r['seed']] for r in rows if r['radius']==radius),
        continuous_goal_after4s=sum(r['continuous_goal_after4s'] for r in rows if r['radius']==radius)) for radius in radii}
    return dict(results=rows,summary=summary,actual_control_steps=sum(r['steps'] for r in rows),seconds=steps*DT)


def select_radius(summary):
    # Each lost home success costs twice each new rescue. Baseline wins ties.
    radii=sorted(float(x) for x in summary)
    return max(radii,key=lambda x:(summary[str(x)]['new_successes']-2*summary[str(x)]['new_failures'],-x))


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--search',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    library=fit_library(args.search);np.savez_compressed(args.output/'trajectory_library.npz',**library)
    c=physical_contract(json.loads(args.contract.read_text()))
    c.update(stage='R24 initial-state trajectory retrieval, near-standing ONLY',training_method='nonparametric whole-trajectory library fit',
        initial_observation_size=50,feature_indices=FEATURES.tolist(),feature_scales=SCALES.tolist(),
        rescue_vs_home_distance_margin=.8,knots_s=[.2,.6,1.2],motor_target_period_s=DT,
        learned_neural_actor=False,default_controller_replaced=False,training_seed_list=library['seed'].tolist())
    contract=args.output/'controller_contract.json';contract.write_text(json.dumps(c,indent=2),encoding='utf-8')
    report=dict(paired_initial_states_verified=True,physics_unchanged=True,success_gate_unchanged=True,
        initialization_only_state_edits=True,auto_resets=0,simulation_only=True,hardware_readiness=False,
        library_rescue_anchors=int(library['rescue'].sum()),library_home_anchors=int((~library['rescue']).sum()),
        stage='near-standing recovery; NOT full fallen get-up',default_controller_replaced=False,
        hashes={str(f):digest(f) for f in (Path(__file__),args.output/'trajectory_library.npz',contract,args.contract)})
    output=args.output/'results.json'
    training=library['seed'].tolist();validation=list(range(370000,370040));short=list(range(380000,380100));long=list(range(390000,390020))
    if set(training)&set(validation+short+long):raise RuntimeError('evaluation seed contamination')
    radii=[0.,.15,.3,.6,1.2]
    with ProcessPoolExecutor(max_workers=4,initializer=init_worker,initargs=(str(contract),str(args.output/'trajectory_library.npz'))) as pool:
        report['selection_40']=paired_block(pool,radii,validation,200)
        selected=select_radius(report['selection_40']['summary']);report['selected_radius']=selected
        print('SELECTION:',json.dumps(report['selection_40']['summary']),'CHOSEN',selected,flush=True)
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        # Radius .3 diagnostic is declared BEFORE seeing held-out data. If baseline
        # wins selection this is only an exploratory candidate, not a adopted one.
        compare=sorted(set([0.,selected,.3]));report['exploratory_radius']=.3
        for name,seeds,steps in [('demonstration_4s',training,200),('independent_100',short,200),('independent_30s',long,1500)]:
            report[name]=paired_block(pool,compare,seeds,steps)
            output.write_text(json.dumps(report,indent=2),encoding='utf-8')
            print('TRAJECTORY POLICY:',name,json.dumps(report[name]['summary']),flush=True)
    report['complete']=True
    report['actual_control_steps']=sum(v['actual_control_steps'] for v in report.values() if isinstance(v,dict) and 'actual_control_steps' in v)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
