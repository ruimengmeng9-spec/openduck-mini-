"""Bounded six-parameter causal right-hip feedback on frozen R122 snapshot.

No frozen neural-feature rescaling, case-indexed controller or root truth input.
Same ReferenceEpisode reward, physical limits and full acceptance.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import train_getup_history_program_r122 as prior
from diagnostics.train_getup_joint_anchor_r102 import aggregate,write_json
from diagnostics.getup_independent_native import digest

ROOT=prior.ROOT
OUTPUT=ROOT/'outputs/getup_local_hip_r130_left_20261005'
JOINTS=('right_hip_yaw','right_hip_roll','right_hip_pitch')
WEIGHTS=NOMINAL=None


def local_feedback(current,nominal,sensor_ids,gains):
    current=np.asarray(current,dtype=np.float32);nominal=np.asarray(nominal,dtype=np.float32)
    gains=np.asarray(gains,dtype=float);ids=np.asarray(sensor_ids,dtype=int)
    if current.shape!=(55,) or nominal.shape!=(55,) or gains.shape!=(6,) or ids.shape!=(3,):raise ValueError('Sensor55, three joint indices and six gains required')
    if not np.isfinite(current).all() or not np.isfinite(nominal).all() or not np.isfinite(gains).all() or np.abs(gains).max()>2.:raise ValueError('Finite bounded input required')
    if len(set(ids))!=3 or np.any(ids<0) or np.any(ids>=14):raise ValueError('Three distinct native joint indices required')
    position=(current[6+ids]-nominal[6+ids]).astype(float)/.05
    velocity=(current[20+ids]-nominal[20+ids]).astype(float)/.05
    return -.18*np.tanh(gains[:3]*position+gains[3:]*velocity)


def merge_target(target,base,extra,ids,lower,upper):
    if not np.any(extra!=0.):return target
    result=target.copy()
    result[ids]=np.clip(base[ids]+np.clip(target[ids]-base[ids]+extra,-.18,.18),lower[ids],upper[ids])
    return result


def init_worker():
    global WEIGHTS,NOMINAL
    with np.load(prior.OUTPUT/'training/snapshot.npz',allow_pickle=False) as z:WEIGHTS={k:z[k].copy() for k in z.files}
    with np.load(prior.OUTPUT/'snapshot/case_None/trajectory.npz',allow_pickle=False) as z:NOMINAL=z['observations'].copy()


def evaluate(job):
    gains,case,full,directory,parity=job
    assert case in {None,*prior.program.TRAIN}
    env=prior.audit.HistoryReferenceEpisode(str(prior.program.SCENE),str(prior.program.STAND),prior.program.REFERENCE,230,full)
    env.reset(case);sim=env.sim
    actuator_ids=np.array([sim.model.actuator(n).id for n in JOINTS])
    sensor_ids=np.array([int(np.flatnonzero(sim.qadr==sim.model.joint(n).qposadr)[0]) for n in JOINTS])
    profile,knots,_=prior.scalar_program(WEIGHTS,env.preparation_sensors)
    original_step=sim.step_target;extras=[];summax=[]
    def controlled(target):
        k=env.controls
        extra=local_feedback(env.observe(),NOMINAL[k],sensor_ids,gains) if k<529 else np.zeros(3)
        adjusted=merge_target(target,env.targets[k],extra,actuator_ids,sim.lower,sim.upper)
        extras.append(extra.copy())
        summax.append(float(np.abs(adjusted[actuator_ids]-env.targets[k][actuator_ids]).max()))
        assert summax[-1]<=.18+1e-12
        return original_step(adjusted)
    sim.step_target=controlled;records=[]
    try:
        while True:
            obs=env.observe();action=prior.program.execute_program(WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    finally:sim.step_target=original_step
    row=step[5];row.update(gains=list(gains),full_path=full,
        success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12. and row['strict_tail_s']>=30.-1e-8) if full else bool(row['training_success']),
        maximum_combined_right_hip_correction_rad=max(summax),no_case_metadata_in_controller=True,
        root_edits_during_recovery=0)
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],
        qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],
        local_hip_extra_rad=extras,preparation_sensors=env.preparation_sensors)
    previous=prior.OUTPUT/'snapshot'/f'case_{case}'
    assert row['initial_hash']==json.loads((previous/'result.json').read_text())['initial_hash']
    if parity or case is None:
        with np.load(previous/'trajectory.npz',allow_pickle=False) as stored:
            equality={k:bool(np.array_equal(np.asarray(v),stored[k][:len(records)])) for k,v in arrays.items() if k not in ('local_hip_extra_rad','preparation_sensors')}
        assert all(equality.values());row['original_complete_or_prefix_trace_bitwise_equal']=equality
        if case is None:np.testing.assert_array_equal(np.array(extras),np.zeros((len(records),3)))
    np.savez_compressed(dest/'trajectory.npz',**arrays);write_json(dest/'result.json',row)
    return row


def group(pool,gains,cases,full,directory,parity=False):
    jobs=[(gains.tolist(),s,full,str(directory/f'case_{s}'),parity) for s in cases]
    report=aggregate(list(pool.map(evaluate,jobs)));write_json(directory/'results.json',report);return report


def rank(report):
    return bool(report['nominal_success']),report['physical_failures']==0,report['successes'],-report['physical_failures'],report['return_sum']


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=OUTPUT);p.add_argument('--smoke',action='store_true')
    p.add_argument('--generations',type=int,default=8);p.add_argument('--population',type=int,default=12);p.add_argument('--workers',type=int,default=6)
    args=p.parse_args();assert args.output.parent.resolve()==(ROOT/'outputs').resolve()
    assert 1<=args.workers<=6 and 1<=args.generations<=8 and 6<=args.population<=12
    args.output.mkdir(exist_ok=False);sources=args.output/'executed_sources';sources.mkdir()
    for name in (Path(__file__).name,'test_getup_local_hip_r130.py','launch_getup_local_hip_r130.py','audit_getup_head_coupling_r129.py',
        'train_getup_history_program_r122.py','probe_getup_sensor_history_r121.py','train_getup_program_r113.py',
        'getup_reference_env_r100.py','getup_independent_native.py','train_getup_fullpath_r27.py','validate_getup_fullpath_r27.py'):
        shutil.copy2(Path(__file__).with_name(name),sources/name)
    frozen=args.output/'frozen';frozen.mkdir()
    for src,name in [(prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:
        shutil.copy2(src,frozen/name);assert digest(src)==digest(frozen/name)
    write_json(args.output/'contract.json',dict(seed=230,smoke=args.smoke,generations=args.generations,population=args.population,workers=args.workers,
        hypothesis='Early trunk/right-hip collision and rapid leg response suggest testing current local hip position/velocity feedback rather than another head gain or frozen neural-feature multiplier',
        training_method='CEM of six local sensor feedback parameters on frozen R122 snapshot',gains_bound=2.,
        joints=JOINTS,position_normalization_rad=.05,velocity_normalization_rad_s=1.,
        combined_original_plus_new_hip_target_cap_rad=.18,home_feedback_unchanged=True,
        head_feedback_disabled=True,physical_rewards_acceptance_unchanged=True,no_case_input_or_lookup=True,
        training_cases=[None,*prior.program.TRAIN],training_controls=629,full_controls=2279,
        control_hz=50,physics_hz=500,entry_deadline_s=12,strict_standing_tail_s=30,
        short_tail_1s_not_acceptance=True,no_mid_episode_root_edits=True,reserved_qualification_never_loaded=True,
        automatic_independent_qualification=False,hardware_readiness=False,full_task_completed=False,
        hashes={str(f):digest(f) for f in [prior.program.SCENE,prior.program.STAND,prior.program.REFERENCE,*sources.iterdir(),*frozen.iterdir()]}))
    zero=np.zeros(6);selected=zero.copy();mean=zero.copy();std=np.full(6,.15);rng=np.random.default_rng(230)
    history=[];cache={};best=None;cases=[None,*prior.program.TRAIN]
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        parity=group(pool,zero,[None,769000,773004],True,args.output/'zero_parity',True)
        nonzero=group(pool,np.array([.03,-.02,.04,.01,-.01,.02]),[None,769000],True,args.output/'nonzero_smoke')
        print('R130_ZERO_AND_NOMINAL_PARITY_PASS',flush=True)
        if args.smoke:
            write_json(args.output/'results.json',dict(smoke=True,parity=parity,nonzero=nonzero,full_task_completed=False));return
        for generation in range(1,args.generations+1):
            proposals=[zero.copy(),selected.copy(),*np.clip(rng.normal(mean,std,(args.population-2,6)),-2.,2.)]
            reports=[];locations=[]
            for i,gains in enumerate(proposals):
                key=gains.tobytes().hex();folder=args.output/'training'/f'generation_{generation:04d}'/f'candidate_{i:02d}'
                if key not in cache:
                    report=group(pool,gains,cases,False,folder);cache[key]=(report,str(folder))
                report,location=cache[key];reports.append(report);locations.append(location)
                print('R130_CANDIDATE',generation,i,report['successes'],report['physical_failures'],flush=True)
            ordered=sorted(range(len(proposals)),key=lambda i:rank(reports[i]),reverse=True);winner=ordered[0]
            if best is None or rank(reports[winner])>rank(best):selected=proposals[winner].copy();best=reports[winner]
            elite=np.stack([proposals[i] for i in ordered[:3]])
            mean=.6*mean+.4*elite.mean(0);std=np.clip(.6*std+.4*elite.std(0),.015,.25)
            history.append(dict(generation=generation,proposals=[g.tolist() for g in proposals],reports=reports,closed_trial_directories=locations,
                best_gains=selected.tolist(),mean=mean.tolist(),std=std.tolist()))
            checkpoint=args.output/'checkpoints'/f'generation_{generation:04d}';checkpoint.mkdir(parents=True,exist_ok=False)
            np.savez_compressed(checkpoint/'state.npz',gains=selected,mean=mean,std=std)
            write_json(checkpoint/'rng.json',rng.bit_generator.state);write_json(checkpoint/'history.json',history)
            write_json(args.output/'progress.json',dict(closed_generation=generation,generations=args.generations,best=best,gains=selected.tolist()))
        write_json(args.output/'training_closed.json',dict(history=history,rng=rng.bit_generator.state,best=best,gains=selected.tolist(),short_label_only=True))
        candidate=group(pool,selected,cases,True,args.output/'development_candidate')
        baseline=group(pool,zero,cases,True,args.output/'development_baseline',True)
        assert [r['initial_hash'] for r in candidate['rows']]==[r['initial_hash'] for r in baseline['rows']]
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and candidate['physical_failures']==0)
        write_json(args.output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,gains=selected.tolist(),
            original_development_gate=gate,expanded_development_run=False,independent_qualification_run=False,
            full_task_completed=False,hardware_readiness=False))
        print('R130_FULL_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)


if __name__=='__main__':main()
