"""Single fixed causal paired-foot command projection; simulator only."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import traceback
import mujoco
import numpy as np
from diagnostics import train_getup_full_episode_pg_r181 as old

prior=old.prior;local=old.local;selector=old.selector;planner=old.planner;ROOT=old.ROOT
OUTPUT=ROOT/'outputs/getup_contact_projection_r183b_left_20261010'
SMOKE=ROOT/'outputs/getup_contact_projection_r183b_smoke_20261010'
CASES=old.CASES;SEED=283;MODEL=None;NOMINAL_PLANNED=None
ITERATIONS=3;ROTATION_LENGTH_M=.05;FD_RAD=1e-6;RCOND=1e-10
STEPS=(1.,.5,.25,.125,.0625)


def rotation_log(rotation,reference):
    if np.array_equal(rotation,reference):return np.zeros(3)
    quat=np.empty(4);mujoco.mju_mat2Quat(quat,(rotation@reference.T).reshape(9))
    if quat[0]<0:quat=-quat
    result=np.empty(3);mujoco.mju_quat2Vel(result,quat,1.)
    return result


class RelativeFootGeometry:
    """Private canonical root, head home; planned joints only, no dynamics."""
    def __init__(self,model):
        self.model=model;self.data=mujoco.MjData(model)
        self.qadr=model.jnt_qposadr[model.actuator_trnid[:,0]].copy()
        self.canonical=model.keyframe('home').qpos.copy();self.canonical[:7]=[0,0,0,1,0,0,0]
        self.home=self.canonical[self.qadr].copy()
        self.legs=np.array([model.actuator(s+'_'+j).id for s in ('left','right') for j in ('hip_yaw','hip_roll','hip_pitch','knee','ankle')])
        self.feet=[model.body(n).id for n in ('foot_assembly','foot_assembly_2')]
        np.testing.assert_array_equal(self.legs,[0,1,2,3,4,9,10,11,12,13])

    def pose(self,planned):
        q=np.asarray(planned,dtype=float)
        if q.shape!=(14,) or not np.isfinite(q).all():raise ValueError('Finite planned14 targets required')
        self.data.qpos[:]=self.canonical
        self.data.qpos[self.qadr[self.legs]]=q[self.legs]
        mujoco.mj_kinematics(self.model,self.data)
        left,right=self.feet
        rl=self.data.xmat[left].reshape(3,3)
        p=rl.T@(self.data.xpos[right]-self.data.xpos[left])
        r=rl.T@self.data.xmat[right].reshape(3,3)
        return p.copy(),r.copy()

    def anchor(self,planned,nominal):
        p,r=self.pose(planned);n,s=self.pose(nominal)
        rotation=np.eye(3) if np.array_equal(planned,nominal) else r@s.T
        return np.concatenate([p-n,rotation.reshape(9)])

    def residual(self,planned,nominal,anchor):
        p,r=self.pose(planned);n,s=self.pose(nominal)
        # Exact identity is not a tolerance deadband.
        desired=s if np.array_equal(anchor[3:].reshape(3,3),np.eye(3)) else anchor[3:].reshape(3,3)@s
        return np.concatenate([p-(n+anchor[:3]),ROTATION_LENGTH_M*rotation_log(r,desired)])

    def jacobian(self,planned,nominal,anchor):
        result=np.empty((6,10))
        for j,index in enumerate(self.legs):
            a=planned.copy();b=planned.copy();a[index]+=FD_RAD;b[index]-=FD_RAD
            result[:,j]=(self.residual(a,nominal,anchor)-self.residual(b,nominal,anchor))/(2*FD_RAD)
        return result


def project(planned,nominal,contacts,anchor,lower,upper,geometry,enabled):
    q=np.asarray(planned,dtype=float);n=np.asarray(nominal,dtype=float)
    c=np.asarray(contacts,dtype=float);a=np.asarray(anchor,dtype=float)
    lo=np.asarray(lower,dtype=float);hi=np.asarray(upper,dtype=float)
    if any(v.shape!=(14,) for v in (q,n,lo,hi)) or c.shape!=(2,) or a.shape!=(12,):raise ValueError('Invalid causal projection shape')
    if not all(np.isfinite(v).all() for v in (q,n,c,a,lo,hi)) or np.any(lo[geometry.legs]>hi[geometry.legs]):raise ValueError('Nonfinite or invalid projection bounds')
    if np.any(c<0) or np.any(c>1):raise ValueError('Native contact flags required')
    trace=np.zeros((ITERATIONS+1,3));jac=np.zeros((6,10));active=bool(enabled and np.all(c>0))
    error=geometry.residual(q,n,a);trace[:,0]=np.linalg.norm(error)
    if not active or not np.any(error):return planned,error,jac,trace,active
    result=q.copy()
    for iteration in range(ITERATIONS):
        jac=geometry.jacobian(result,n,a)
        delta=-np.linalg.pinv(jac,rcond=RCOND)@error
        accepted=0.
        for scale in STEPS:
            trial=result.copy();trial[geometry.legs]=np.clip(result[geometry.legs]+scale*delta,lo[geometry.legs],hi[geometry.legs])
            e=geometry.residual(trial,n,a)
            if np.linalg.norm(e)<np.linalg.norm(error):
                result=trial;error=e;accepted=scale;break
        trace[iteration+1]=[np.linalg.norm(error),accepted,np.linalg.matrix_rank(jac,tol=RCOND)]
    np.testing.assert_array_equal(result[5:9],q[5:9])
    assert np.all(result[geometry.legs]>=lo[geometry.legs]) and np.all(result[geometry.legs]<=hi[geometry.legs])
    if np.array_equal(result,q):return planned,error,jac,trace,active
    return result,error,jac,trace,active


def init_worker(frozen):
    global MODEL,NOMINAL_PLANNED
    old.init_worker(frozen);MODEL=old.MODEL
    with np.load(Path(frozen)/'nominal_sensor_trajectory.npz',allow_pickle=False) as z:NOMINAL_PLANNED=z['applied'].copy()


def evaluate(job):
    enabled,case,directory,parity=job
    env=local.prior.audit.HistoryReferenceEpisode(str(local.prior.program.SCENE),str(local.prior.program.STAND),local.prior.program.REFERENCE,SEED,True)
    env.reset(case);sim=env.sim;fingerprint=prior.physics_hash(sim);geometry=RelativeFootGeometry(sim.model)
    ids=np.array([sim.model.actuator(n).id for n in local.JOINTS]);sensors=np.array([9,10,11])
    gains,choice,logits=selector.old.predict(MODEL,env.preparation_sensors[-1])
    profile,knots,_=local.prior.scalar_program(local.WEIGHTS,env.preparation_sensors)
    original=sim.step_target;anchor=None;records=[]
    signals={k:[] for k in ('projection_extra_rad','relative_foot_residual_m_equivalent','projection_jacobian','projection_trace','contact_projection_active','causal_planned_anchor','local_hip_extra_rad','planned_before_integration_rad','same_state_unperturbed_planned_rad','original_double_pre_slew_target_rad','adjusted_double_pre_slew_target_rad')}
    maxima=[];last_inputs={};initial_sensor=env.observe().copy()
    def controlled(target):
        nonlocal anchor
        k=env.controls;observed=env.observe().copy()
        last_inputs.update(control=np.asarray(k),current_native55=observed.copy(),original_requested_target=target.copy(),previous_applied=sim.prev.copy(),original_reference=env.targets[k].copy())
        np.testing.assert_array_equal(observed[34:48],(sim.prev-sim.home).astype(np.float32))
        frozen=local.local_feedback(observed,local.NOMINAL[k],sensors,gains) if k<529 else np.zeros(3)
        fixed=local.merge_target(target,env.targets[k],frozen,ids,sim.lower,sim.upper)
        base=planner.apply_limits(fixed,sim.prev,sim.lower,sim.upper)
        adjusted=fixed;error=np.zeros(6);jac=np.zeros((6,10));trace=np.zeros((ITERATIONS+1,3));active=False
        if k==0:anchor=geometry.anchor(base,NOMINAL_PLANNED[k])
        elif k<529:
            # Use precisely the original planner to intersect original reference,
            # joint and slew boxes. These boxes are not relaxed or searched.
            raw_lo=np.maximum(env.targets[k]-.18,sim.lower);raw_hi=np.minimum(env.targets[k]+.18,sim.upper)
            lo=np.maximum(raw_lo,planner.apply_limits(raw_lo,sim.prev,sim.lower,sim.upper))
            hi=np.minimum(raw_hi,planner.apply_limits(raw_hi,sim.prev,sim.lower,sim.upper))
            last_inputs.update(base_planned=base.copy(),lower_projection_box=lo.copy(),upper_projection_box=hi.copy(),anchor=anchor.copy())
            feasible=bool(np.all(lo[geometry.legs]<=hi[geometry.legs]) and np.all(base[geometry.legs]>=lo[geometry.legs]) and np.all(base[geometry.legs]<=hi[geometry.legs]))
            if feasible:
                projected,error,jac,trace,active=project(base,NOMINAL_PLANNED[k],observed[48:50],anchor,lo,hi,geometry,enabled)
                if projected is not base:
                    adjusted=projected.copy();adjusted[5:9]=fixed[5:9]
        else:anchor=np.zeros(12)
        wanted=planner.apply_limits(adjusted,sim.prev,sim.lower,sim.upper)
        extra=wanted-base
        assert np.abs(adjusted-env.targets[k]).max()<=.18+1e-12
        if k==0:np.testing.assert_array_equal(extra,np.zeros(14));assert adjusted is fixed
        values=(extra,error,jac,trace,active,anchor,frozen,wanted,base,fixed,adjusted)
        for key,value in zip(signals,values):signals[key].append(np.asarray(value).copy())
        maxima.append(float(np.abs(adjusted-env.targets[k]).max()))
        applied=original(adjusted);np.testing.assert_array_equal(applied,wanted);np.testing.assert_array_equal(sim.prev,wanted)
        return applied
    sim.step_target=controlled
    try:
        while True:
            obs=env.observe();action=local.prior.program.execute_program(local.WEIGHTS,obs,env.controls,profile,knots)
            step=env.step(action,auto_reset=False)
            records.append((obs.copy(),action.copy(),sim.data.time,sim.data.qpos.copy(),sim.data.qvel.copy(),sim.prev.copy(),env.tail>0))
            if step[2]:break
    except Exception:
        # Preserve every completed frame and the exact failed causal planner
        # inputs even on unexpected assertions; no success/validity relabeling.
        dest=Path(directory);dest.mkdir(parents=True,exist_ok=False)
        partial=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],initial_sensor=initial_sensor,preparation_sensors=env.preparation_sensors,**signals)
        np.savez_compressed(dest/'partial_trajectory.npz',**partial)
        np.savez_compressed(dest/'failed_causal_inputs.npz',**last_inputs)
        local.write_json(dest/'failure.json',dict(execution_error=True,controls_completed=len(records),terminal_result_saved=False,original_physical_labels_not_reclassified=True,traceback=traceback.format_exc(),source_sha256=prior.digest(Path(__file__))))
        raise
    finally:sim.step_target=original
    row=step[5]
    arrays=dict(observations=[r[0] for r in records],normalized_residual=[r[1] for r in records],time=[r[2] for r in records],qpos=[r[3] for r in records],qvel=[r[4] for r in records],applied=[r[5] for r in records],strict=[r[6] for r in records],preparation_sensors=env.preparation_sensors,**signals)
    reference=selector.OUTPUT/'candidate'/f'case_{case}';previous=json.loads((reference/'result.json').read_text());assert row['initial_hash']==previous['initial_hash']
    with np.load(reference/'trajectory.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(np.asarray(arrays['applied'])[0],z['applied'][0])
        if parity or case is None:
            equal={key:bool(np.array_equal(np.asarray(arrays[key]),z[key])) for key in z.files}
            assert all(equal.values()),equal;assert row['peaks']==previous['peaks'];row['complete_R157_parity']=equal
    if case is None:np.testing.assert_array_equal(np.asarray(signals['projection_extra_rad']),np.zeros((row['controls'],14)))
    row.update(enabled=enabled,base_choice=choice,base_gains=gains.tolist(),initial_sensor_logits=logits.tolist(),success=bool(row['valid'] and row['controls']==2279 and row['entry_time_s'] is not None and row['entry_time_s']<=12 and row['strict_tail_s']>=30-1e-8),full_path=True,maximum_combined_14_joint_correction_rad=max(maxima),maximum_projection_extra_rad=float(np.abs(signals['projection_extra_rad']).max()),active_contact_controls=int(np.count_nonzero(signals['contact_projection_active'])),physics_sha256=fingerprint,physics_unchanged=fingerprint==prior.physics_hash(sim),no_root_truth_or_case_metadata_in_controller=True,root_edits_during_recovery=0)
    assert row['physics_unchanged']
    dest=Path(directory);dest.mkdir(parents=True,exist_ok=False);np.savez_compressed(dest/'trajectory.npz',**arrays);local.write_json(dest/'result.json',row)
    return row


def group(pool,enabled,cases,path,parity=False):
    report=local.aggregate(list(pool.map(evaluate,[(enabled,case,str(path/f'case_{case}'),parity) for case in cases])))
    local.write_json(path/'results.json',report);return report


def compare_smoke(output):
    for name,cases in [('zero_parity',[None,769000,773004]),('projection_smoke',[None,769002,773004])]:
        for case in cases:
            fresh=output/name/f'case_{case}';previous=SMOKE/name/f'case_{case}'
            with np.load(fresh/'trajectory.npz',allow_pickle=False) as a,np.load(previous/'trajectory.npz',allow_pickle=False) as b:
                assert a.files==b.files
                for key in a.files:np.testing.assert_array_equal(a[key],b[key],err_msg=key)
            x=json.loads((fresh/'result.json').read_text());y=json.loads((previous/'result.json').read_text());assert x['initial_hash']==y['initial_hash'] and x['peaks']==y['peaks'] and x['success']==y['success']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args();output=SMOKE if args.smoke else OUTPUT
    assert not output.exists() and shutil.disk_usage(ROOT).free>12*1024**3
    output.mkdir();src=output/'executed_sources';src.mkdir()
    for name,module in list(sys.modules.items()):
        f=getattr(module,'__file__',None)
        if name.startswith('diagnostics.') and f and Path(f).suffix=='.py':shutil.copy2(f,src/Path(f).name)
    for name in ('test_getup_contact_projection_r183b.py','launch_getup_contact_projection_r183b.py'):shutil.copy2(Path(__file__).with_name(name),src/name)
    frozen=output/'frozen';frozen.mkdir()
    for source,name in [(selector.OUTPUT/'training/model.npz','initial_selector.npz'),(local.prior.OUTPUT/'training/snapshot.npz','snapshot.npz'),(local.prior.OUTPUT/'snapshot/case_None/trajectory.npz','nominal_sensor_trajectory.npz')]:shutil.copy2(source,frozen/name);assert prior.digest(source)==prior.digest(frozen/name)
    local.write_json(output/'contract.json',dict(seed=SEED,simulation_only=True,smoke=args.smoke,controller='One fixed three-iteration paired relative-foot pose command projection in current double contact; minimum-norm numerical Jacobian corrections with bounded backtracking, no learned gain/axis/window/parameter grid.',inputs='Current causal native foot contact flags, already planned original target, causal control-zero planned anchor, fixed same-phase nominal planned targets and original joint/reference/slew bounds; no actual root or future actual state.',iterations=ITERATIONS,rotation_length_m=ROTATION_LENGTH_M,finite_difference_rad=FD_RAD,pseudoinverse_rcond=RCOND,fixed_backtracking=STEPS,workers=6,formal_dynamic_attempt_budget=56,independent_smoke_attempts=6,storage_budget_gb=2,reserve_gb=10,no_learning_or_new_reward=True,nominal_first_home_exact_original=True,qualification_never_loaded=True,full_task_completed=False,hardware_readiness=False,hashes={str(f):prior.digest(f) for f in [*src.iterdir(),*frozen.iterdir(),local.prior.program.SCENE,local.prior.program.STAND,local.prior.program.REFERENCE]}))
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(frozen,)) as pool:
        parity=group(pool,False,[None,769000,773004],output/'zero_parity',True)
        projection=group(pool,True,[None,769002,773004],output/'projection_smoke')
        if not args.smoke:compare_smoke(output)
        local.write_json(output/'startup_closed.json',dict(parity=parity,projection=projection,independent_smoke_bitwise_equal=not args.smoke,terminal_result_saved=False,training_generations_saved=0))
        if args.smoke:
            local.write_json(output/'results.json',dict(smoke=True,parity=parity,projection=projection,terminal_result_saved=True,full_task_completed=False,hardware_readiness=False));return
        assert shutil.disk_usage(ROOT).free>10*1024**3
        baseline=group(pool,False,CASES,output/'development_baseline',True)
        candidate=group(pool,True,CASES,output/'development_candidate')
        assert [r['initial_hash'] for r in baseline['rows']]==[r['initial_hash'] for r in candidate['rows']]
        local.write_json(output/'training_closed.json',dict(fixed_controller=True,learned_parameters=0,search_trials=0,formal_dynamic_attempts=56,rng=np.random.default_rng(SEED).bit_generator.state))
        gate=bool(candidate['nominal_success'] and candidate['successes']>=22 and candidate['successes']>baseline['successes'] and not candidate['physical_failures'])
        local.write_json(output/'results.json',dict(smoke=False,candidate=candidate,baseline=baseline,original_development_gate=gate,terminal_result_saved=True,formal_dynamic_attempts=56,full_task_completed=False,hardware_readiness=False,expanded_development_run=False,independent_qualification_run=False))
        print('R183b_FIXED_PROJECTION_DEVELOPMENT',candidate['successes'],baseline['successes'],candidate['physical_failures'],flush=True)


if __name__=='__main__':main()
