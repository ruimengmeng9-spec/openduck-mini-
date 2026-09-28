"""Simulation-only native search for independent, phase-dependent leg corrections.

Frozen R2 remains outside the optimizer. Native 50 Hz targets retain joint clamps
and slew limits. Search artifacts are NOT compatible with the production agent.
"""
import argparse
import hashlib
import json
import math
import time
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import mujoco
import numpy as np
import onnxruntime as ort
from open_duck_agent.duck_sim import DuckSimulation
from diagnostics.reference_residual_policy import ReferenceResidualPolicy


class CpuActor:
    def __init__(self,path):
        options = ort.SessionOptions()
        options.intra_op_num_threads=1
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.input_size=self.session.get_inputs()[0].shape[-1]

    def infer(self,obs):
        return self.session.run(None,{'obs':np.asarray(obs,dtype=np.float32)[None,:]})[0][0]


def heading(q):
    w,x,y,z=q[3:7]
    return math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def phase_delta(weights,phase,ramp,feedback=None,error=0.):
    """Six independent hips; compensate hip pitch at ankle for foot pitch."""
    vector=np.array([1.,phase[0],phase[1]])
    raw=np.asarray(weights).reshape(6,3)@vector
    if feedback is not None:
        raw=raw+np.asarray(feedback)*math.sin(error)
    correction=np.tanh(raw)*np.array([.06,.035,.06,.06,.035,.06])*ramp
    delta=np.zeros(14)
    delta[[0,1,2,9,10,11]]=correction
    delta[4]=-correction[2]
    delta[13]=-correction[5]
    return delta


def body_pitch(q):
    w,x,y,z=q[3:7]
    return math.asin(float(np.clip(2*(w*y-z*x),-1.,1.)))


def phase_pitch(weights,phase):
    phi=math.atan2(float(phase[1]),float(phase[0]))
    basis=[1.]+[v for harmonic in (1,2,3) for v in (math.cos(harmonic*phi),math.sin(harmonic*phi))]
    return float(np.asarray(weights)@np.asarray(basis))


def balance_features(pitch,rate,deadband=.10):
    # Ignore the small normal gait lean, but react before the fall threshold.
    return np.array([math.copysign(max(0.,abs(pitch)-deadband),pitch)/.2,
                     np.clip(rate,-4.,4.)/.8])


def balance_delta(gains,pitch,rate,ramp,deadband=.10):
    value=.06*math.tanh(float(np.asarray(gains)@balance_features(pitch,rate,deadband)))*ramp
    delta=np.zeros(14)
    delta[[2,11]]=value
    delta[[4,13]]=-value
    return delta


class NativeRollout:
    def __init__(self,root):
        self.root=Path(root)
        self.model_path=self.root/'training/backward_reference_residual_r2/final.onnx'
        robot=DuckSimulation(self.root/'projects/Open_Duck_Playground',self.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',output_root=self.root/'outputs/phase_search_r5/traces',warmup_s=0.)
        self.sim=robot.sim
        self.robot=robot
        self.actor=CpuActor(self.model_path)
        self.dt=self.sim.sim_dt*self.sim.decimation

    def run(self,weights,seed,duration,save=None,feedback=None,corrector=None,up_gate=None,balance=None,pitch_template=None,balance_deadband=.10):
        s=self.sim
        mujoco.mj_resetData(s.model,s.data)
        s.data.qpos[:]=s.model.keyframe('home').qpos
        s.data.qvel[:]=np.random.default_rng(seed).uniform(-.02,.02,s.model.nv)
        s.data.ctrl[:]=s.default_actuator
        s.motor_targets=np.array(s.default_actuator).copy()
        s.prev_motor_targets=s.motor_targets.copy()
        s.last_action=np.zeros(14,dtype=np.float32)
        s.last_last_action=s.last_action.copy()
        s.last_last_last_action=s.last_action.copy()
        s.imitation_i=0.
        s.imitation_phase=np.array([1.,0.],dtype=np.float32)
        mujoco.mj_forward(s.model,s.data)
        initial=s.get_floating_base_qpos(s.data.qpos).copy()
        h0=heading(initial)
        previous=h0
        accumulated=0.
        previous_pitch=body_pitch(initial)
        previous_nominal_pitch=0.
        decoder=ReferenceResidualPolicy(self.actor,s,.12,1.)
        angles=[]
        velocities=[]
        qposes=[]
        times=[]
        min_up=1.
        fallen=False
        for k in range(round(duration/self.dt)):
            obs=np.asarray(s.get_obs(s.data,[-.074,0,0,0,0,0,0]),dtype=np.float32)
            base=s.default_actuator+s.action_scale*decoder.infer(obs)
            ramp=min(float(s.data.time),1.)
            error=heading(s.get_floating_base_qpos(s.data.qpos))-h0
            quat=s.get_floating_base_qpos(s.data.qpos)[3:7]
            current_up=1-2*(quat[1]**2+quat[2]**2)
            pitch=body_pitch(s.get_floating_base_qpos(s.data.qpos))
            pitch_rate=(pitch-previous_pitch)/self.dt
            previous_pitch=pitch
            if pitch_template is not None:
                nominal_pitch=ramp*phase_pitch(pitch_template,s.imitation_phase)
                pitch-=nominal_pitch
                pitch_rate-=(nominal_pitch-previous_nominal_pitch)/self.dt
                previous_nominal_pitch=nominal_pitch
            if corrector is None:
                delta=phase_delta(weights,s.imitation_phase,ramp,feedback,error)
                if up_gate:
                    high,low=up_gate
                    delta*=np.clip((current_up-low)/(high-low),0.,1.)
                if balance is not None:
                    delta+=balance_delta(balance,pitch,pitch_rate,ramp,balance_deadband)
            else:
                features=[*s.imitation_phase,math.sin(error),ramp]
                if up_gate:
                    features.append(current_up)
                if balance is not None:
                    features.extend(balance_features(pitch,pitch_rate,balance_deadband))
                if len(features)!=corrector.input_size:
                    raise ValueError('controller feature contract mismatch')
                delta=corrector.infer(features)
            if not np.isfinite(delta).all() or delta.shape!=(14,):
                raise ValueError('invalid correction output')
            target=np.clip(base+delta,decoder.lower,decoder.upper)
            motor_action=((target-s.default_actuator)/s.action_scale).astype(np.float32)
            slew=np.clip(target,s.prev_motor_targets-s.max_motor_velocity*self.dt,s.prev_motor_targets+s.max_motor_velocity*self.dt)
            s.data.ctrl[:]=slew
            s.motor_targets=slew.copy()
            s.prev_motor_targets=slew.copy()
            s.last_last_last_action=s.last_last_action.copy()
            s.last_last_action=s.last_action.copy()
            s.last_action=motor_action.copy()
            for _ in range(s.decimation):
                mujoco.mj_step(s.model,s.data)
            s.imitation_i=(s.imitation_i+1)%s.PRM.nb_steps_in_period
            phi=s.imitation_i/s.PRM.nb_steps_in_period*2*np.pi
            s.imitation_phase=np.array([np.cos(phi),np.sin(phi)],dtype=np.float32)
            q=s.get_floating_base_qpos(s.data.qpos).copy()
            hn=heading(q)
            accumulated+=math.atan2(math.sin(hn-previous),math.cos(hn-previous))
            previous=hn
            up=1-2*(q[4]**2+q[5]**2)
            min_up=min(min_up,up)
            angles.append(accumulated)
            velocities.append(self.robot._sensor('local_linvel').copy())
            if save is not None:
                qposes.append(s.data.qpos.copy())
                times.append(float(s.data.time))
            fallen=bool(up<.5 or q[2]<.08 or not np.isfinite(s.data.qpos).all() or not np.isfinite(s.data.qvel).all())
            if fallen:
                break
        elapsed=len(angles)*self.dt
        q=s.get_floating_base_qpos(s.data.qpos)
        dx,dy=q[:2]-initial[:2]
        axial=(math.cos(h0)*dx+math.sin(h0)*dy)/elapsed
        lateral=-math.sin(h0)*dx+math.cos(h0)*dy
        tail=np.asarray(velocities)[min(len(velocities)-1,round(1/self.dt)):]
        # Explicit progress and heading metrics avoid optimizing "stand still".
        cost=4*np.mean(np.square(angles))+2*angles[-1]**2+8*(lateral/(duration*.074))**2
        cost+=1000*(axial+.074)**2+2000*max(0.,axial+.05)**2
        cost+=.01*np.mean(np.square(weights))
        if fallen:
            cost+=1000+100*(duration-elapsed)
        row=dict(seed=seed,duration_s=elapsed,fallen=fallen,speed_mps=float(axial),lateral_m=float(lateral),yaw_change_deg=math.degrees(accumulated),heading_rms_deg=math.degrees(float(np.sqrt(np.mean(np.square(angles))))),minimum_up_z=float(min_up),mean_local_velocity_mps=tail.mean(axis=0).tolist(),cost=float(cost))
        if save is not None:
            np.savez_compressed(save,qpos=np.asarray(qposes),time=np.asarray(times),yaw=np.asarray(angles))
        return row


def worker_init(root):
    global worker_engine
    worker_engine=NativeRollout(root)


def worker_run(task):
    weights,feedback,balance,seeds,duration,template,deadband=task
    return [worker_engine.run(weights,s,duration,feedback=feedback,balance=balance,pitch_template=template,balance_deadband=deadband) for s in seeds]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',default='/data/shijinsheng/open_duck')
    p.add_argument('--generations',type=int,default=8)
    p.add_argument('--population',type=int,default=20)
    p.add_argument('--duration',type=float,default=10.)
    p.add_argument('--seeds',default='0,1')
    p.add_argument('--validate',type=Path)
    p.add_argument('--seed-start',type=int,default=0)
    p.add_argument('--seed-count',type=int,default=5)
    p.add_argument('--resume',type=Path)
    p.add_argument('--feedback-base',type=Path,help='freeze a learned phase pattern and search six heading-feedback gains')
    p.add_argument('--balance-base',type=Path,help='freeze phase and heading feedback; search two body-pitch stabilizer gains')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--corrector-onnx',type=Path,help='independent exported-controller native gate')
    a=p.parse_args()
    if not 0 < a.duration <= 60 or not 1 <= a.workers <= 4:
        raise ValueError('duration/worker limits exceeded')
    if os.environ.get('REFERENCE_DX')!='-0.0925' or os.environ.get('REFERENCE_DX_INTERPOLATION')!='1':
        raise ValueError('fixed reference configuration required')
    a.output.mkdir(parents=True,exist_ok=True)
    engine=NativeRollout(a.root)
    if a.validate:
        artifact=json.loads(a.validate.read_text())
        if hashlib.sha256(engine.model_path.read_bytes()).hexdigest()!=artifact['baseline_sha256']:
            raise ValueError('baseline hash mismatch')
        weights=np.asarray(artifact['weights'])
        if weights.shape!=(18,) or not np.isfinite(weights).all() or np.max(abs(weights))>2.:
            raise ValueError('invalid phase controller weights')
        feedback=artifact.get('feedback_weights')
        if feedback is not None and (len(feedback)!=6 or not np.isfinite(feedback).all() or np.max(np.abs(feedback))>3.):
            raise ValueError('invalid feedback weights')
        balance=artifact.get('balance_weights')
        if balance is not None and (len(balance)!=2 or not np.isfinite(balance).all() or np.max(np.abs(balance))>3.):
            raise ValueError('invalid balance weights')
        template=artifact.get('pitch_template_weights')
        deadband=float(artifact.get('balance_deadband_rad',.10))
        if template is not None and (len(template)!=7 or not np.isfinite(template).all() or max(abs(np.asarray(template)))>.5):
            raise ValueError('invalid pitch template')
        if not 0 <= deadband <= .2:
            raise ValueError('invalid balance deadband')
        if template is not None and abs(artifact['pitch_template_period_s']-engine.dt*engine.sim.PRM.nb_steps_in_period)>1e-6:
            raise ValueError('pitch template phase period mismatch')
        corrector=None
        if a.corrector_onnx:
            if hashlib.sha256(a.corrector_onnx.read_bytes()).hexdigest()!=artifact['onnx_sha256']:
                raise ValueError('corrector checksum mismatch')
            corrector=CpuActor(a.corrector_onnx)
        rows=[engine.run(weights,s,a.duration,a.output/f'seed_{s}.npz',feedback,corrector,artifact.get('up_gate'),balance,template,deadband) for s in range(a.seed_start,a.seed_start+a.seed_count)]
        (a.output/'results.json').write_text(json.dumps({'rows':rows,'controller':str(a.validate),'corrector_onnx':str(a.corrector_onnx) if a.corrector_onnx else None},indent=2))
        for r in rows:
            print(json.dumps(r),flush=True)
        return
    seeds=[int(x) for x in a.seeds.split(',')]
    fixed=None
    fixed_feedback=None
    template=None
    template_metadata={}
    deadband=.10
    mean=np.zeros(18)
    if a.balance_base and a.feedback_base:
        raise ValueError('choose exactly one optimization mode')
    if a.balance_base:
        restored=json.loads(a.balance_base.read_text())
        fixed=np.array(restored['weights'])
        fixed_feedback=np.array(restored['feedback_weights'])
        template=restored.get('pitch_template_weights')
        template_metadata={k:v for k,v in restored.items() if k.startswith('pitch_template_')}
        deadband=float(restored.get('balance_deadband_rad',.10))
        mean=np.zeros(2)
    if a.feedback_base:
        fixed=np.array(json.loads(a.feedback_base.read_text())['weights'])
        mean=np.zeros(6)
    if a.resume:
        restored=json.loads(a.resume.read_text())
        key='balance_weights' if a.balance_base else 'feedback_weights' if fixed is not None else 'weights'
        mean=np.array(restored[key]).reshape(mean.size)
    std=np.full(mean.size,.8 if fixed is not None else .35)
    rng=np.random.default_rng(87)
    best_score=float('inf')
    start=time.monotonic()
    baseline=[engine.run(np.zeros(18) if fixed is None else fixed,s,a.duration,feedback=fixed_feedback) for s in seeds]
    print('ZERO BASELINE:',json.dumps(baseline),flush=True)
    records=[]
    pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(a.root,)) if a.workers>1 else None
    for gen in range(a.generations):
        limit=3. if fixed is not None else 2.
        candidates=np.clip(mean+rng.normal(size=(a.population,mean.size))*std,-limit,limit)
        candidates[0]=mean
        tasks=[(candidate if fixed is None else fixed,fixed_feedback if a.balance_base else None if fixed is None else candidate,candidate if a.balance_base else None,seeds,a.duration,template,deadband) for candidate in candidates]
        evaluated=pool.map(worker_run,tasks) if pool else (worker_run_direct(engine,task) for task in tasks)
        for candidate,rows in zip(candidates,evaluated):
            weights=candidate if fixed is None else fixed
            score=float(np.mean([r['cost'] for r in rows])+.3*max(r['cost'] for r in rows))
            records.append(dict(generation=gen,score=score,weights=candidate.tolist(),rows=rows))
            if score<best_score:
                best_score=score
                artifact=dict(controller_type='phase_leg_correction_v1',simulation_only=True,baseline_sha256=hashlib.sha256(engine.model_path.read_bytes()).hexdigest(),reference_dx=-.0925,reference_interpolation=True,residual_gain=.12,ramp_s=1.,control_dt=engine.dt,weights=weights.tolist(),score=score,rows=rows)
                if a.balance_base:
                    artifact.update(controller_type='phase_heading_balance_v1',feedback_weights=fixed_feedback.tolist(),balance_weights=candidate.tolist(),balance_deadband_rad=deadband,balance_contract='deadband_pitch_scaled_0.2;finite_difference_rate_clipped_4_scaled_0.8;gain_0.06_rad',**template_metadata)
                elif fixed is not None:
                    artifact.update(controller_type='phase_leg_feedback_v1',feedback_weights=candidate.tolist())
                (a.output/'best.json').write_text(json.dumps(artifact,indent=2))
                print('BEST:',gen,round(score,4),json.dumps(rows),flush=True)
        generation=records[-a.population:]
        elite=sorted(generation,key=lambda r:r['score'])[:max(4,a.population//4)]
        matrix=np.array([r['weights'] for r in elite])
        mean=.2*mean+.8*matrix.mean(axis=0)
        std=np.maximum(.04,.25*std+.75*matrix.std(axis=0))
        (a.output/'search.json').write_text(json.dumps(dict(baseline=baseline,evaluations=records,elapsed_seconds=time.monotonic()-start),indent=2))
        print('GENERATION COMPLETE:',gen,'best:',best_score,'elapsed:',round(time.monotonic()-start,1),flush=True)
    if pool:
        pool.shutdown()


def worker_run_direct(engine,task):
    weights,feedback,balance,seeds,duration,template,deadband=task
    return [engine.run(weights,s,duration,feedback=feedback,balance=balance,pitch_template=template,balance_deadband=deadband) for s in seeds]


if __name__=='__main__':
    main()
