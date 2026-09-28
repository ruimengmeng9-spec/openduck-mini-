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
from unittest.mock import patch
from pathlib import Path
import mujoco
import numpy as np
import onnxruntime as ort
from open_duck_agent.duck_sim import DuckSimulation
from diagnostics.reference_residual_policy import ReferenceResidualPolicy
from diagnostics.capture_point_observer import CapturePointObserver


class CpuActor:
    def __init__(self,path):
        options = ort.SessionOptions()
        options.intra_op_num_threads=1
        options.inter_op_num_threads=1
        self.session=ort.InferenceSession(path if isinstance(path,bytes) else str(path),sess_options=options,providers=['CPUExecutionProvider'])
        self.input_size=self.session.get_inputs()[0].shape[-1]

    def infer(self,obs):
        return self.session.run(None,{'obs':np.asarray(obs,dtype=np.float32)[None,:]})[0][0]


class UnusedLegacyPolicy:
    """A scoped constructor sentinel; native search never calls legacy motion."""
    def __init__(self,*args,**kwargs):
        pass

    def infer(self,*args,**kwargs):
        raise RuntimeError('legacy policy execution forbidden in native search')


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


def backward_path_goal(lateral,gain):
    # Backward travel: positive yaw gives negative world-side velocity.
    # This outer loop needs simulation XY or equivalent external odometry.
    return float(np.clip(gain*lateral,-.15,.15))


def balance_features(pitch,rate,deadband=.10):
    # Ignore the small normal gait lean, but react before the fall threshold.
    return np.array([math.copysign(max(0.,abs(pitch)-deadband),pitch)/.2,
                     np.clip(rate,-4.,4.)/.8])


def balance_delta(gains,pitch,rate,ramp,deadband=.10,contacts=(0.,0.)):
    gains=np.asarray(gains)
    features=balance_features(pitch,rate,deadband)
    value=.06*math.tanh(float(gains[:2]@features))*ramp
    delta=np.zeros(14)
    delta[[2,11]]=value
    delta[[4,13]]=-value
    if gains.size==4:
        ankle=.06*math.tanh(float(gains[2:]@features))*ramp
        delta[[4,13]]+=ankle*np.asarray(contacts)
    return delta


def capture_features(error,filtered_rate,deadband=.012):
    # Approximate capture-point error in metres, not torso pitch in radians.
    return np.array([math.copysign(max(0.,abs(error)-deadband),error)/.02,
                     np.clip(filtered_rate,-.8,.8)/.2])


def capture_delta(gains,features,ramp,contacts):
    values=.04*np.tanh(np.asarray(gains).reshape(2,2)@features)*ramp
    delta=np.zeros(14)
    delta[[2,11]]=values[0]
    delta[[4,13]]=-values[0]+values[1]*np.asarray(contacts)
    return delta


def native_capture_actor(template,gains):
    # Optimize the actual float32 ONNX graph, not a float64 surrogate rollout.
    import onnx
    from onnx import numpy_helper
    model=onnx.load(template)
    kernels={'capture_hip_kernel':np.asarray(gains[:2],dtype=np.float32).reshape(2,1),
             'capture_ankle_kernel':np.asarray(gains[2:],dtype=np.float32).reshape(2,1)}
    found=set()
    for initializer in model.graph.initializer:
        if initializer.name in kernels:
            found.add(initializer.name)
            initializer.CopyFrom(numpy_helper.from_array(kernels[initializer.name],initializer.name))
    if found!=set(kernels):
        raise ValueError('native capture template missing kernels')
    return CpuActor(model.SerializeToString())


class NativeRollout:
    def __init__(self,root):
        self.root=Path(root)
        self.model_path=self.root/'training/backward_reference_residual_r2/final.onnx'
        # Only substitute the unused constructor in this process and this scope.
        # The production files and simulator observation builder remain intact.
        with patch('playground.open_duck_mini_v2.mujoco_infer.OnnxInfer',UnusedLegacyPolicy),patch('open_duck_agent.duck_sim.OnnxInfer',UnusedLegacyPolicy):
            robot=DuckSimulation(self.root/'projects/Open_Duck_Playground',self.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',output_root=self.root/'outputs/phase_search_r5/traces',warmup_s=0.)
        self.sim=robot.sim
        self.robot=robot
        self.actor=CpuActor(self.model_path)
        # This experiment owns its actors/decoders. Release the unused legacy
        # policy sessions (and large default thread pools); never call robot
        # motion tools with the R2 residual actor as a legacy motor-action actor.
        self.sim.policy=None
        robot._walk_policy=None
        robot._turn_policy=None
        self.dt=self.sim.sim_dt*self.sim.decimation
        self.capture_observer=CapturePointObserver(self.sim.model)
        expected=['left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle',
                  'neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw',
                  'right_hip_roll','right_hip_pitch','right_knee','right_ankle']
        if [self.sim.model.actuator(i).name for i in range(self.sim.model.nu)]!=expected:
            raise ValueError('actuator order mismatch')
        repo=self.root/'projects/Open_Duck_Playground'
        files=[repo/'playground/open_duck_mini_v2/mujoco_infer.py',
               repo/'playground/open_duck_mini_v2/mujoco_infer_base.py',
               repo/'playground/open_duck_mini_v2/data/polynomial_coefficients.pkl',
               repo/'playground/common/poly_reference_motion_numpy.py',
               repo/'diagnostics/reference_residual_policy.py']
        files+=list((repo/'playground/open_duck_mini_v2/xmls').glob('*.xml'))
        self.source_hashes={str(path.relative_to(repo)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}

    def run(self,weights,seed,duration,save=None,feedback=None,corrector=None,up_gate=None,balance=None,pitch_template=None,balance_deadband=.10,path_gain=0.,capture=None,capture_template=None):
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
        previous_capture_error=None
        filtered_capture_rate=0.
        decoder=ReferenceResidualPolicy(self.actor,s,.12,1.)
        angles=[]
        velocities=[]
        qposes=[]
        times=[]
        capture_trace=[]
        min_up=1.
        fallen=False
        for k in range(round(duration/self.dt)):
            obs=np.asarray(s.get_obs(s.data,[-.074,0,0,0,0,0,0]),dtype=np.float32)
            base=s.default_actuator+s.action_scale*decoder.infer(obs)
            ramp=min(float(s.data.time),1.)
            error=heading(s.get_floating_base_qpos(s.data.qpos))-h0
            if path_gain:
                position=s.get_floating_base_qpos(s.data.qpos)[:2]-initial[:2]
                cross=-math.sin(h0)*position[0]+math.cos(h0)*position[1]
                error-=backward_path_goal(cross,path_gain)
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
            if capture is not None:
                capture_error=self.capture_observer.forward_offset(s.data)-phase_pitch(capture_template,s.imitation_phase)
                raw_rate=0. if previous_capture_error is None else (capture_error-previous_capture_error)/self.dt
                previous_capture_error=capture_error
                filtered_capture_rate+=self.dt/(.08+self.dt)*(raw_rate-filtered_capture_rate)
                capture_obs=capture_features(capture_error,filtered_capture_rate)
                capture_ramp=float(np.clip((s.data.time-2.)/1.,0.,1.))
            if corrector is None:
                delta=phase_delta(weights,s.imitation_phase,ramp,feedback,error)
                if up_gate:
                    high,low=up_gate
                    delta*=np.clip((current_up-low)/(high-low),0.,1.)
                if balance is not None:
                    delta+=balance_delta(balance,pitch,pitch_rate,ramp,balance_deadband,s.get_feet_contacts(s.data))
                if capture is not None:
                    delta+=capture_delta(capture,capture_obs,capture_ramp,s.get_feet_contacts(s.data))
            else:
                features=[*s.imitation_phase,math.sin(error),ramp]
                if up_gate:
                    features.append(current_up)
                if balance is not None:
                    features.extend(balance_features(pitch,pitch_rate,balance_deadband))
                    if len(balance)==4:
                        features.extend(s.get_feet_contacts(s.data))
                if capture is not None:
                    features.extend([*capture_obs,capture_ramp])
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
                capture_trace.append(self.capture_observer.forward_offset(s.data))
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
            np.savez_compressed(save,qpos=np.asarray(qposes),time=np.asarray(times),yaw=np.asarray(angles),capture_forward_offset_m=np.asarray(capture_trace))
        return row


def worker_init(root):
    global worker_engine
    worker_engine=NativeRollout(root)


def worker_run(task):
    return worker_run_direct(worker_engine,task)


def worker_validate(task):
    weights,feedback,balance,seed,duration,save,up_gate,template,deadband,model,path_gain=task[:11]
    capture,capture_template=task[11:] if len(task)>11 else (None,None)
    global worker_corrector, worker_corrector_path
    if model:
        if globals().get('worker_corrector_path')!=model:
            worker_corrector=CpuActor(model)
            worker_corrector_path=model
        corrector=worker_corrector
    else:
        corrector=None
    return worker_engine.run(weights,seed,duration,save,feedback,corrector,up_gate,balance,template,deadband,path_gain,capture,capture_template)


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
    p.add_argument('--capture-base',type=Path,help='freeze all prior corrections; search four simulation-COM feedback gains')
    p.add_argument('--native-controller-template',type=Path,help='evaluate capture candidates using their actual float32 ONNX graph')
    p.add_argument('--ankle-balance',action='store_true',help='also search two support-contact-gated ankle feedback gains')
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--initial-std',type=float,help='optional smaller local CEM exploration for refinement')
    p.add_argument('--corrector-onnx',type=Path,help='independent exported-controller native gate')
    a=p.parse_args()
    if not 0 < a.duration <= 60 or not 1 <= a.workers <= 8:
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
        if balance is not None and (len(balance) not in (2,4) or not np.isfinite(balance).all() or np.max(np.abs(balance))>3.):
            raise ValueError('invalid balance weights')
        template=artifact.get('pitch_template_weights')
        capture=artifact.get('capture_weights')
        capture_template=artifact.get('capture_template_weights')
        if capture is not None:
            if len(capture)!=4 or not np.isfinite(capture).all() or max(abs(np.asarray(capture)))>3:
                raise ValueError('invalid capture feedback weights')
            if capture_template is None or len(capture_template)!=7 or not np.isfinite(capture_template).all() or max(abs(np.asarray(capture_template)))>.3:
                raise ValueError('invalid capture template')
            if abs(artifact['capture_template_period_s']-engine.dt*engine.sim.PRM.nb_steps_in_period)>1e-6:
                raise ValueError('capture template period mismatch')
        deadband=float(artifact.get('balance_deadband_rad',.10))
        path_gain=float(artifact.get('path_heading_gain_rad_per_m',0.))
        if not np.isfinite(path_gain) or not 0 <= path_gain <= 2:
            raise ValueError('invalid path-heading gain')
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
        seeds=range(a.seed_start,a.seed_start+a.seed_count)
        if a.workers==1:
            rows=[engine.run(weights,s,a.duration,a.output/f'seed_{s}.npz',feedback,corrector,artifact.get('up_gate'),balance,template,deadband,path_gain,capture,capture_template) for s in seeds]
        else:
            tasks=[(weights,feedback,balance,s,a.duration,a.output/f'seed_{s}.npz',artifact.get('up_gate'),template,deadband,str(a.corrector_onnx) if a.corrector_onnx else None,path_gain,capture,capture_template) for s in seeds]
            with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(a.root,)) as pool:
                rows=list(pool.map(worker_validate,tasks))
        (a.output/'results.json').write_text(json.dumps({'rows':rows,'controller':str(a.validate),'corrector_onnx':str(a.corrector_onnx) if a.corrector_onnx else None,'environment_source_sha256':engine.source_hashes,'motor_velocity_limit_rad_s':float(engine.sim.max_motor_velocity)},indent=2))
        for r in rows:
            print(json.dumps(r),flush=True)
        return
    seeds=[int(x) for x in a.seeds.split(',')]
    fixed=None
    fixed_feedback=None
    fixed_balance=None
    template=None
    template_metadata={}
    capture_template=None
    capture_metadata={}
    deadband=.10
    mean=np.zeros(18)
    if sum(bool(v) for v in (a.balance_base,a.feedback_base,a.capture_base))>1:
        raise ValueError('choose exactly one optimization mode')
    if a.ankle_balance and not a.balance_base:
        raise ValueError('ankle feedback requires a frozen balance base')
    if a.native_controller_template and not a.capture_base:
        raise ValueError('native graph search currently requires capture mode')
    if a.balance_base:
        restored=json.loads(a.balance_base.read_text())
        fixed=np.array(restored['weights'])
        fixed_feedback=np.array(restored['feedback_weights'])
        template=restored.get('pitch_template_weights')
        template_metadata={k:v for k,v in restored.items() if k.startswith('pitch_template_')}
        deadband=float(restored.get('balance_deadband_rad',.10))
        mean=np.zeros(4 if a.ankle_balance else 2)
    if a.feedback_base:
        restored=json.loads(a.feedback_base.read_text())
        fixed=np.array(restored['weights'])
        fixed_balance=restored.get('balance_weights')
        template=restored.get('pitch_template_weights')
        template_metadata={k:v for k,v in restored.items() if k.startswith('pitch_template_')}
        deadband=float(restored.get('balance_deadband_rad',.10))
        mean=np.zeros(6)
    if a.capture_base:
        restored=json.loads(a.capture_base.read_text())
        fixed=np.array(restored['weights'])
        fixed_feedback=np.array(restored['feedback_weights'])
        fixed_balance=restored['balance_weights']
        if len(fixed_balance)!=4:
            raise ValueError('capture experiment requires existing support-contact balance')
        template=restored['pitch_template_weights']
        template_metadata={k:v for k,v in restored.items() if k.startswith('pitch_template_')}
        deadband=float(restored['balance_deadband_rad'])
        capture_template=restored['capture_template_weights']
        capture_metadata={k:v for k,v in restored.items() if k.startswith('capture_template_')}
        if abs(restored['capture_template_period_s']-engine.dt*engine.sim.PRM.nb_steps_in_period)>1e-6:
            raise ValueError('capture template period mismatch')
        mean=np.zeros(4)
    if a.resume:
        restored=json.loads(a.resume.read_text())
        key='capture_weights' if a.capture_base else 'balance_weights' if a.balance_base else 'feedback_weights' if fixed is not None else 'weights'
        values=np.array(restored[key])
        if a.ankle_balance and values.size==2:
            values=np.concatenate([values,np.zeros(2)])
        mean=values.reshape(mean.size)
    std=np.full(mean.size,.8 if fixed is not None else .35)
    if a.initial_std is not None:
        if not 0 < a.initial_std <= 1:
            raise ValueError('invalid CEM initial std')
        std[:]=a.initial_std
    rng=np.random.default_rng(87)
    best_score=float('inf')
    start=time.monotonic()
    baseline_actor=native_capture_actor(a.native_controller_template,np.zeros(4)) if a.native_controller_template else None
    baseline=[engine.run(np.zeros(18) if fixed is None else fixed,s,a.duration,feedback=fixed_feedback,balance=fixed_balance,pitch_template=template,balance_deadband=deadband,capture=np.zeros(4) if baseline_actor else None,capture_template=capture_template,corrector=baseline_actor) for s in seeds]
    print('ZERO BASELINE:',json.dumps(baseline),flush=True)
    records=[]
    pool=ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=worker_init,initargs=(a.root,)) if a.workers>1 else None
    for gen in range(a.generations):
        limit=3. if fixed is not None else 2.
        candidates=np.clip(mean+rng.normal(size=(a.population,mean.size))*std,-limit,limit)
        candidates[0]=mean
        tasks=[(candidate if fixed is None else fixed,fixed_feedback if a.balance_base or a.capture_base else None if fixed is None else candidate,candidate if a.balance_base else fixed_balance,seeds,a.duration,template,deadband,candidate if a.capture_base else None,capture_template,str(a.native_controller_template) if a.native_controller_template else None) for candidate in candidates]
        evaluated=pool.map(worker_run,tasks) if pool else (worker_run_direct(engine,task) for task in tasks)
        for candidate,rows in zip(candidates,evaluated):
            weights=candidate if fixed is None else fixed
            score=float(np.mean([r['cost'] for r in rows])+.3*max(r['cost'] for r in rows))
            records.append(dict(generation=gen,score=score,weights=candidate.tolist(),rows=rows))
            if score<best_score:
                best_score=score
                artifact=dict(controller_type='phase_leg_correction_v1',simulation_only=True,baseline_sha256=hashlib.sha256(engine.model_path.read_bytes()).hexdigest(),reference_dx=-.0925,reference_interpolation=True,residual_gain=.12,ramp_s=1.,control_dt=engine.dt,weights=weights.tolist(),score=score,rows=rows)
                if a.capture_base:
                    artifact.update(controller_type='phase_heading_contact_capture_v1',feedback_weights=fixed_feedback.tolist(),balance_weights=list(fixed_balance),balance_deadband_rad=deadband,capture_weights=candidate.tolist(),capture_contract='approximate_LIPM_COM;deadband_0.012m;scale_0.02m;rate_EMA_0.08s;rate_clip_0.8_scaled_0.2;gain_0.04rad;activation_2s_ramp_1s;simulation_state_required',**template_metadata,**capture_metadata)
                    if a.native_controller_template:
                        artifact.update(training_inference='actual_float32_ONNX',native_template_sha256=hashlib.sha256(a.native_controller_template.read_bytes()).hexdigest())
                elif a.balance_base:
                    artifact.update(controller_type='phase_heading_balance_v1',feedback_weights=fixed_feedback.tolist(),balance_weights=candidate.tolist(),balance_deadband_rad=deadband,balance_contract='deadband_pitch_scaled_0.2;finite_difference_rate_clipped_4_scaled_0.8;gain_0.06_rad',**template_metadata)
                elif fixed is not None:
                    artifact.update(controller_type='phase_leg_feedback_v1',feedback_weights=candidate.tolist())
                    if fixed_balance is not None:
                        artifact.update(controller_type='phase_heading_balance_v1',balance_weights=list(fixed_balance),balance_deadband_rad=deadband,**template_metadata)
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
    weights,feedback,balance,seeds,duration,template,deadband=task[:7]
    capture,capture_template=task[7:9] if len(task)>7 else (None,None)
    native_template=task[9] if len(task)>9 else None
    corrector=native_capture_actor(native_template,capture) if native_template else None
    return [engine.run(weights,s,duration,feedback=feedback,balance=balance,pitch_template=template,balance_deadband=deadband,capture=capture,capture_template=capture_template,corrector=corrector) for s in seeds]


if __name__=='__main__':
    main()
