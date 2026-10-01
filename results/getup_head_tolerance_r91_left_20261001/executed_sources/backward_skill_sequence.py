"""Simulation-only standing/backward/stopping sequence with persistent state.

Standing uses the original normalized-action policy. Backward uses frozen R2,
R11's 8-feature joint corrector and the explicit target-filter contract. A zero
command must not be sent through the fixed-negative-reference backward decoder.
"""
import argparse
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import mujoco
import numpy as np
from diagnostics.backward_phase_search import NativeRollout, CpuActor, heading, body_pitch, phase_pitch, balance_features
from diagnostics.reference_residual_policy import ReferenceResidualPolicy


SCHEDULE=[('stand',3.),('backward',10.),('stop',3.),('restart',10.),('stop_final',3.)]


def controller_delta(corrector,contract,phase,error,ramp,pitch,pitch_rate,contacts):
    features=[*phase,math.sin(error),ramp,*balance_features(pitch,pitch_rate,contract['balance_deadband_rad']),*contacts]
    return corrector.infer(features)


class Sequence:
    def __init__(self,root,contract_path,model_path,backward_only=False,stop_blend=0.,align_support=False):
        self.engine=NativeRollout(root)
        self.contract=json.loads(Path(contract_path).read_text())
        self.model=CpuActor(model_path)
        self.schedule=[('backward',60.)] if backward_only else SCHEDULE
        self.stop_blend=stop_blend
        self.align_support=align_support
        if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=self.contract['onnx_sha256']:
            raise ValueError('corrector hash mismatch')
        if hashlib.sha256(self.engine.model_path.read_bytes()).hexdigest()!=self.contract['baseline_sha256']:
            raise ValueError('R2 hash mismatch')
        if self.model.input_size!=8 or self.contract.get('capture_weights') or self.contract.get('up_gate') or self.contract.get('path_heading_gain_rad_per_m'):
            raise ValueError('this transition test supports only the R11 8-feature contract')
        self.tau=float(self.contract['target_smoothing_tau_s'])
        if not 0<self.tau<=.1:
            raise ValueError('invalid target-filter time constant')
        if abs(self.contract['pitch_template_period_s']-self.engine.dt*self.engine.sim.PRM.nb_steps_in_period)>1e-6:
            raise ValueError('phase-template period mismatch')
        self.stand=CpuActor(Path(root)/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx')

    def run(self,seed,out):
        engine=self.engine
        s=engine.sim
        mujoco.mj_resetData(s.model,s.data)
        engine.robot.reset()
        s.data.qvel[:]=np.random.default_rng(seed).uniform(-.02,.02,s.model.nv)
        s.imitation_phase=np.array([1.,0.],dtype=np.float32)
        mujoco.mj_forward(s.model,s.data)
        decoder=ReferenceResidualPolicy(engine.actor,s,.12,1.)
        filtered=s.prev_motor_targets.copy()
        traces=[]
        phases=[]
        backward_heading=heading(s.get_floating_base_qpos(s.data.qpos))
        backward_start_s=float(s.data.time)
        previous_pitch=body_pitch(s.get_floating_base_qpos(s.data.qpos))
        previous_nominal=0.
        for name,seconds in self.schedule:
            active=name in ('backward','restart')
            command=[-.074,0,0,0,0,0,0] if active else [0.]*7
            start_time=float(s.data.time)
            start=s.get_floating_base_qpos(s.data.qpos).copy()
            start_heading=heading(start)
            previous_heading=start_heading
            accumulated=0.
            if active:
                previous_pitch=body_pitch(start)
                previous_nominal=0.
                backward_heading=start_heading
                backward_start_s=start_time
                decoder.start_s=start_time
            blend_start=None if self.align_support else start_time
            positions=[start[:2]]
            min_up=1.
            fallen=False
            for tick in range(round(seconds/engine.dt)):
                obs=np.asarray(s.get_obs(s.data,command),dtype=np.float32)
                stopping=name in ('stop','stop_final') and self.stop_blend>0
                if stopping and blend_start is None:
                    if all(s.get_feet_contacts(s.data)) or s.data.time-start_time>=.6:
                        blend_start=float(s.data.time)
                u=0. if blend_start is None else float(np.clip((s.data.time-blend_start)/self.stop_blend,0.,1.)) if stopping else 1.
                transitioning=stopping and u<1.
                if active or transitioning:
                    back_obs=obs.copy()
                    back_obs[6:13]=[-.074,0,0,0,0,0,0]
                    base=s.default_actuator+s.action_scale*decoder.infer(back_obs)
                    ramp=min(float(s.data.time)-backward_start_s,1.)
                    q=s.get_floating_base_qpos(s.data.qpos)
                    pitch=body_pitch(q)
                    rate=(pitch-previous_pitch)/engine.dt
                    previous_pitch=pitch
                    nominal=ramp*phase_pitch(self.contract['pitch_template_weights'],s.imitation_phase)
                    rate-=(nominal-previous_nominal)/engine.dt
                    previous_nominal=nominal
                    delta=controller_delta(self.model,self.contract,s.imitation_phase,heading(q)-backward_heading,ramp,pitch-nominal,rate,s.get_feet_contacts(s.data))
                    target=base+delta
                    if transitioning:
                        stand_target=s.default_actuator+s.action_scale*self.stand.infer(obs)
                        stand_weight=u*u*(3-2*u)
                        target=(1-stand_weight)*target+stand_weight*stand_target
                else:
                    target=s.default_actuator+s.action_scale*self.stand.infer(obs)
                if not np.isfinite(target).all():
                    raise ValueError('nonfinite policy target')
                target=np.clip(target,decoder.lower,decoder.upper)
                filtered+=engine.dt/(self.tau+engine.dt)*(target-filtered)
                slew=np.clip(filtered,s.prev_motor_targets-s.max_motor_velocity*engine.dt,s.prev_motor_targets+s.max_motor_velocity*engine.dt)
                s.data.ctrl[:]=slew
                s.motor_targets=slew.copy()
                s.prev_motor_targets=slew.copy()
                s.last_last_last_action=s.last_last_action.copy()
                s.last_last_action=s.last_action.copy()
                s.last_action=((filtered-s.default_actuator)/s.action_scale).astype(np.float32)
                for _ in range(s.decimation):
                    mujoco.mj_step(s.model,s.data)
                s.imitation_i=(s.imitation_i+1)%s.PRM.nb_steps_in_period
                phi=s.imitation_i/s.PRM.nb_steps_in_period*2*np.pi
                s.imitation_phase=np.array([np.cos(phi),np.sin(phi)],dtype=np.float32)
                q=s.get_floating_base_qpos(s.data.qpos).copy()
                current_heading=heading(q)
                accumulated+=math.atan2(math.sin(current_heading-previous_heading),math.cos(current_heading-previous_heading))
                previous_heading=current_heading
                up=1-2*(q[4]**2+q[5]**2)
                min_up=min(min_up,up)
                positions.append(q[:2].copy())
                traces.append((float(s.data.time),s.data.qpos.copy(),name))
                fallen=bool(up<.5 or q[2]<.08 or not np.isfinite(s.data.qpos).all() or not np.isfinite(s.data.qvel).all())
                if fallen:
                    break
            elapsed=(len(positions)-1)*engine.dt
            displacement=positions[-1]-positions[0]
            axial=(math.cos(start_heading)*displacement[0]+math.sin(start_heading)*displacement[1])/elapsed
            lateral=-math.sin(start_heading)*displacement[0]+math.cos(start_heading)*displacement[1]
            tail_start=max(0,len(positions)-1-round(1/engine.dt))
            tail_drift=float(np.linalg.norm(positions[-1]-positions[tail_start]))
            qualified=not fallen and min_up>=.94
            if active:
                qualified=qualified and -.10<=axial<=-.05 and abs(math.degrees(accumulated))<=15 and abs(lateral)<=.25
            else:
                qualified=qualified and tail_drift<=.005
            row=dict(phase=name,duration_s=elapsed,fallen=fallen,qualified=bool(qualified),speed_mps=float(axial),lateral_m=float(lateral),yaw_change_deg=math.degrees(accumulated),minimum_up_z=float(min_up),last_1s_horizontal_drift_m=tail_drift)
            phases.append(row)
            if fallen:
                break
        np.savez_compressed(out/f'seed_{seed}.npz',time=np.asarray([r[0] for r in traces]),qpos=np.asarray([r[1] for r in traces]),phase=np.asarray([r[2] for r in traces]))
        return dict(seed=seed,completed=len(phases)==len(self.schedule) and not phases[-1]['fallen'],qualified=len(phases)==len(self.schedule) and all(r['qualified'] for r in phases),phases=phases)


def initialize(root,contract,model,backward_only,stop_blend,align_support):
    global sequence
    sequence=Sequence(root,contract,model,backward_only,stop_blend,align_support)


def run_worker(task):
    seed,out=task
    return sequence.run(seed,out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/data/shijinsheng/open_duck')
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed-start',type=int,default=1000)
    p.add_argument('--seed-count',type=int,default=20)
    p.add_argument('--workers',type=int,default=8)
    p.add_argument('--backward-only-regression',action='store_true')
    p.add_argument('--stop-blend-s',type=float,default=0.)
    p.add_argument('--align-double-support',action='store_true')
    a=p.parse_args()
    if a.output.exists() or not 1<=a.workers<=8 or not 0<=a.stop_blend_s<=1.5:
        raise ValueError('existing experiment preserved or invalid worker count')
    a.output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(a.root,a.contract,a.model,a.backward_only_regression,a.stop_blend_s,a.align_double_support)) as pool:
        rows=list(pool.map(run_worker,[(seed,a.output) for seed in range(a.seed_start,a.seed_start+a.seed_count)]))
        (a.output/'results.json').write_text(json.dumps(dict(rows=rows,schedule=[('backward',60.)] if a.backward_only_regression else SCHEDULE,contract=str(a.contract),model=str(a.model),initialization='mj_resetData then exact home pose and seeded qvel',stop_blend_s=a.stop_blend_s,align_double_support=a.align_double_support,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),simulation_only=True,hardware_readiness=False),indent=2))
    print(json.dumps(dict(tested=len(rows),completed=sum(r['completed'] for r in rows),qualified=sum(r['qualified'] for r in rows),failed_seeds=[r['seed'] for r in rows if not r['completed']],unqualified_seeds=[r['seed'] for r in rows if not r['qualified']]),indent=2),flush=True)
