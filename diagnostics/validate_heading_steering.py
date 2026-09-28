"""Dedicated native gate for the new 107-observation, two-action corrector.

No production agent changes. A legacy actor cannot decode this controller.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import mujoco
import numpy as np

from open_duck_agent.duck_sim import DuckSimulation
from playground.common.onnx_infer import OnnxInfer
from playground.open_duck_mini_v2.heading_steering import yaw_and_features, steering_delta
from diagnostics.reference_residual_policy import ReferenceResidualPolicy


def yaw(q):
    w,x,y,z = q[3:7]
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path)
    p.add_argument("--baseline", type=Path, default=Path("/data/shijinsheng/open_duck/training/backward_reference_residual_r2/final.onnx"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration-s", type=float, default=10)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--initial-error", type=float, default=0.)
    p.add_argument("--constant-yaw", type=float, default=0., help="simulation actuation-authority probe")
    p.add_argument("--constant-roll", type=float, default=0.)
    p.add_argument("--feedback-gain", type=float, default=0., help='simulation-only proportional heading authority probe; not a learned policy')
    a = p.parse_args()
    if not 0 < a.duration_s <= 60 or not 1 <= a.seeds <= 50:
        raise ValueError('gate duration/seeds out of bounds')
    if not all(math.isfinite(x) for x in [a.initial_error,a.constant_yaw,a.constant_roll]):
        raise ValueError('non-finite control argument')
    if abs(a.initial_error) > .5 or max(abs(a.constant_yaw),abs(a.constant_roll)) > 1:
        raise ValueError('heading offset or correction out of bounds')
    if not 0 <= a.feedback_gain <= 8 or (a.model and a.feedback_gain):
        raise ValueError('invalid or mixed feedback probe')
    root = Path("/data/shijinsheng/open_duck")
    if a.model:
        contract = json.loads((a.model.parent/"controller_contract.json").read_text())
        if contract['controller_type'] != 'heading_steering_v1':
            raise ValueError('wrong controller decoder')
        if hashlib.sha256(a.model.read_bytes()).hexdigest() != contract['onnx_sha256']:
            raise ValueError('corrector checksum mismatch')
        if hashlib.sha256(a.baseline.read_bytes()).hexdigest() != contract['baseline_sha256']:
            raise ValueError('baseline checksum mismatch')
        if os.environ.get('REFERENCE_DX') != contract['reference_dx'] or os.environ.get('REFERENCE_DX_INTERPOLATION','0') != contract['reference_dx_interpolation']:
            raise ValueError('reference configuration mismatch')
        if contract['residual_gain_rad'] != .12 or contract['reference_ramp_s'] != 1.:
            raise ValueError('unsupported reference decoder gains')
    rows = []
    for seed in range(a.seed_start, a.seed_start+a.seeds):
        robot = DuckSimulation(root/"projects/Open_Duck_Playground",
                               root/"projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx",
                               output_root=a.output.parent/"native_traces", warmup_s=0.)
        sim = robot.sim
        sim.imitation_i = 0.
        sim.imitation_phase = np.array([1.,0.], dtype=np.float32)
        sim.data.qvel[:] += np.random.default_rng(seed).uniform(-.02, .02, sim.model.nv)
        mujoco.mj_forward(sim.model, sim.data)
        start = sim.get_floating_base_qpos(sim.data.qpos).copy()
        h0 = yaw(start)
        target_heading = h0+a.initial_error
        previous_heading = h0
        total_yaw = 0.
        baseline = ReferenceResidualPolicy(OnnxInfer(str(a.baseline), awd=True), sim, .12, 1.)
        actor = OnnxInfer(str(a.model), awd=True) if a.model else None
        yaw_gain = contract["steering_yaw_gain_rad"] if a.model else .05
        roll_gain = contract["steering_roll_gain_rad"] if a.model else .025
        commands = [-.074,0.,0.,0.,0.,0.,0.]
        dt = sim.sim_dt*sim.decimation
        trajectory = []
        qpos_trace = []
        qvel_trace = []
        correction_magnitude = []
        min_up = 1.
        fallen = False
        completed = 0
        for k in range(round(a.duration_s/dt)):
            # Exact contract: sense -> infer -> slew target -> advance physics.
            obs = np.asarray(sim.get_obs(sim.data, commands), dtype=np.float32)
            base_target = sim.default_actuator+sim.action_scale*baseline.infer(obs)
            features = yaw_and_features(sim.get_floating_base_qpos(sim.data.qpos), target_heading,
                                        sim.data.time-baseline.start_s, 1.)
            augmented = np.concatenate([obs, features]).astype(np.float32)
            correction = np.asarray(actor.infer(augmented), dtype=np.float32) if actor else np.array([a.constant_yaw,a.constant_roll])
            if a.feedback_gain:
                # Short-run sign hypothesis only; the long-run gate may reject it.
                u = np.clip(-a.feedback_gain*features[0],-1.,1.)
                correction = np.array([u,-u],dtype=np.float32)
            if correction.shape != (2,) or not np.isfinite(correction).all():
                raise ValueError('invalid steering actor output')
            correction_magnitude.append(np.abs(correction))
            target = np.clip(base_target+steering_delta(correction,yaw_gain,roll_gain), baseline.lower,baseline.upper)
            motor_action = ((target-sim.default_actuator)/sim.action_scale).astype(np.float32)
            motor_target = np.clip(target, sim.prev_motor_targets-sim.max_motor_velocity*dt,
                                   sim.prev_motor_targets+sim.max_motor_velocity*dt)
            sim.data.ctrl[:] = motor_target
            sim.motor_targets = motor_target.copy()
            sim.prev_motor_targets = motor_target.copy()
            sim.last_last_last_action = sim.last_last_action.copy()
            sim.last_last_action = sim.last_action.copy()
            sim.last_action = motor_action.copy()
            for _ in range(sim.decimation):
                mujoco.mj_step(sim.model, sim.data)
            sim.imitation_i = (sim.imitation_i+1) % sim.PRM.nb_steps_in_period
            angle = sim.imitation_i/sim.PRM.nb_steps_in_period*2*np.pi
            sim.imitation_phase = np.array([np.cos(angle),np.sin(angle)], dtype=np.float32)
            now = sim.get_floating_base_qpos(sim.data.qpos).copy()
            hn = yaw(now)
            total_yaw += math.atan2(math.sin(hn-previous_heading),math.cos(hn-previous_heading))
            previous_heading = hn
            up = 1-2*(now[4]**2+now[5]**2)
            min_up = min(min_up, up)
            trajectory.append([float(sim.data.time), *now[:3], hn, up, *correction])
            qpos_trace.append(sim.data.qpos.copy())
            qvel_trace.append(sim.data.qvel.copy())
            completed += 1
            fallen = bool(up < .5 or now[2] < .08 or not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all())
            if fallen:
                break
        end = sim.get_floating_base_qpos(sim.data.qpos)
        dx,dy = end[:2]-start[:2]
        speed = (math.cos(h0)*dx+math.sin(h0)*dy)/(completed*dt)
        lateral = -math.sin(h0)*dx+math.cos(h0)*dy
        row = dict(seed=seed, model=str(a.model) if a.model else "zero_or_constant_correction",
                   baseline_sha256=hashlib.sha256(a.baseline.read_bytes()).hexdigest(),
                   initial_target_offset_rad=a.initial_error, constant_yaw=a.constant_yaw,
                   constant_roll=a.constant_roll, feedback_gain=a.feedback_gain,
                   duration_s=completed*dt, fallen=fallen,
                   speed_mps=float(speed), lateral_m=float(lateral), yaw_change_deg=math.degrees(total_yaw),
                   final_heading_error_deg=math.degrees(math.atan2(math.sin(previous_heading-target_heading),math.cos(previous_heading-target_heading))),
                   minimum_up_z=float(min_up), mean_abs_correction=np.mean(correction_magnitude,axis=0).tolist())
        rows.append(row)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(a.output.parent/f"seed_{seed}_trajectory.npz", trajectory=np.asarray(trajectory),
                            qpos=np.asarray(qpos_trace), qvel=np.asarray(qvel_trace))
        print(json.dumps(row),flush=True)
    a.output.write_text(json.dumps({"rows":rows,"step_contract":"infer_then_physics",
                                  "history_contract":"fresh_three_motor_actions"},indent=2))
    print("SAVED",a.output,flush=True)


if __name__ == "__main__":
    main()
