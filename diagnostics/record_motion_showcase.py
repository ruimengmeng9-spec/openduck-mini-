"""Record one uninterrupted, simulation-only forward/turn/backward/stop run.

Uses the verified R22 backward decoder, not a legacy normalized-action decoder.
No state resets between skills, no friction changes, and no hardware access.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from diagnostics.backward_skill_sequence import Sequence, controller_delta
from diagnostics.backward_phase_search import CpuActor, heading, body_pitch, phase_pitch
from diagnostics.reference_residual_policy import ReferenceResidualPolicy
from diagnostics.negative_turn_residual import ResidualNegativePolicy
from diagnostics.skill_target_execution import SkillTargetExecutor


SCHEDULE = [('Stand', 3., 0., 0.), ('Forward', 5., .15, 0.),
            ('Settle before turn', 1., 0., 0.), ('Turn right', 5., 0., -.15),
            ('Settle before backward', 2., 0., 0.), ('Backward', 7., -.074, 0.),
            ('Stop', 3., 0., 0.)]


def rollout(root, output, seed, legacy_shared_controls=False):
    candidate = root / 'training/backward_sim_candidate_r22'
    controller = Sequence(root, candidate/'controller_contract.json', candidate/'corrector.onnx', stop_blend=1.)
    turn_base = root/'training/official_seed_turn_balance_v5_yaw_error/final.onnx'
    turn_residual = root/'training/negative_mirror_distill_v8_model/residual.onnx'
    turn = ResidualNegativePolicy(CpuActor(turn_base), turn_residual, 'extended')
    engine, contract = controller.engine, controller.contract
    s = engine.sim
    mujoco.mj_resetData(s.model, s.data)
    engine.robot.reset()
    s.data.qvel[:] = np.random.default_rng(seed).uniform(-.02, .02, s.model.nv)
    s.imitation_phase = np.array([1., 0.], dtype=np.float32)
    mujoco.mj_forward(s.model, s.data)
    decoder = ReferenceResidualPolicy(engine.actor, s, .12, 1.)
    executor = SkillTargetExecutor(s, engine.dt, decoder.lower, decoder.upper, controller.tau)
    records, metrics = [], []
    back_start = back_heading = previous_pitch = previous_nominal = 0.
    fallen = False
    for name, duration, vx, yaw in SCHEDULE:
        command = [vx, 0., yaw, 0., 0., 0., 0.]
        start_time = float(s.data.time)
        q0 = s.get_floating_base_qpos(s.data.qpos).copy()
        yaw0 = last_yaw = heading(q0)
        yaw_change = 0.
        min_up = 1.
        positions = [q0[:2].copy()]
        if name == 'Backward':
            back_start = decoder.start_s = start_time
            back_heading = yaw0
            previous_pitch = body_pitch(q0)
            previous_nominal = 0.
        for _ in range(round(duration/engine.dt)):
            raw_action = None
            obs = np.asarray(s.get_obs(s.data, command), dtype=np.float32)
            u = float(np.clip((s.data.time-start_time)/1., 0., 1.))
            blending_stop = name == 'Stop' and u < 1.
            if name == 'Backward' or blending_stop:
                back_obs = obs.copy()
                back_obs[6:13] = [-.074, 0., 0., 0., 0., 0., 0.]
                base = s.default_actuator + s.action_scale*decoder.infer(back_obs)
                ramp = min(float(s.data.time)-back_start, 1.)
                q = s.get_floating_base_qpos(s.data.qpos)
                pitch = body_pitch(q)
                rate = (pitch-previous_pitch)/engine.dt
                previous_pitch = pitch
                nominal = ramp*phase_pitch(contract['pitch_template_weights'], s.imitation_phase)
                rate -= (nominal-previous_nominal)/engine.dt
                previous_nominal = nominal
                delta = controller_delta(controller.model, contract, s.imitation_phase,
                                         heading(q)-back_heading, ramp, pitch-nominal,
                                         rate, s.get_feet_contacts(s.data))
                target = base + delta
                if blending_stop:
                    stand_target = s.default_actuator+s.action_scale*controller.stand.infer(obs)
                    weight = u*u*(3.-2.*u)
                    target = (1.-weight)*target+weight*stand_target
            else:
                policy = turn if name == 'Turn right' else controller.stand
                raw_action = policy.infer(obs)
                target = s.default_actuator+s.action_scale*raw_action
            # Keep the verified backward contract through its stop/settle phase.
            # Forward/turn/ordinary stand retain the legacy policy contract.
            executor.apply(target, raw_action, legacy_shared_controls or name in ('Backward','Stop'))
            for _ in range(s.decimation):
                mujoco.mj_step(s.model, s.data)
            s.imitation_i = (s.imitation_i+1)%s.PRM.nb_steps_in_period
            phi = s.imitation_i/s.PRM.nb_steps_in_period*2.*np.pi
            s.imitation_phase = np.array([np.cos(phi), np.sin(phi)], dtype=np.float32)
            q = s.get_floating_base_qpos(s.data.qpos).copy()
            current_yaw = heading(q)
            yaw_change += math.atan2(math.sin(current_yaw-last_yaw), math.cos(current_yaw-last_yaw))
            last_yaw = current_yaw
            up = 1.-2.*(q[4]**2+q[5]**2)
            min_up = min(min_up, up)
            positions.append(q[:2].copy())
            fallen = bool(up < .5 or q[2] < .08 or not np.isfinite(s.data.qpos).all() or not np.isfinite(s.data.qvel).all())
            records.append((float(s.data.time), s.data.qpos.copy(), s.data.qvel.copy(), name,
                            command, engine.robot._sensor('local_linvel'), up, current_yaw))
            if fallen:
                break
        elapsed = float(s.data.time)-start_time
        displacement = positions[-1]-positions[0]
        metrics.append(dict(phase=name, duration_s=elapsed, fallen=fallen, minimum_up_z=float(min_up),
                            displacement_xy_m=displacement.tolist(),
                            axial_speed_mps=float(np.dot(displacement,[math.cos(yaw0),math.sin(yaw0)])/elapsed),
                            mean_local_forward_last2s_mps=float(np.mean([r[5][0] for r in records[-min(len(positions)-1,100):]])),
                            yaw_change_deg=math.degrees(yaw_change),
                            last_1s_drift_m=float(np.linalg.norm(positions[-1]-positions[max(0,len(positions)-51)]))))
        print(json.dumps(metrics[-1]), flush=True)
        if fallen:
            break
    trajectory = dict(time=np.array([r[0] for r in records]), qpos=np.array([r[1] for r in records]),
                      qvel=np.array([r[2] for r in records]), phase=np.array([r[3] for r in records]),
                      command=np.array([r[4] for r in records]), local_velocity=np.array([r[5] for r in records]),
                      up_z=np.array([r[6] for r in records]), yaw=np.array([r[7] for r in records]))
    np.savez_compressed(output/'trajectory.npz', **trajectory)
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [candidate/'corrector.onnx', candidate/'controller_contract.json', engine.model_path,
               turn_base, turn_residual, root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx', Path(__file__),
               Path(__file__).with_name('skill_target_execution.py')]}
    summary = dict(seed=seed, continuous=True, resets_between_skills=0, simulation_only=True,
                   hardware_readiness=False, schedule=SCHEDULE, completed=not fallen and len(metrics)==len(SCHEDULE),
                   phases=metrics, hashes=hashes, target_filter_tau_s=controller.tau, stop_blend_s=1.,
                   render_fps=25, playback_speed=1.)
    summary['execution_contract'] = 'legacy_shared_filter' if legacy_shared_controls else 'per_skill_v2'
    summary['legacy_target_filter_tau_s'] = controller.tau if legacy_shared_controls else 0.
    summary['backward_target_filter_tau_s'] = controller.tau
    (output/'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return s.model, trajectory, summary


def render(model, trajectory, output):
    import imageio_ffmpeg
    data = mujoco.MjData(model)
    width, height = 640, 480
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    follow = mujoco.MjvCamera()
    follow.type = mujoco.mjtCamera.mjCAMERA_FREE
    follow.distance, follow.azimuth, follow.elevation = .85, 135., -18.
    overview = mujoco.MjvCamera()
    overview.type = mujoco.mjtCamera.mjCAMERA_FREE
    xy = trajectory['qpos'][:,:2]
    overview.lookat[:] = [*(((xy.min(axis=0)+xy.max(axis=0))/2.)), .05]
    overview.distance = max(1.3, float(np.ptp(xy,axis=0).max())*2.8)
    overview.azimuth, overview.elevation = 90., -89.
    colors = {'Forward':(50,205,120), 'Turn right':(255,195,65), 'Backward':(80,170,255), 'Stop':(245,110,110)}
    video = output/'forward_turn_backward_stop.mp4'
    writer = imageio_ffmpeg.write_frames(str(video), (1280,544), fps=25, codec='libx264', quality=8,
                                        output_params=['-movflags','+faststart'], macro_block_size=16)
    writer.send(None)
    samples = []
    sample_indices = [74, 200, 450, 700, 1000, len(xy)-26]
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        try:
            for i in range(0,len(xy),2):
                data.qpos[:] = trajectory['qpos'][i]
                data.qvel[:] = trajectory['qvel'][i]
                data.time = float(trajectory['time'][i])
                mujoco.mj_forward(model, data)
                follow.lookat[:] = [*data.qpos[:2], .17]
                panels = []
                for camera in (follow, overview):
                    renderer.update_scene(data, camera=camera)
                    if camera is overview:
                        for j in range(0,i+1,10):
                            scene = renderer.scene
                            if scene.ngeom >= scene.maxgeom:
                                break
                            color = colors.get(str(trajectory['phase'][j]), (170,170,170))
                            mujoco.mjv_initGeom(scene.geoms[scene.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                               np.array([.004,.004,.004]), np.array([*xy[j],.006]),
                                               np.eye(3).ravel(), np.array([*(v/255 for v in color),1.]))
                            scene.ngeom += 1
                    panels.append(Image.fromarray(renderer.render().copy()))
                frame = Image.new('RGB', (1280,544), (20,25,32))
                frame.paste(panels[0],(0,64))
                frame.paste(panels[1],(640,64))
                draw = ImageDraw.Draw(frame)
                phase = str(trajectory['phase'][i])
                draw.text((12,7),f"{phase.upper()}  |  t={data.time:05.2f}s  |  1x playback",font=font,fill=colors.get(phase,'white'))
                cmd = trajectory['command'][i]
                draw.text((12,34),f"Command: vx={cmd[0]:+.3f} m/s, yaw={cmd[2]:+.2f} rad/s  |  up_z={trajectory['up_z'][i]:.3f}",font=small,fill='white')
                draw.text((660,8),'TOP VIEW: colored dots = recorded travel path',font=small,fill='white')
                draw.text((660,34),'Continuous simulation | no resets | no hardware',font=small,fill='white')
                writer.send(np.asarray(frame))
                if i in sample_indices:
                    samples.append(frame.copy())
                if i%200==0:
                    print(f'RENDER {i}/{len(xy)}',flush=True)
        finally:
            writer.close()
    sheet = Image.new('RGB',(1280,544*len(samples)),(20,25,32))
    for k,frame in enumerate(samples):
        sheet.paste(frame,(0,544*k))
    sheet.save(output/'contact_sheet.jpg')
    if samples:
        samples[1 if len(samples)>1 else 0].save(output/'preview.jpg')
    count, duration = imageio_ffmpeg.count_frames_and_secs(str(video))
    if count != math.ceil(len(xy)/2):
        raise RuntimeError('video frame count mismatch')
    reader = imageio_ffmpeg.read_frames(str(video))
    metadata = next(reader)
    next(reader)
    reader.close()
    (output/'video_metadata.json').write_text(json.dumps(dict(frames=count,duration_s=duration,metadata=metadata,
                                                            bytes=video.stat().st_size),indent=2))
    print('VIDEO SAVED:', video, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/data/shijinsheng/open_duck'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=1600)
    parser.add_argument('--no-render',action='store_true')
    parser.add_argument('--legacy-shared-controls',action='store_true',help='Diagnostic reproduction of the old recording only')
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model, trajectory, summary = rollout(args.root,args.output,args.seed,args.legacy_shared_controls)
    if not args.no_render:
        render(model,trajectory,args.output)
    if not summary['completed']:
        raise RuntimeError('Recorded sequence contains a fall; inspect results, not a successful showcase')
