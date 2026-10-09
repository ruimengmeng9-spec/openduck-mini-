"""R2 reference search with an independently verified, state-gated stand handoff.

Open-loop references are an exploration tool, NOT an autonomous get-up actor.
The gate never edits robot pose or velocities and never injects external forces.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from diagnostics.getup_independent_native import (
    DT, SLEW, RecoverySim, build_model, digest, sustained_tail, POSES)


def ready_to_handoff(m):
    return bool(m['up_z'] > .85 and .115 < m['height_m'] < .23
                and any(m['feet']) and not m['torso_contact']
                and m['angular_speed_rad_s'] < 2. and m['linear_speed_mps'] < .35)


def run(sim, initial, targets, duration, record=False):
    sim.restore(initial)
    sim.data.ctrl[:] = sim.prev
    mujoco.mj_forward(sim.model, sim.data)
    initial_measure = sim.measure()
    knots = np.concatenate([sim.home[None], targets])
    flags, measures, trace = [], [], []
    handoff_t, handoff_target, ready_count = None, None, 0
    max_force, max_slew, max_joint_overshoot = 0., 0., 0.
    final_target = targets[-1]
    previous = sim.prev.copy()
    for i in range(round((duration+5.)/DT)):
        t = i*DT
        if handoff_t is None:
            before = sim.measure()
            ready_count = ready_count+1 if ready_to_handoff(before) else 0
            if ready_count >= 5 or t >= duration+1.:
                handoff_t, handoff_target = t, sim.prev.copy()
        if handoff_t is not None:
            raw = sim.stand_action()
            stand = sim.home+.25*raw
            blend = min((t-handoff_t)/1., 1.)
            blend = blend*blend*(3.-2.*blend)
            target = (1-blend)*handoff_target+blend*stand
        elif t >= duration:
            target = final_target
        else:
            f = t/duration*(len(knots)-1)
            j = min(int(f), len(knots)-2)
            a = f-j
            a = a*a*(3.-2.*a)
            target = (1-a)*knots[j]+a*knots[j+1]
        applied = sim.step_target(target)
        max_force = max(max_force, float(np.abs(sim.data.actuator_force).max()))
        max_slew = max(max_slew, float(np.abs(applied-previous).max()/DT))
        previous = applied.copy()
        joints = sim.data.qpos[sim.qadr]
        max_joint_overshoot = max(max_joint_overshoot, float(np.maximum(sim.lower-joints, joints-sim.upper).max()))
        m = sim.measure()
        flags.append(m['stable'])
        measures.append(m)
        if record:
            trace.append((float(sim.data.time), sim.data.qpos.copy(), sim.data.qvel.copy(), applied.copy(),
                          'stand_handoff' if handoff_t is not None else 'reference'))
        if not m['finite']:
            break
    stable_s = sustained_tail(flags)
    penetration = max(x['penetration_m'] for x in measures)
    fallen = initial_measure['up_z'] < .5 and initial_measure['torso_contact']
    safe = all(x['finite'] for x in measures) and penetration < .01 and max_joint_overshoot < .08
    success = fallen and safe and stable_s >= 2.-1e-8 and handoff_t is not None
    # Dense progress rewards a maintained intermediate support state, not a
    # single airborne upright sample. Final success remains a separate test.
    quality = np.array([2.*np.clip(m['up_z'], -1., 1.)
                        + min(m['height_m']/.16, 1.2)
                        + .35*sum(m['feet'])-.4*m['torso_contact']
                        -.08*min(m['angular_speed_rad_s'], 8.) for m in measures])
    window = np.convolve(quality, np.ones(25)/25, mode='valid')
    score = 1.5*window.max()+quality[-100:].mean()+3.*stable_s+40.*success
    score -= 40.*penetration+2.*max_joint_overshoot
    return dict(success=bool(success), score=float(score), initial_fallen=bool(fallen),
                initial=initial_measure, final=measures[-1], sustained_final_stand_s=stable_s,
                maximum_up_z=max(m['up_z'] for m in measures), max_support_window_score=float(window.max()),
                maximum_penetration_m=penetration, maximum_joint_overshoot_rad=max_joint_overshoot,
                max_sampled_actuator_force_nm=max_force, max_command_slew_rad_s=max_slew,
                handoff_time_s=handoff_t, duration_s=len(measures)*DT,
                simulation_only=True, hardware_readiness=False), trace


_context = None


def init_worker(scene, stand, pose, duration):
    global _context
    sim = RecoverySim(scene, stand)
    _context = sim, sim.prepare(pose), duration


def evaluate(targets):
    sim, initial, duration = _context
    return run(sim, initial, targets, duration)[0]


def render(scene, trajectory_path, output):
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    m = mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    trajectory = np.load(trajectory_path, allow_pickle=False)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = .95, 120., -22.
    m.vis.global_.offwidth, m.vis.global_.offheight = 640, 480
    writer = imageio_ffmpeg.write_frames(str(output/'recovery_search_preview.mp4'), (640, 480),
                                        fps=25, codec='libx264', quality=7,
                                        output_params=['-movflags', '+faststart'])
    writer.send(None)
    with mujoco.Renderer(m, 480, 640) as renderer:
        try:
            for i in range(0, len(trajectory['time']), 2):
                d.qpos[:] = trajectory['qpos'][i]
                d.qvel[:] = trajectory['qvel'][i]
                d.time = float(trajectory['time'][i])
                mujoco.mj_forward(m, d)
                cam.lookat[:] = [*d.qpos[:2], .12]
                renderer.update_scene(d, camera=cam)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 42), fill='black')
                draw.text((10, 4), 'INDEPENDENT GET-UP SEARCH | SIMULATION ONLY | NOT A DEPLOYABLE POLICY', fill='white')
                draw.text((10, 22), f"t={d.time:.2f}s  {trajectory['phase'][i]}", fill='white')
                writer.send(np.asarray(frame))
                if i in (0, 100, 200, 300, 450):
                    frame.save(output/f'frame_{i:04d}.png')
        finally:
            writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pose', choices=POSES, default='supine')
    parser.add_argument('--generations', type=int, default=24)
    parser.add_argument('--population', type=int, default=96)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--knots', type=int, default=5)
    parser.add_argument('--duration', type=float, default=4.)
    parser.add_argument('--seed', type=int, default=83)
    parser.add_argument('--warm-reference', type=Path)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.render:
        render(args.output/'model/scene.xml', args.output/'best_trajectory.npz', args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    scene, audit = build_model(args.root, args.output/'model')
    stand = args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    sim = RecoverySim(scene, stand)
    mean = np.tile(sim.home, (args.knots, 1))
    if args.warm_reference:
        reference = np.load(args.warm_reference, allow_pickle=False)['targets']
        if reference.shape != mean.shape:
            raise ValueError('warm reference knot shape differs')
        mean[:] = reference
    std = np.tile((sim.upper-sim.lower)*.35, (args.knots, 1))
    # Keep head yaw and roll close to home; independent legs can still roll the body.
    std[:, [7, 8]] *= .2
    best, best_score, history = mean.copy(), -np.inf, []
    rng, start = np.random.default_rng(args.seed), time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand), args.pose, args.duration)) as pool:
        for generation in range(args.generations):
            samples = np.clip(rng.normal(mean, std, (args.population, args.knots, sim.model.nu)), sim.lower, sim.upper)
            samples[0], samples[1] = best, mean
            # A home endpoint is included in half the population, not forced in all.
            samples[2:args.population//2, -1] = sim.home
            results = list(pool.map(evaluate, samples, chunksize=1))
            scores = np.array([r['score'] for r in results])
            elites = samples[np.argsort(scores)[-max(6, args.population//8):]]
            mean = .3*mean+.7*elites.mean(axis=0)
            floor_std = np.tile((sim.upper-sim.lower)*.025, (args.knots, 1))
            floor_std[:, [7, 8]] *= .2
            std = np.maximum(.3*std+.7*elites.std(axis=0), floor_std)
            j = int(scores.argmax())
            if scores[j] > best_score:
                best_score, best = float(scores[j]), samples[j].copy()
                np.savez_compressed(args.output/'best_reference.npz', targets=best, duration_s=args.duration)
            row = dict(generation=generation, best_score=best_score,
                       successful_candidates=sum(r['success'] for r in results),
                       best_this_generation=results[j], wall_seconds=time.time()-start)
            history.append(row)
            (args.output/'search_progress.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
    results = []
    for seed in range(20):
        initial = sim.prepare(args.pose, 10000+seed, perturb=seed>0)
        result, trace = run(sim, initial, best, args.duration, record=seed == 0)
        result['seed'] = 10000+seed
        results.append(result)
        if trace:
            np.savez_compressed(args.output/'best_trajectory.npz', time=[t[0] for t in trace],
                                qpos=[t[1] for t in trace], qvel=[t[2] for t in trace],
                                ctrl=[t[3] for t in trace], phase=[t[4] for t in trace])
    summary = dict(pose=args.pose, method='independent asymmetric CEM references with state-gated stand handoff',
                   successful_validation_runs=sum(r['success'] for r in results), validation_runs=20,
                   results=results, audit=audit, pose_seed=args.seed, generations=args.generations,
                   population=args.population, knot_count=args.knots, duration_s=args.duration,
                   script_sha256=digest(__file__), native_sim_sha256=digest(Path(__file__).with_name('getup_independent_native.py')),
                   stand_policy_sha256=digest(stand), hardware_readiness=False,
                   controller_contract=dict(action='absolute_joint_targets_rad', interpolation='smoothstep',
                                            joint_limits='original MuJoCo joint ranges', slew_rad_s=SLEW,
                                            torque='unchanged original actuator force ranges', handoff='state gate or timeout; 1s blend'))
    (args.output/'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('VALIDATION:', summary['successful_validation_runs'], '/20', flush=True)


if __name__ == '__main__':
    main()
