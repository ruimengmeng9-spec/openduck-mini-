"""Full-fallen, whole-path trajectory optimization (not near-standing/PPO).

Recovery may temporarily lose height or reverse knee support. Optimize the whole
sequence instead of discarding such transitions in a greedy beam. Never mutate
root state after initialization. Preserve collision model and motor limits.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np

from diagnostics.audit_getup_load_support import LoadSupportSim
from diagnostics.getup_aligned_extension import AlignedSim
from diagnostics.getup_independent_native import DT, SLEW, POSES, digest
from diagnostics.getup_load_search import loaded_entry


class AuditedSim(LoadSupportSim):
    """Same dynamics; lightweight support metrics and substep limit audit."""
    def clear_audit(self):
        self.peaks = dict(floor=0., self=0., joint=0., force=0., slew=0.)
        self.finite = True

    def measure(self):
        m = AlignedSim.measure(self)
        by_body = {}
        for i, c in enumerate(self.data.contact):
            if self.floor not in c.geom or c.dist > .001:
                continue
            other = int(c.geom[1] if c.geom[0] == self.floor else c.geom[0])
            b = int(self.model.geom_bodyid[other])
            f = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, i, f)
            by_body[b] = by_body.get(b, 0.) + max(float(f[0]), 0.)
        feet = [by_body.get(b, 0.) for b in self.feet]
        m['foot_normal_forces_n'] = feet
        m['foot_load_fraction'] = sum(feet)/max(sum(by_body.values()), 1e-9)
        m['foot_up_alignment_to_home'] = [
            float((self.data.xmat[b].reshape(3, 3) @ axis)[2])
            for b, axis in zip(self.feet, self.home_up_axes)]
        m['stable'] = bool(loaded_entry(m))
        return m

    def step_target(self, target):
        target = np.clip(np.asarray(target), self.lower, self.upper)
        applied = np.clip(target, self.prev-SLEW*DT, self.prev+SLEW*DT)
        self.peaks['slew'] = max(self.peaks['slew'], float(np.abs(applied-self.prev).max()/DT))
        self.history[2] = self.history[1].copy()
        self.history[1] = self.history[0].copy()
        self.history[0] = ((target-self.home)/.25).astype(np.float32)
        self.prev = applied.copy()
        self.data.ctrl[:] = applied
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            self.finite &= bool(np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
            self.peaks['force'] = max(self.peaks['force'], float(np.abs(self.data.actuator_force).max()))
            q = self.data.qpos[self.qadr]
            self.peaks['joint'] = max(self.peaks['joint'], float(np.maximum(self.lower-q, q-self.upper).max()))
            for c in self.data.contact:
                k = 'floor' if self.floor in c.geom else 'self'
                self.peaks[k] = max(self.peaks[k], float(-c.dist))
        return applied

    def physical_valid(self):
        p = self.peaks
        return bool(self.finite and p['floor'] < .01 and p['self'] < .004
                    and p['joint'] < .08 and p['force'] <= 3.23+1e-8 and p['slew'] <= SLEW+1e-8)

    def __init__(self, scene, stand):
        super().__init__(scene, stand)
        self.clear_audit()


def progress(m):
    """Search shaping only. No shaping threshold counts as recovery success."""
    upright = np.clip((m['up_z']+.2)/1.2, 0., 1.)
    lift = np.clip((m['height_m']-.04)/.12, 0., 1.)
    feet = np.clip(m['foot_load_fraction'], 0., 1.)
    flat = np.clip(min(m['foot_up_alignment_to_home']), 0., 1.)
    home = np.exp(-2*m['joint_home_error_mean_rad'])
    return float(2*upright+2*upright*lift+3*upright*lift*feet*flat
                 +2*upright*lift*home - .15*m['torso_contact'])


def state_hash(sim):
    h = hashlib.sha256()
    for x in (sim.data.qpos, sim.data.qvel, sim.prev, sim.history):
        h.update(np.asarray(x).tobytes())
    return h.hexdigest()


def rollout(sim, pose, targets, durations, seed=0, perturb=False, hold_s=4., record=False):
    sim.prepare(pose, seed, perturb)
    initial, initial_hash = sim.measure(), state_hash(sim)
    sim.clear_audit()
    # A real fallen start is checked after settling, not inferred from a label.
    initial_fallen = initial['up_z'] < .5 and initial['torso_contact']
    trace, measures, reached, stable_steps, controls = [], [], False, 0, 0
    def advance(target, phase):
        nonlocal controls, stable_steps
        sim.step_target(target)
        m = sim.measure()
        controls += 1
        stable_steps = stable_steps+1 if m['stable'] else 0
        measures.append(m)
        if record:
            trace.append((float(sim.data.time), sim.data.qpos.copy(), sim.data.qvel.copy(), sim.prev.copy(), phase))
        return m
    for target, duration in zip(targets, durations):
        start = sim.prev.copy()
        n = max(1, round(float(duration)/DT))
        for i in range(n):
            a = min((i+1)/(.7*n), 1.)
            a = a*a*(3-2*a)
            m = advance((1-a)*start+a*target, 'recovery')
            if not sim.physical_valid():
                break
        if not sim.physical_valid():
            break
        if loaded_entry(m):
            reached = True
            break
    recovery_end = sim.measure()
    # Only a qualified physical state may enter the walking/standing actor.
    if sim.physical_valid():
        start = sim.prev.copy()
        for i in range(round(hold_s/DT)):
            if reached:
                a = min((i+1)*DT, 1.)
                a = a*a*(3-2*a)
                target = (1-a)*start+a*(sim.home+.25*sim.stand_action())
            else:
                target = sim.home
            advance(target, 'stand_actor' if reached else 'failed_entry_home_hold')
            if not sim.physical_valid():
                break
    valid = sim.physical_valid()
    tail = measures[-min(50, len(measures)):]
    quality = float(np.mean([progress(m) for m in tail])) if tail else -10.
    best = max((progress(m) for m in measures), default=-10.)
    success = bool(initial_fallen and reached and valid and stable_steps*DT >= hold_s-1.-1e-8)
    # Penalize invalid trajectories; do not trade physical violations for height.
    score = (1000.*success + 15*reached + 2*stable_steps*DT + quality
             + 1.5*progress(recovery_end) + .15*best
             if valid and initial_fallen else -100.+controls*.0001)
    result = dict(success=success, entry_reached=reached, valid=valid,
                  initial_fallen=bool(initial_fallen), initial=initial, initial_hash=initial_hash,
                  recovery_end=recovery_end, final=sim.measure(), score=float(score),
                  sustained_tail_s=stable_steps*DT, control_steps=controls,
                  peaks=sim.peaks.copy(), seed=seed, hold_s=hold_s)
    return result, trace


_sim = None


def init_worker(scene, stand):
    global _sim
    _sim = AuditedSim(scene, stand)


def evaluate(job):
    pose, targets, durations, seed, perturb, hold = job
    return rollout(_sim, pose, targets, durations, seed, perturb, hold)[0]


def encode(targets, durations, sim):
    return np.concatenate([((targets-sim.lower)/(sim.upper-sim.lower)*2-1),
                           durations[:, None]], axis=1)


def decode(candidate, sim):
    q = sim.lower+(np.clip(candidate[:, :14], -1, 1)+1)*.5*(sim.upper-sim.lower)
    q[:, [7, 8]] = sim.home[[7, 8]]
    return q, np.clip(candidate[:, 14], .12, .9)


def initial_candidates(root, sim, pose, knots):
    priors = []
    # Earlier failed paths are initialization only, never successful teachers.
    for name in ('getup_dynamic_r4b_'+pose, 'getup_load_r8_'+pose,
                 'getup_aligned_r6_'+pose):
        p = root/'training'/name/'best_reference.npz'
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=False)
        if 'durations_s' not in z.files or len(z['targets']) > knots-2:
            continue
        q = np.tile(sim.home, (knots, 1))
        d = np.full(knots, .3)
        n = len(z['targets'])
        q[:n], d[:n] = z['targets'], z['durations_s']
        priors.append(encode(q, d, sim))
    home = encode(np.tile(sim.home, (knots, 1)), np.full(knots, .3), sim)
    priors.append(home)
    return priors


def save_trace(path, trace):
    if trace:
        np.savez_compressed(path, time=[x[0] for x in trace], qpos=[x[1] for x in trace],
                            qvel=[x[2] for x in trace], ctrl=[x[3] for x in trace], phase=[x[4] for x in trace])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/data/shijinsheng/open_duck'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pose', choices=POSES, required=True)
    p.add_argument('--generations', type=int, default=64)
    p.add_argument('--population', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--knots', type=int, default=14)
    p.add_argument('--seed', type=int, default=527)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scene = args.root/'training/getup_decomposed_r4/model/scene.xml'
    stand = args.root/'projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx'
    qualification = json.loads((scene.parent/'qualification.json').read_text())
    if qualification['standing_passed_runs'] != qualification['standing_runs']:
        raise RuntimeError('collision model has not passed standing qualification')
    sim = AuditedSim(scene, stand)
    source = args.output/'executed_sources'
    source.mkdir()
    for name in ('train_getup_fullpath_r27.py', 'audit_getup_load_support.py',
                 'audit_getup_crouch.py', 'getup_aligned_extension.py', 'getup_dynamic_beam.py',
                 'getup_independent_native.py', 'getup_load_search.py'):
        shutil.copy2(Path(__file__).parent/name, source/name)
    contract = dict(method='whole-path mixed CEM / coordinate mutations', pose=args.pose,
                    simulation_only=True, hardware_readiness=False, root_edits_after_initialization=0,
                    completion_gate='all four orientations >=18/20 independent perturbed starts; loaded standing for 30s after 1s handoff',
                    seed=args.seed, generation_budget=args.generations, population=args.population,
                    workers=args.workers, scene_path=str(scene), substep_audits=True,
                    limits=dict(floor_penetration_m=.01, self_penetration_m=.004, joint_overshoot_rad=.08,
                                force_nm=3.23, slew_rad_s=SLEW),
                    hashes={str(f): digest(f) for f in (scene, stand, Path(__file__))})
    (args.output/'contract.json').write_text(json.dumps(contract, indent=2))
    priors = initial_candidates(args.root, sim, args.pose, args.knots)
    rng = np.random.default_rng(args.seed)
    mean, std = priors[0].copy(), np.full_like(priors[0], .55)
    std[:, 14] = .2
    archive, history, best, controls = [], [], None, 0
    start_time = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             initargs=(str(scene), str(stand))) as pool:
        for gen in range(args.generations):
            population = []
            for i in range(args.population):
                if i < len(priors) and gen == 0:
                    x = priors[i].copy()
                elif best is not None and i == 0:
                    x = best[0].copy()
                elif archive and i % 2:
                    x = archive[int(rng.integers(len(archive)))][0].copy()
                    # Change one contiguous section, retaining useful prefixes.
                    a = int(rng.integers(args.knots))
                    b = min(a+int(rng.integers(1, 5)), args.knots)
                    x[a:b, :14] += rng.normal(0, .25 if gen%4 else .65, (b-a, 14))
                    x[a:b, 14] += rng.normal(0, .12, b-a)
                else:
                    x = rng.normal(mean, std)
                q, d = decode(x, sim)
                population.append(encode(q, d, sim))
            jobs = [(args.pose, *decode(x, sim), 0, False, 4.) for x in population]
            rows = list(pool.map(evaluate, jobs, chunksize=1))
            controls += sum(r['control_steps'] for r in rows)
            ranked = sorted(zip(population, rows), key=lambda item: -item[1]['score'])
            archive = sorted(archive+ranked[:6], key=lambda item: -item[1]['score'])
            # Diversity by end posture + trajectory, rather than only highest height.
            kept, keys = [], set()
            for x, r in archive:
                m = r['recovery_end']
                key = (round(m['up_z']/.1), round(m['height_m']/.01),
                       round(m['foot_load_fraction']/.1), *np.round(x[-4:, [3, 12]]/.3).astype(int).ravel())
                if key not in keys:
                    keys.add(key); kept.append((x, r))
                if len(kept) == 12:
                    break
            archive = kept
            best = archive[0]
            elites = np.asarray([x for x, r in ranked[:max(4, args.population//5)]])
            mean = .4*mean+.6*elites.mean(axis=0)
            std = np.maximum(.4*std+.6*elites.std(axis=0), .12)
            std[:, 14] = np.maximum(std[:, 14], .05)
            q, d = decode(best[0], sim)
            np.savez_compressed(args.output/'best_reference.npz', targets=q, durations_s=d)
            np.savez_compressed(args.output/'checkpoint.npz', mean=mean, std=std,
                                archive=np.asarray([x for x, r in archive]))
            row = dict(generation=gen+1, best=best[1], valid_candidates=sum(r['valid'] for r in rows),
                       nominal_successes=sum(r['success'] for r in rows), control_steps_total=controls,
                       wall_seconds=time.monotonic()-start_time)
            history.append(row)
            (args.output/'search_progress.json').write_text(json.dumps(history, indent=2))
            print(json.dumps(dict(generation=gen+1, score=best[1]['score'], success=best[1]['success'],
                                  entry=best[1]['entry_reached'], height=best[1]['final']['height_m'],
                                  up=best[1]['final']['up_z'], loaded=best[1]['final']['foot_load_fraction'],
                                  valid=row['valid_candidates'], seconds=row['wall_seconds'])), flush=True)
            if best[1]['success']:
                break
        q, d = decode(best[0], sim)
        # Fresh starts do not participate in search or selection.
        training_controls = controls
        validation = list(pool.map(evaluate, [(args.pose, q, d, 600000+i, True, 31.) for i in range(20)], chunksize=1))
        validation_controls = sum(r['control_steps'] for r in validation)
    nominal, trace = rollout(sim, args.pose, q, d, hold_s=31., record=True)
    save_trace(args.output/'nominal_trajectory.npz', trace)
    # Always save at least one perturbed failure for review.
    failed = next((r for r in validation if not r['success']), None)
    if failed:
        _, trace = rollout(sim, args.pose, q, d, failed['seed'], True, 31., True)
        save_trace(args.output/'failure_trajectory.npz', trace)
    summary = dict(contract=contract, training_controls=training_controls,
                   validation_controls=validation_controls, nominal=nominal,
                   successful_validation_runs=sum(r['success'] for r in validation), validation_runs=20,
                   orientation_passed=sum(r['success'] for r in validation)>=18,
                   full_task_completed=False, validation=validation)
    (args.output/'results.json').write_text(json.dumps(summary, indent=2))
    print('VALIDATION', summary['successful_validation_runs'], '/20', flush=True)


if __name__ == '__main__':
    main()
