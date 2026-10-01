"""Per-start executable early-leg rescue search, NOT a general recovery actor.

Each candidate is a separate deterministic replay from the same initialization.
No root/state edits after reset, no physics or success-gate changes. Only targets
of eight leg joints change during the first 1.2 s; then exact home is commanded.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from diagnostics.getup_native_curriculum import NativeEpisode, at_goal, physical_safe, reward_terms
from diagnostics.getup_independent_native import DT, SLEW, digest


INDICES = np.array([1, 2, 3, 4, 10, 11, 12, 13])
BOUNDS = np.array([.7, .9, .8, .9, .7, .9, .8, .9])
_episode = None


def target_at(sim, knots, elapsed):
    knots = np.asarray(knots, dtype=float)
    if knots.shape != (2, 8) or not np.isfinite(knots).all():
        raise ValueError('two finite eight-joint knots required')
    if np.any(np.abs(knots) > BOUNDS + 1e-8):
        raise ValueError('rescue knots outside declared bounds')
    if elapsed < .2:
        delta = knots[0]
    elif elapsed < .6:
        alpha = (elapsed - .2) / .4
        delta = (1 - alpha) * knots[0] + alpha * knots[1]
    elif elapsed < 1.2:
        delta = (1.2 - elapsed) / .6 * knots[1]
    else:
        delta = np.zeros(8)
    target = sim.home.copy()
    target[INDICES] += delta
    return np.clip(target, sim.lower, sim.upper)


def init_worker(contract_path, tilt):
    global _episode
    c = json.loads(Path(contract_path).read_text())
    _episode = NativeEpisode(c['scene_path'],
        '/data/shijinsheng/open_duck/projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx',
        c['starts_rad'], 1, tilt)


def replay(seed, knots, steps=200, record=False):
    e = _episode
    e.rng = np.random.default_rng(seed)
    e.reset()
    sim = e.sim
    initial = sim.measure()
    initial_hash = hashlib.sha256(sim.data.qpos.tobytes() + sim.data.qvel.tobytes()).hexdigest()
    initial_obs = sim.observation().tolist()
    safe = True; tail = 0; ret = 0.; success4 = False
    previous_height = initial['height_m']; before = sim.prev.copy()
    max_slew = 0.; min_up = initial['up_z']; trace = []; arrays = []
    for step in range(steps):
        target = target_at(sim, knots, step * DT)
        if record:
            arrays.append((sim.observation(), target.copy(), sim.data.qpos.copy(), sim.data.qvel.copy()))
        applied = sim.step_target(target)
        max_slew = max(max_slew, float(np.abs(applied - before).max() / DT))
        if max_slew > SLEW + 1e-6:
            raise RuntimeError('motor target slew violated')
        before = applied.copy()
        m = sim.measure(); valid = physical_safe(sim, m); safe = safe and valid
        min_up = min(min_up, m['up_z'])
        ret += sum(reward_terms(m, previous_height, m['joint_home_error_mean_rad']).values()) if valid else -20.
        previous_height = m['height_m']
        tail = tail + 1 if at_goal(m) else 0
        if step == 199:
            success4 = bool(safe and tail >= 100)
        if record or step % 10 == 0:
            trace.append(dict(step=step+1, height=m['height_m'], up_z=m['up_z'],
                gyro=sim.sensor('gyro').tolist(), foot_load=m['foot_load_fraction'],
                goal=at_goal(m), support_margin=m['com_support_margin_m']))
        if not valid or m['up_z'] < .45:
            break
    completed = step + 1
    success = bool(completed == steps and tail >= 100 and safe)
    row = dict(seed=seed, initial_hash=initial_hash, initial=initial, initial_obs=initial_obs,
        steps=completed, success=success, success_4s=success4, safe=safe, final=m,
        stable_tail_steps=tail, min_up_z=min_up, max_target_slew_rad_s=max_slew,
        return_sum=ret, score=float(100000 * success + 10 * completed + ret + 15 * tail), trace=trace)
    return row, arrays


def scan(seed):
    return replay(seed, np.zeros((2, 8)))[0]


def search(job):
    seed, population, generations, output = job
    output = Path(output)
    baseline, _ = replay(seed, np.zeros((2, 8)))
    if baseline['success']:
        raise ValueError('search must target a failed baseline start')
    best = np.zeros((2, 8)); best_row = baseline
    mean = best.copy(); std = np.broadcast_to(BOUNDS * .4, (2, 8)).copy()
    rng = np.random.default_rng(seed + 422); history = []; controls = baseline['steps']
    started = time.time()
    for generation in range(generations):
        candidates = np.clip(rng.normal(mean, std, (population, 2, 8)), -BOUNDS, BOUNDS)
        candidates[0] = best; candidates[1] = 0.
        rows = []
        for knots in candidates:
            row, _ = replay(seed, knots)
            if row['initial_hash'] != baseline['initial_hash']:
                raise RuntimeError('candidate initialization differs from baseline')
            controls += row['steps']; rows.append(row)
        order = np.argsort([r['score'] for r in rows])
        elites = candidates[order[-max(4, population // 5):]]
        mean = .25 * mean + .75 * elites.mean(axis=0)
        std = np.maximum(.25 * std + .75 * elites.std(axis=0), .025)
        index = int(order[-1])
        if rows[index]['score'] > best_row['score']:
            best = candidates[index].copy(); best_row = rows[index]
        history.append(dict(generation=generation+1, best_success=best_row['success'],
            successful_candidates=sum(r['success'] for r in rows), best_steps=best_row['steps'],
            best_score=best_row['score'], control_steps=controls, elapsed_s=time.time()-started))
        (output / f'seed_{seed}_progress.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print('RESCUE SEARCH:', seed, json.dumps(history[-1]), flush=True)
    replayed, arrays = replay(seed, best, record=True)
    if replayed['initial_hash'] != baseline['initial_hash'] or replayed['success'] != best_row['success']:
        raise RuntimeError('selected trajectory did not reproduce')
    long, _ = replay(seed, best, steps=1500)
    home_long, _ = replay(seed, np.zeros((2, 8)), steps=1500)
    if len({r['initial_hash'] for r in (baseline, replayed, long, home_long)}) != 1:
        raise RuntimeError('paired long replay mismatch')
    controls += replayed['steps'] + long['steps'] + home_long['steps']
    obs, targets, qpos, qvel = map(np.asarray, zip(*arrays))
    np.savez_compressed(output / f'seed_{seed}_replay.npz', obs=obs, targets=targets,
        qpos=qpos, qvel=qvel, elapsed=np.arange(len(obs))*DT, knots=best)
    result = dict(seed=seed, baseline=baseline, selected=replayed, selected_long=long,
        baseline_long=home_long, knots=best.tolist(), history=history, actual_control_steps=controls,
        training_start_fitted=True, general_policy=False, initialization_only_state_edits=True,
        simulation_only=True, hardware_readiness=False)
    (output / f'seed_{seed}_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--contract', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed-base', type=int, default=310000); p.add_argument('--scan-seeds', type=int, default=20)
    p.add_argument('--failures', type=int, default=4); p.add_argument('--population', type=int, default=32)
    p.add_argument('--generations', type=int, default=12); p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if min(args.failures, args.scan_seeds, args.generations, args.workers) < 1 or args.population < 4:
        raise ValueError('invalid search budget')
    args.output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
            initargs=(str(args.contract), .55)) as pool:
        scanned = list(pool.map(scan, range(args.seed_base, args.seed_base+args.scan_seeds)))
        (args.output/'baseline_scan.json').write_text(json.dumps(scanned, indent=2), encoding='utf-8')
        failed = [r for r in scanned if not r['success']][:args.failures]
        print('BASELINE SCAN:', sum(r['success'] for r in scanned), '/', len(scanned),
              'SEARCH SEEDS:', [r['seed'] for r in failed], flush=True)
        jobs = [(r['seed'], args.population, args.generations, str(args.output)) for r in failed]
        results = [f.result() for f in as_completed([pool.submit(search, j) for j in jobs])]
    c = json.loads(args.contract.read_text())
    report = dict(results=results, scanned_runs=len(scanned), searched_failed_starts=len(results),
        rescued_4s=sum(r['selected']['success'] for r in results),
        rescued_30s=sum(r['selected_long']['success'] for r in results),
        actual_control_steps=sum(r['steps'] for r in scanned)+sum(r['actual_control_steps'] for r in results),
        optimization='per-start trajectory CEM, NOT PPO or trained general actor',
        physics_unchanged=True, success_gate_unchanged=True, root_edits_after_reset=0,
        target_knots_s=[.2, .6, 1.2], varied_joint_indices=INDICES.tolist(),
        knot_bounds_rad=BOUNDS.tolist(), simulation_only=True, hardware_readiness=False,
        stage='near-standing rescue only, NOT full fallen get-up',
        hashes={str(f):digest(f) for f in (Path(__file__), args.contract, Path(c['scene_path']))})
    (args.output/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('RESCUE COMPLETE:', json.dumps({k:v for k,v in report.items() if k not in ('results','hashes')}), flush=True)


if __name__ == '__main__':
    main()
