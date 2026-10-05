"""R108 factorial initial-state diagnosis, not relaxed qualification.

Each trial starts a real fall, settles with the unchanged forty home controls,
then runs the unchanged full reference path. Disabled initial perturbations
are explicitly diagnostic ablations and never independent success evidence.
Full/no-perturbation initialization must match the original implementation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import shutil
from types import MethodType
import mujoco
import numpy as np
from diagnostics.getup_independent_native import canonical_quaternion, digest
from diagnostics.getup_reference_env_r100 import ReferenceEpisode, TRAIN
from diagnostics.probe_getup_anchor_r101 import ROOT, SCENE, STAND, REFERENCE
from diagnostics import train_getup_joint_anchor_r102 as r102
from diagnostics.train_getup_joint_anchor_r102 import write_json

GAINS = None


def draw_components(seed, nu, nv):
    rng = np.random.default_rng(seed)
    # Always draw all three fields in original order, even when masked.
    return rng.uniform(-.05, .05), rng.uniform(-.02, .02, nu), rng.uniform(-.01, .01, nv)


def prepare_components(sim, pose, seed=0, perturb=False):
    mujoco.mj_resetDataKeyframe(sim.model, sim.data, sim.model.keyframe('home').id)
    if perturb:
        tilt, joints, velocity = draw_components(seed, sim.model.nu, sim.model.nv)
    else:
        tilt, joints, velocity = 0., np.zeros(sim.model.nu), np.zeros(sim.model.nv)
    mask = sim.component_mask
    if pose != 'standing':
        sim.data.qpos[3:7] = canonical_quaternion(pose, tilt if mask[0] else 0.)
    if perturb and mask[1]:
        sim.data.qpos[sim.qadr] = np.clip(sim.home+joints, sim.lower, sim.upper)
    sim.data.qpos[2] = 0.
    mujoco.mj_forward(sim.model, sim.data)
    sim.data.qpos[2] += .005-sim.lowest_geom()
    sim.data.qvel[:] = velocity if perturb and mask[2] else 0.
    sim.prev, sim.history = sim.home.copy(), np.zeros_like(sim.history)
    mujoco.mj_forward(sim.model, sim.data)
    for _ in range(40):
        sim.step_target(sim.home)
    return sim.snapshot()


def make_env(mask, seed):
    env = ReferenceEpisode(str(SCENE), str(STAND), REFERENCE, 208, True)
    env.sim.component_mask = tuple(mask)
    env.sim.prepare = MethodType(prepare_components, env.sim)
    env.reset(seed)
    return env


def init_worker(anchors, gains):
    global GAINS
    r102.init_worker(anchors)
    GAINS = np.asarray(gains)


def run(job):
    mask, profile, seed, directory, parity_directory = job
    env = make_env(mask, seed)
    initial_obs = env.observe().copy()
    trace, actions = [], []
    while True:
        action = np.zeros(10) if profile == 'zero' else r102.action_for(
            r102.WEIGHTS, env.observe(), r102.ANCHORS[env.controls], GAINS)
        step = env.step(action, auto_reset=False)
        actions.append(action)
        trace.append((env.sim.data.time, env.sim.data.qpos.copy(), env.sim.data.qvel.copy(),
                      env.sim.prev.copy(), env.tail > 0))
        if step[2]:
            break
    row = step[5]
    row.update(mask=list(mask), profile=profile, diagnostic_only=True,
        qualification_evidence=False, initial_observation=initial_obs.tolist(),
        success=bool(row['valid'] and row['controls'] == len(env.targets)
            and row['entry_time_s'] is not None and row['entry_time_s'] <= 12.
            and row['strict_tail_s'] >= 30.-1e-8))
    arrays = dict(time=[r[0] for r in trace], qpos=[r[1] for r in trace],
        qvel=[r[2] for r in trace], applied=[r[3] for r in trace],
        strict=[r[4] for r in trace], normalized_residual=actions,
        initial_observation=initial_obs)
    if parity_directory:
        with np.load(Path(parity_directory)/f'case_{seed}.npz') as old:
            for name in ('qpos', 'qvel', 'applied', 'strict', 'normalized_residual'):
                np.testing.assert_array_equal(np.asarray(arrays[name]), old[name])
        old_rows = json.loads((Path(parity_directory)/'results.json').read_text())['rows']
        old_row = next(r for r in old_rows if r['case_seed'] == seed)
        assert row['initial_hash'] == old_row['initial_hash']
        row['original_full_path_bitwise_parity'] = True
    dest = Path(directory)
    dest.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(dest/'trajectory.npz', **arrays)
    write_json(dest/'result.json', row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 6:
        raise ValueError('Bounded worker count required')
    args.output.mkdir(exist_ok=False)
    prior_dir = ROOT/'outputs/getup_state_mixture_r107_left_20261005'
    prior = json.loads((prior_dir/'results.json').read_text())
    assert prior['selected_parameters'] == [1., 0., 0., 0., 0.]
    assert not prior['candidate_promoted'] and prior['independent'] is None
    with np.load(prior_dir/'frozen_profiles.npz') as data:
        anchors, gains = data['anchors'].copy(), data['gains'].copy()
    sources = args.output/'executed_sources'
    sources.mkdir()
    for name in (Path(__file__).name, 'test_getup_initial_factors_r108.py',
        'getup_reference_env_r100.py', 'train_getup_joint_anchor_r102.py',
        'getup_independent_native.py', 'validate_getup_fullpath_r27.py',
        'train_getup_fullpath_r27.py', 'probe_getup_anchor_r101.py',
        'getup_fullfallen_env_r32.py', 'getup_fullfallen_contract_r32.py',
        'search_getup_reference_feedback_r64.py', 'search_getup_sustained_bridge_r42.py'):
        shutil.copy2(Path(__file__).with_name(name), sources/name)
    write_json(args.output/'contract.json', dict(simulation_only=True,
        hardware_readiness=False, full_task_completed=False, diagnostic_only=True,
        hypothesis='Initial tilt, encoder and velocity perturbations may have distinct or interacting effects',
        factors=['pre-settling root tilt', 'pre-settling joint positions', 'pre-settling velocities'],
        masks=list(itertools.product((0,1), repeat=3)),
        all_random_draws_preserved=True, settling_controls=40,
        complete_path_controls=2279, physics_hz=500, control_hz=50,
        no_midpath_state_reset=True, physics_rewards_acceptance_unchanged=True,
        no_ablation_counts_as_qualification=True, independent_seeds_unused=True,
        development_seeds=list(TRAIN), seed=208, workers=args.workers, smoke=args.smoke,
        hashes={str(f):digest(f) for f in [SCENE, STAND, REFERENCE, prior_dir/'results.json',
            prior_dir/'frozen_profiles.npz', *sources.iterdir()]}))
    rows = []
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
        initializer=init_worker, initargs=(anchors,gains)) as pool:
        parity_jobs=[]
        for profile, old in [('zero','development_baseline'), ('r102','development_r102')]:
            for seed in (None,769000):
                parity_jobs.append(((1,1,1), profile, seed,
                    str(args.output/'parity'/profile/f'case_{seed}'), str(prior_dir/old)))
        parity=list(pool.map(run,parity_jobs))
        write_json(args.output/'parity.json',parity)
        print('R108_FULL_PATH_PARITY_PASS', flush=True)
        if not args.smoke:
            for mask in itertools.product((0,1), repeat=3):
                for profile in ('zero','r102'):
                    name=''.join(map(str,mask))
                    group=list(pool.map(run,[(mask,profile,seed,
                        str(args.output/'factorial'/name/profile/f'case_{seed}'),
                        str(prior_dir/('development_baseline' if profile=='zero' else 'development_r102'))
                        if mask==(1,1,1) else None) for seed in TRAIN]))
                    rows.extend(group)
                    write_json(args.output/'progress.json',dict(rows=rows, groups_completed=len(rows)//24,
                        successes=sum(r['success'] for r in group),last_mask=name,last_profile=profile,
                        diagnostic_only=True,qualification_evidence=False))
                    print('R108_GROUP',name,profile,sum(r['success'] for r in group),
                          'INVALID',sum(not r['valid'] for r in group),flush=True)
        write_json(args.output/'results.json',dict(rows=rows,parity=parity,smoke=args.smoke,
            diagnostic_only=True,qualification_evidence=False,full_task_completed=False,
            hardware_readiness=False,simulation_only=True))
    print('R108_TERMINAL',flush=True)


if __name__ == '__main__':
    main()
