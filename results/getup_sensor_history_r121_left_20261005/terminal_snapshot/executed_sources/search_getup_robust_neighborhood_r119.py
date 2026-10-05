"""Bounded common-program teacher search on known three-start neighborhoods.

Case keys select offline simulation jobs only. A proposal is one identical
profile and knot matrix for every start in a neighborhood, not a lookup policy.
Original ReferenceEpisode reward, motor limits, physics and acceptance remain.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import search_getup_case_teachers_r109 as search
from diagnostics import train_getup_program_r113 as program
from diagnostics.search_getup_expanded_teachers_r115 import OUTPUT as TEACHERS
from diagnostics.probe_getup_teacher_neighborhood_r118 import OUTPUT as NEIGHBORS
from diagnostics.train_getup_joint_anchor_r102 import write_json
from diagnostics.getup_independent_native import digest

ROOT = program.ROOT
OUTPUT = ROOT / 'outputs/getup_robust_neighborhood_r119_left_20261005'
SOURCES = (None, 769000, 3160001)
KNOWN = {None, *program.TRAIN, *range(3160000, 3160040)}


def groups_from_pairs(pairs):
    groups = []
    for source in SOURCES:
        targets = [p['target_case'] for p in pairs if p['source_case'] == source]
        cases = [source, *targets]
        if len(cases) != 3 or len(set(cases)) != 3 or not set(cases) <= KNOWN:
            raise ValueError('Exactly three distinct already-known starts required')
        groups.append(cases)
    return groups


def recipe_key(profile, knots):
    k = np.asarray(knots, dtype=np.float64)
    search.knot_action(k, 0)
    if profile not in (0, 1):
        raise ValueError('Unchanged fixed feedback profiles only')
    return str(profile) + k.tobytes().hex()


def rank_group(rows):
    """All-start validity first, then original short labels and original rewards."""
    valid = [bool(r['valid']) for r in rows]
    successful = [bool(r['valid'] and r['training_success']) for r in rows]
    returns = [r['return_sum'] for r in rows]
    return all(valid), sum(successful), sum(valid), min(returns), sum(returns)


def full_group_pass(rows):
    return bool(len(rows) == 3 and all(
        r['full_path'] and r['valid'] and r['controls'] == 2279
        and r['entry_time_s'] is not None and r['entry_time_s'] <= 12.
        and r['strict_tail_s'] >= 30. - 1e-8 for r in rows))


def proposals(state, members, count):
    zero = np.zeros((6, 10))
    items = [(0, zero.copy()), (1, zero.copy()),
             (state['profile'], state['knots'].copy()),
             *[(p, k.copy()) for p, k in members]]
    if count < len(items):
        raise ValueError('Population must retain all fixed comparisons')
    while len(items) < count:
        p = int(state['rng'].integers(2))
        center = state['knots'] if len(items) % 3 == 0 else state['mean']
        k = np.clip(center + state['rng'].normal(0., state['std']), -1., 1.)
        items.append((p, k))
    return items


def init_worker(anchors, gains, rows):
    search.init_worker(anchors, gains)
    search.BASE_ROWS = {r['case_seed']: r for r in rows}


def rollout(job):
    case, profile, knots, full, directory = job
    if case not in KNOWN:
        raise ValueError('Unseen qualification states must not be loaded')
    row = search.rollout((case, profile, knots, full, directory, None))
    row.update(neighborhood_teacher_search=True, independent_qualification_trial=False,
               case_key_offline_only=True, deployed_lookup_policy=False)
    if directory:
        write_json(Path(directory) / 'result.json', row)
    return row


def parity(directory, previous):
    now_row = json.loads((directory / 'result.json').read_text())
    old_row = json.loads((previous / 'result.json').read_text())
    assert now_row['initial_hash'] == old_row['initial_hash']
    with np.load(directory / 'trajectory.npz') as now, np.load(previous / 'trajectory.npz') as old:
        for key in ('observations', 'qpos', 'qvel', 'applied', 'strict', 'normalized_residual'):
            np.testing.assert_array_equal(now[key], old[key])


def run_group(pool, cases, profile, knots, full, directory):
    # The same numeric program is sent to all starts. No case-dependent action.
    return list(pool.map(rollout, [(s, profile, knots, full,
        str(directory / f'case_{s}')) for s in cases]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--generations', type=int, default=6)
    parser.add_argument('--population', type=int, default=14)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not (1 <= args.workers <= 6 and 1 <= args.generations <= 6 and 6 <= args.population <= 14):
        raise ValueError('Fresh bounded CPU experiment required')
    if args.output.parent.resolve() != (ROOT / 'outputs').resolve():
        raise ValueError('Output must be a fresh direct simulation output directory')
    groups = groups_from_pairs(json.loads((NEIGHBORS / 'pairs.json').read_text()))
    if args.smoke:
        groups = groups[:1]
    cases = list(dict.fromkeys(s for group in groups for s in group))
    rows = [json.loads((TEACHERS / f'teachers/case_{s}/result.json').read_text()) for s in cases]
    recipes = {}
    for s in cases:
        with np.load(TEACHERS / f'teachers/case_{s}/teacher_parameters.npz') as data:
            recipes[s] = int(data['profile']), data['knots'].copy()
    with np.load(TEACHERS / 'frozen_profiles.npz') as data:
        anchors, gains = data['anchors'].copy(), data['gains'].copy()
    args.output.mkdir(exist_ok=False)
    np.savez_compressed(args.output / 'frozen_profiles.npz', anchors=anchors, gains=gains)
    sources = args.output / 'executed_sources'
    sources.mkdir()
    for name in (Path(__file__).name, 'test_getup_robust_neighborhood_r119.py',
        'launch_getup_robust_neighborhood_r119.py', 'search_getup_case_teachers_r109.py',
        'search_getup_expanded_teachers_r115.py', 'probe_getup_teacher_neighborhood_r118.py',
        'train_getup_expanded_program_r116.py', 'train_getup_program_calibration_r114.py',
        'train_getup_program_r113.py', 'train_getup_joint_anchor_r102.py',
        'getup_reference_env_r100.py', 'getup_independent_native.py', 'validate_getup_fullpath_r27.py',
        'train_getup_fullpath_r27.py', 'probe_getup_anchor_r101.py', 'getup_fullfallen_env_r32.py',
        'getup_fullfallen_contract_r32.py', 'search_getup_reference_feedback_r64.py',
        'search_getup_sustained_bridge_r42.py'):
        shutil.copy2(Path(__file__).with_name(name), sources / name)
    write_json(args.output / 'contract.json', dict(
        hypothesis='Joint multi-start optimization may find wider verified teacher stability basins than per-case exact fitting',
        groups=groups, selection_uses_completed_R118_development_pairs=True,
        fixed_seed=219, rng_per_group='219 + group ordinal', workers=args.workers,
        generations=args.generations, population=args.population, smoke=args.smoke,
        full_checks_per_group_per_generation=2, teacher_data_only=True,
        unified_policy_success=False, independent_qualification_run=False,
        unseen_3200000_to_3200039_never_loaded=True, simulation_only=True,
        hardware_readiness=False, full_task_completed=False,
        common_program_identical_across_group=True, no_deployed_nearest_neighbor_policy=True,
        training_rank='all valid, valid short successes, valid count, minimum original return, sum original return',
        short_training_tail_s=1, full_path_controls=2279, strict_tail_s=30, entry_deadline_s=12,
        normalized_bounds=[-1, 1], combined_correction_limit_rad=.18,
        physics_hz=500, control_hz=50, rewards_physics_acceptance_unchanged=True,
        no_mid_episode_root_reset=True, all_short_and_full_trajectories_saved=True,
        hashes={str(f): digest(f) for f in [program.SCENE, program.STAND, program.REFERENCE,
            program.MODEL, NEIGHBORS / 'results.json', TEACHERS / 'results.json', *sources.iterdir()]}))
    history, baseline, states = [], [], {}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
        initializer=init_worker, initargs=(anchors, gains, rows)) as pool:
        for g, group in enumerate(groups):
            members = [recipes[s] for s in group]
            fixed = [*members, (0, np.zeros((6, 10))), (1, np.zeros((6, 10)))]
            comparisons = []
            for i, (p, k) in enumerate(fixed):
                dest = args.output / 'fixed_comparisons' / f'group_{g}' / f'program_{i}'
                reports = run_group(pool, group, p, k, True, dest)
                if i < 3:
                    parity(dest / f'case_{group[i]}', TEACHERS / f'teachers/case_{group[i]}')
                if i == 0:
                    for s in group:
                        old = NEIGHBORS / ('own_replay' if s == group[0] else 'neighbors') / f'source_{group[0]}/case_{s}'
                        parity(dest / f'case_{s}', old)
                comparisons.append(dict(profile=p, knots=k.tolist(), rows=reports,
                    full_group_pass=full_group_pass(reports)))
            baseline.append(dict(group=g, cases=group, comparisons=comparisons,
                member_own_parity=True, R118_source_transfer_parity=True))
            p, k = members[0]
            qualified = next((r for r in comparisons if r['full_group_pass']), None)
            if qualified is not None:
                p, k = qualified['profile'], np.array(qualified['knots'])
            states[g] = dict(profile=p, knots=k.copy(), mean=k.copy(), std=np.full((6, 10), .10),
                rng=np.random.default_rng(219 + g), best=None, full_checks={}, qualified=qualified)
            write_json(args.output / 'fixed_comparisons.json', baseline)
            print('R119_BASELINE_PARITY', g, 'COMMON_PROGRAMS', sum(r['full_group_pass'] for r in comparisons), flush=True)
        if args.smoke:
            p, k = recipes[groups[0][0]]
            short = run_group(pool, groups[0], p, k, False, args.output / 'smoke_short')
            write_json(args.output / 'results.json', dict(smoke=True, baseline=baseline,
                short_rows=short, parity_passed=True, unified_policy_success=False,
                independent_qualification_run=False, full_task_completed=False))
        else:
            for gen in range(1, args.generations + 1):
                active = [g for g, st in states.items() if st['qualified'] is None]
                if not active:
                    break
                generation = []
                for g in active:
                    st, group = states[g], groups[g]
                    items = proposals(st, [recipes[s] for s in group], args.population)
                    jobs = [(s, p, k, False, str(args.output / 'short_trials' /
                        f'generation_{gen:04d}' / f'group_{g}' / f'candidate_{i:03d}' / f'case_{s}'))
                        for i, (p, k) in enumerate(items) for s in group]
                    flat = list(pool.map(rollout, jobs))
                    reports = [flat[i * 3:(i + 1) * 3] for i in range(len(items))]
                    order = sorted(range(len(items)), key=lambda i: rank_group(reports[i]), reverse=True)
                    i = order[0]
                    if st['best'] is None or rank_group(reports[i]) > rank_group(st['best']):
                        st.update(best=reports[i], profile=items[i][0], knots=items[i][1].copy())
                    elite = np.stack([items[i][1] for i in order[:3]])
                    st['mean'] = .6 * st['mean'] + .4 * elite.mean(0)
                    st['std'] = np.clip(.6 * st['std'] + .4 * elite.std(0), .025, .35)
                    checked = 0
                    for i in order:
                        if not all(r['valid'] and r['training_success'] for r in reports[i]):
                            continue
                        p, k = items[i]
                        key = recipe_key(p, k)
                        if key in st['full_checks']:
                            continue
                        dest = args.output / 'full_checks' / f'group_{g}' / f'generation_{gen:04d}_candidate_{i:03d}'
                        full = run_group(pool, group, p, k, True, dest)
                        verified = dict(profile=p, knots=k.tolist(), rows=full, full_group_pass=full_group_pass(full))
                        st['full_checks'][key] = verified
                        checked += 1
                        if verified['full_group_pass']:
                            st.update(qualified=verified, profile=p, knots=k.copy())
                        print('R119_FULL_CHECK', gen, g, i, [r['success'] for r in full], flush=True)
                        if st['qualified'] is not None or checked >= 2:
                            break
                    checkpoint = args.output / 'checkpoints' / f'group_{g}'
                    checkpoint.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(checkpoint / f'generation_{gen:04d}.npz',
                        profile=st['profile'], knots=st['knots'], mean=st['mean'], std=st['std'])
                    generation.append(dict(group=g, cases=group,
                        proposals=[dict(profile=p, knots=k.tolist()) for p, k in items], reports=reports,
                        best=st['best'], full_checks=list(st['full_checks'].values()), qualified=st['qualified'],
                        rng_state=st['rng'].bit_generator.state))
                    print('R119_GROUP', gen, g, 'SHORT_SUCCESSES', rank_group(st['best'])[1], 'FULL', st['qualified'] is not None, flush=True)
                history.append(dict(generation=gen, groups=generation))
                write_json(args.output / 'history.json', history)
                write_json(args.output / 'progress.json', dict(completed_generations=gen,
                    qualified_groups=sum(st['qualified'] is not None for st in states.values()),
                    group_count=len(groups), teacher_data_only=True, unified_policy_success=False))
            # Full replay of each final common candidate, including failed groups.
            final = []
            for g, st in states.items():
                final_rows = run_group(pool, groups[g], st['profile'], st['knots'], True,
                    args.output / 'final_common_programs' / f'group_{g}')
                final.append(dict(group=g, cases=groups[g], profile=st['profile'], knots=st['knots'].tolist(),
                    rows=final_rows, full_group_pass=full_group_pass(final_rows)))
            write_json(args.output / 'results.json', dict(groups=final, baseline=baseline,
                completed_generations=len(history), qualified_groups=sum(r['full_group_pass'] for r in final),
                teacher_data_only=True, unified_policy_success=False, independent_qualification_run=False,
                hardware_readiness=False, simulation_only=True, full_task_completed=False))
    print('R119_TERMINAL', flush=True)


if __name__ == '__main__':
    main()
