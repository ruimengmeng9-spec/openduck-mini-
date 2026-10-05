"""Frozen R119 common program on 65 known starts, no case-based inference."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import numpy as np
from diagnostics import search_getup_case_teachers_r109 as search
from diagnostics import train_getup_program_r113 as program
from diagnostics.search_getup_expanded_teachers_r115 import OUTPUT as TEACHERS
from diagnostics.search_getup_robust_neighborhood_r119 import OUTPUT as R119, parity
from diagnostics.train_getup_joint_anchor_r102 import write_json, aggregate
from diagnostics.getup_independent_native import digest

ROOT = program.ROOT
OUTPUT = ROOT / 'outputs/getup_common_program_r120_left_20261005'
PROFILE = KNOTS = None


def frozen_program(result):
    selected = [g for g in result['groups'] if g['group'] == 2]
    if len(selected) != 1 or not selected[0]['full_group_pass']:
        raise ValueError('A fully verified common program is required')
    p, k = selected[0]['profile'], np.asarray(selected[0]['knots'], dtype=np.float64)
    if p not in (0, 1):
        raise ValueError('Existing feedback profiles only')
    search.knot_action(k, 0)
    return p, k


def init_worker(anchors, gains, initial_rows, profile, knots):
    global PROFILE, KNOTS
    search.init_worker(anchors, gains)
    search.BASE_ROWS = {r['case_seed']: r for r in initial_rows}
    PROFILE, KNOTS = profile, knots.copy()


def rollout(job):
    case, directory, previous = job
    # Offline seed affects initial simulation state, never chooses a recipe.
    row = search.rollout((case, PROFILE, KNOTS, True, directory, None))
    if previous:
        parity(Path(directory), Path(previous))
        row['R119_full_path_bitwise_parity'] = True
    row.update(fixed_common_program=True, independent_qualification_trial=False,
        case_identifiers_in_policy=False, no_nearest_neighbor_lookup=True)
    write_json(Path(directory) / 'result.json', row)
    return row


def main():
    profile, knots = frozen_program(json.loads((R119 / 'results.json').read_text()))
    cases = [None, *program.TRAIN, *range(3160000, 3160040)]
    initial_rows = [json.loads((TEACHERS / f'teachers/case_{s}/result.json').read_text()) for s in cases]
    # Saved fixed baselines use precisely the same original complete starts.
    old_zero = json.loads((ROOT / 'outputs/getup_program_r113_left_20261005/results.json').read_text())['baseline_reused_from_R110']['rows']
    new_zero = json.loads((ROOT / 'outputs/getup_program_calibration_r114_left_20261005/results.json').read_text())['independent']['baseline']['rows']
    old_r102 = json.loads((ROOT / 'outputs/getup_state_mixture_r107_left_20261005/results.json').read_text())['previous_profile']['rows']
    new_r102 = json.loads((TEACHERS / 'expanded_r102/results.json').read_text())['rows']
    baseline = {label: {r['case_seed']: r for r in rows} for label, rows in
        [('zero', [*old_zero, *new_zero]), ('r102', [*old_r102, *new_r102])]}
    for rows in baseline.values():
        assert set(rows) == set(cases)
        for r in initial_rows:
            assert rows[r['case_seed']]['initial_hash'] == r['initial_hash']
    with np.load(TEACHERS / 'frozen_profiles.npz') as data:
        anchors, gains = data['anchors'].copy(), data['gains'].copy()
    OUTPUT.mkdir(exist_ok=False)
    np.savez_compressed(OUTPUT / 'frozen_common_program.npz', profile=profile, knots=knots,
        anchors=anchors, gains=gains)
    shutil.copytree(R119 / 'executed_sources', OUTPUT / 'executed_sources')
    for name in (Path(__file__).name, 'test_getup_common_program_r120.py', 'launch_getup_common_program_r120.py'):
        shutil.copy2(Path(__file__).with_name(name), OUTPUT / 'executed_sources' / name)
    write_json(OUTPUT / 'contract.json', dict(simulation_only=True,
        hypothesis='A recipe jointly optimized for three actual fallen starts may have wider known-state coverage than individually fitted teachers',
        program_frozen_before_expanded_rollouts=True, source_group=2, workers=6,
        known_development_cases=cases, no_parameter_training_in_this_run=True,
        actual_complete_fall_only=True, no_case_dependent_recipe=True,
        policy_never_loads_teacher_library=True, unseen_3200000_to_3200039_never_loaded=True,
        full_path_controls=2279, control_hz=50, physics_hz=500, strict_tail_s=30, entry_deadline_s=12,
        correction_limit_rad=.18, physics_rewards_acceptance_unchanged=True,
        no_mid_episode_root_reset=True, independent_qualification_run=False, hardware_readiness=False,
        full_task_completed=False, hashes={str(f): digest(f) for f in
        [program.SCENE, program.STAND, program.REFERENCE, program.MODEL, R119 / 'results.json',
            OUTPUT / 'frozen_common_program.npz', *(OUTPUT / 'executed_sources').iterdir()]}))
    with ProcessPoolExecutor(6, mp_context=mp.get_context('spawn'), initializer=init_worker,
        initargs=(anchors, gains, initial_rows, profile, knots)) as pool:
        source_cases = (3160001, 773014, 773008)
        parity_rows = list(pool.map(rollout, [(s, str(OUTPUT / 'parity' / f'case_{s}'),
            str(R119 / 'final_common_programs/group_2' / f'case_{s}')) for s in source_cases]))
        write_json(OUTPUT / 'parity.json', dict(rows=parity_rows, bitwise_equal=True))
        print('R120_FROZEN_PARITY_PASS', flush=True)
        rows = list(pool.map(rollout, [(s, str(OUTPUT / 'development' / f'case_{s}'), None) for s in cases]))
        groups = {}
        for label, subset in [('original', [None, *program.TRAIN]), ('expanded', list(range(3160000, 3160040)))]:
            candidate = [r for r in rows if r['case_seed'] in subset]
            groups[label] = dict(candidate=aggregate(candidate), **{name: aggregate([fixed[s] for s in subset])
                for name, fixed in baseline.items()})
        original, expanded = groups['original'], groups['expanded']
        development_gate = bool(original['candidate']['nominal_success']
            and original['candidate']['successes'] >= 22 and expanded['candidate']['successes'] >= 36
            and original['candidate']['physical_failures'] == expanded['candidate']['physical_failures'] == 0
            and all(groups[g]['candidate']['successes'] > groups[g][b]['successes']
                for g in ('original', 'expanded') for b in ('zero', 'r102')))
        write_json(OUTPUT / 'results.json', dict(groups=groups, development_gate=development_gate,
            paired_initial_hashes_verified=True, frozen_program_before_development=True,
            independent_qualification_run=False, left_stage_passed=False,
            hardware_readiness=False, full_task_completed=False))
        print('R120_DEVELOPMENT', original['candidate']['successes'], expanded['candidate']['successes'],
            'GATE', development_gate, flush=True)
    print('R120_TERMINAL', flush=True)


if __name__ == '__main__':
    main()
