"""Freeze completed research artifacts and diagnose paired regressions; no control."""
import argparse
import json
from pathlib import Path
from diagnostics.getup_independent_native import digest
from diagnostics.train_getup_sequence_r24 import select_radius


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    args = p.parse_args()
    root = args.experiment
    report = json.loads((root / 'results.json').read_text())
    assert report['complete'] and report['selected_radius_frozen_before_final_tests']
    assert report['selected_radius'] == select_radius(report['selection_60']['summary'])
    assert digest(root / 'trajectory_library.npz') == report['library_sha256']
    regressions = []
    for name in ('independent_4s', 'independent_30s', 'mild_4s'):
        block = report[name]
        new = {r['seed']: r for r in block['r26']['results'] if r['radius'] == report['selected_radius']}
        old = {r['seed']: r for r in block['r25']['results'] if r['radius'] == report['previous_audit']['selected_radius']}
        for seed in new:
            assert new[seed]['initial_hash'] == old[seed]['initial_hash']
            if old[seed]['success'] and not new[seed]['success']:
                regressions.append(dict(block=name, seed=seed, initial_hash=new[seed]['initial_hash'],
                    r25_choice=old[seed]['choice'], r26_choice=new[seed]['choice'],
                    radius_rejected_existing_rescue=bool(old[seed]['choice']['mode'] == 'rescue' and
                        new[seed]['choice']['mode'] == 'home' and
                        new[seed]['choice']['rescue_distance'] > report['selected_radius'])))
    c = json.loads((root / 'controller_contract.json').read_text())
    manifest = dict(type='initial_state_whole_trajectory_atlas', library_file='trajectory_library.npz',
                    library_sha256=report['library_sha256'], radius=report['selected_radius'],
                    feature_indices=c['feature_indices'], feature_scales=c['feature_scales'],
                    rescue_vs_home_distance_margin=.8, initial_observation_size=50,
                    first_knot_hold_s=.2, second_knot_time_s=.6, home_tail_after_s=1.2,
                    motor_target_period_s=.02, root_state_modification_allowed=False,
                    simulation_only=True, hardware_readiness=False, default_controller_replaced=False,
                    adoption_gate_passed=report['adoption_gate_passed'],
                    qualification='research artifact only; NOT full fallen get-up or hardware policy',
                    results_sha256=digest(root / 'results.json'))
    for name, value in [('frozen_policy.json', manifest), ('regression_analysis.json', regressions)]:
        with (root / name).open('x', encoding='utf-8') as handle:
            json.dump(value, handle, indent=2)
    print(json.dumps(dict(adoption_gate_passed=report['adoption_gate_passed'], regressions=regressions), indent=2), flush=True)


if __name__ == '__main__':
    main()
