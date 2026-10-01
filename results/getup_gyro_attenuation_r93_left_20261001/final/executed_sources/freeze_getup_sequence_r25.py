"""Save a self-contained simulation-only atlas recipe after finished audits."""
import argparse
import json
from pathlib import Path
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--validation',type=Path,required=True);args=p.parse_args()
    trained=json.loads((args.experiment/'results.json').read_text())
    checked=json.loads((args.validation/'results.json').read_text())
    c=json.loads((args.experiment/'controller_contract.json').read_text())
    if not trained.get('complete') or not checked.get('complete'):raise RuntimeError('unfinished evaluation')
    sha=digest(args.experiment/'trajectory_library.npz')
    if sha!=trained['library_sha256'] or sha!=checked['audit']['library_sha256']:
        raise RuntimeError('library mismatch')
    output=args.experiment/'frozen_policy.json'
    if output.exists():raise FileExistsError(output)
    recipe=dict(type='initial_state_whole_trajectory_atlas',library_file='trajectory_library.npz',
        library_sha256=sha,radius=trained['selected_radius'],
        feature_indices=c['feature_indices'],feature_scales=c['feature_scales'],
        rescue_vs_home_distance_margin=c['rescue_vs_home_distance_margin'],
        initial_observation_size=50,first_knot_hold_s=.2,second_knot_time_s=.6,
        home_tail_after_s=1.2,motor_target_period_s=.02,root_state_modification_allowed=False,
        simulation_only=True,hardware_readiness=False,default_controller_replaced=False,
        qualification='research candidate for near-standing starts; NOT full fallen get-up',
        source_sha256=digest(Path(__file__).with_name('train_getup_sequence_r24.py')),
        scene_sha256=digest(c['scene_path']))
    output.write_text(json.dumps(recipe,indent=2),encoding='utf-8')
    print('FROZEN RESEARCH RECIPE:',sha,'radius',recipe['radius'])


if __name__=='__main__':main()
