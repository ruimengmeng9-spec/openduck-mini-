"""Freeze development evidence and classify physical failures, never certify success."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil


def summarize(rows):
    summary = {}
    for pose in ('standing', 'prone', 'supine', 'left_side', 'right_side'):
        selected = [row for row in rows if row['pose'] == pose]
        causes = Counter()
        invalid = [row for row in selected if not row['valid']]
        for row in invalid:
            peaks = row['peaks']
            # Match AuditedSim: penetration and joint gates are strict '<'.
            for key, bound in (('floor', .01), ('self', .004), ('joint', .08)):
                if peaks[key] >= bound:
                    causes[key] += 1
            for key, bound in (('force', 3.23), ('slew', 5.24)):
                if peaks[key] > bound + 1e-8:
                    causes[key] += 1
            if not row['final'].get('finite', True):
                causes['nonfinite'] += 1
        summary[pose] = dict(episodes=len(selected), physical_failures=len(invalid),
                             discovery_successes=sum(bool(row['training_success']) for row in selected),
                             violation_counts=dict(causes),
                             max_final_up=max((row['final']['up_z'] for row in selected), default=None))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Parse snapshots before publishing, so a partially written live JSON fails.
    history = json.loads((args.training / 'training_history.json').read_text())
    episodes = json.loads((args.training / 'episodes.json').read_text())
    development = json.loads((args.development / 'results.json').read_text())
    if not development['discovery_only'] or development['full_task_completed']:
        raise RuntimeError('Expected development failures, not a final acceptance artifact')
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.development, args.output / 'checkpoint0032_development')
    for suffix in ('onnx', 'msgpack'):
        shutil.copy2(args.training / ('checkpoint_0032.' + suffix), args.output)
    shutil.copy2(args.training / 'controller_contract.json', args.output)
    (args.output / 'history_snapshot.json').write_text(json.dumps(history, indent=2))
    (args.output / 'episodes_snapshot.json').write_text(json.dumps(episodes, indent=2))
    result = dict(last_logged_iteration=history[-1]['iteration'],
                  actual_motor_controls=history[-1]['actual_motor_controls'],
                  episode_snapshot_may_lag_history=True, by_pose=summarize(episodes),
                  deterministic_development=[dict(pose=r['pose'], entry=r['entry_reached'],
                      valid=r['valid'], final_up=r['final']['up_z'], final_height=r['final']['height_m'],
                      final_load=r['final']['foot_load_fraction'],
                      final_joint_home_error=r['final']['joint_home_error_mean_rad'])
                      for r in development['results']],
                  full_task_completed=False, simulation_only=True, hardware_readiness=False,
                  default_controller_replaced=False,
                  diagnosis='Early training violations are mostly floor/self penetration; deterministic development has not reached strict standing. Cause counts may overlap.',
                  next_hypothesis='After final evaluation, test longer temporally coherent exploration against independent per-decision noise without changing geometry or physical gates.')
    result['snapshot_hashes'] = {str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in args.output.rglob('*') if p.is_file()}
    (args.output / 'analysis.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'snapshot_hashes'}), flush=True)


if __name__ == '__main__':
    main()
