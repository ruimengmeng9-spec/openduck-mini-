"""Render only a saved, revalidated recovery trajectory (success or failure)."""
import argparse
import json
from pathlib import Path

from diagnostics.getup_feedback_reference import render


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--experiment', type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.experiment/'verified_results.json').read_text())
    render(args.experiment/'model/scene.xml', args.experiment/'verified_trajectory.npz', args.experiment)
    print('Validated candidate successes:', summary['successful_validation_runs'], '/', summary['validation_runs'])
    print('Preview:', args.experiment/'recovery_search_preview.mp4')


if __name__ == '__main__':
    main()
