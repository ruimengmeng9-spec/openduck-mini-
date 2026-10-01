"""Render a saved dynamic-search candidate, keeping failures explicitly labelled."""
import argparse
import json
from pathlib import Path

from diagnostics.getup_feedback_reference import render


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    args=p.parse_args()
    result=json.loads((args.experiment/'results.json').read_text())
    render(Path(result['scene_path']),args.experiment/'best_trajectory.npz',args.experiment)
    print('Candidate validation:',result['successful_validation_runs'],'/',result['validation_runs'])
    print('Simulation-only candidate; not ready for hardware')


if __name__=='__main__':
    main()
