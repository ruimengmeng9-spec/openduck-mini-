"""Count actual native evaluation control steps; these are not PPO updates."""
import argparse
import json
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('files',nargs='+',type=Path)
    args=p.parse_args()
    for path in args.files:
        data=json.loads(path.read_text())
        evaluations=data['evaluations']
        rows=[row for candidate in evaluations for row in candidate['rows']]
        print(json.dumps(dict(file=str(path),method='native MuJoCo CEM; not PPO',
                              candidates=len(evaluations),rollouts=len(rows),
                              control_steps=sum(round(row['duration_s']/.02) for row in rows),
                              generations=1+max(c['generation'] for c in evaluations),
                              best_score=min(c['score'] for c in evaluations),
                              elapsed_seconds=data['elapsed_seconds'])),flush=True)
