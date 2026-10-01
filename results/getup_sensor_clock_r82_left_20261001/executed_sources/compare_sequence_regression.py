"""Compare complete saved states, not just survival, after step-code extraction."""
import argparse
import json
from pathlib import Path
import numpy as np


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('before',type=Path)
    p.add_argument('after',type=Path)
    a=p.parse_args()
    first=np.load(a.before,allow_pickle=False)
    second=np.load(a.after,allow_pickle=False)
    difference=max(float(np.max(abs(first[key]-second[key]))) for key in ('time','qpos'))
    if difference>1e-7:
        raise ValueError(f'sequence controller execution changed: {difference}')
    print(json.dumps(dict(maximum_state_difference=difference,passed=True)),flush=True)
