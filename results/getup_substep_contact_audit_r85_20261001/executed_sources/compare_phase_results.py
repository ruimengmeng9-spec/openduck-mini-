"""Check execution-only changes against a matched native regression result."""
import argparse
import json
from pathlib import Path
import numpy as np


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('before',type=Path)
    p.add_argument('after',type=Path)
    p.add_argument('--tolerance',type=float,default=1e-7)
    args=p.parse_args()
    before={r['seed']:r for r in json.loads(args.before.read_text())['rows']}
    after={r['seed']:r for r in json.loads(args.after.read_text())['rows']}
    if not after or not set(after).issubset(before):
        raise ValueError('matched seeds required')
    fields=['duration_s','speed_mps','lateral_m','yaw_change_deg','heading_rms_deg','minimum_up_z','mean_local_velocity_mps','cost']
    maximum=0.
    for seed,row in after.items():
        if row['fallen']!=before[seed]['fallen']:
            raise ValueError('fall outcome changed')
        for field in fields:
            maximum=max(maximum,float(np.max(abs(np.asarray(row[field])-np.asarray(before[seed][field])))))
    if maximum>args.tolerance:
        raise ValueError(f'closed-loop regression changed: {maximum}')
    print(json.dumps(dict(matched_seeds=sorted(after),maximum_difference=maximum,passed=True)),flush=True)
