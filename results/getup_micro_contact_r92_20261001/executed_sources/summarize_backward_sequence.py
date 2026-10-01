"""Report all sequence phases; survival alone is not a successful skill."""
import argparse
import json
from pathlib import Path


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('result',type=Path)
    a=p.parse_args()
    data=json.loads(a.result.read_text())
    rows=data['rows']
    result=dict(tested=len(rows),completed=sum(r['completed'] for r in rows),qualified=sum(r['qualified'] for r in rows),failed_seeds=[r['seed'] for r in rows if not r['completed']],unqualified_seeds=[r['seed'] for r in rows if not r['qualified']],phases={})
    for phase,_ in data['schedule']:
        items=[p for row in rows for p in row['phases'] if p['phase']==phase]
        result['phases'][phase]=dict(tested=len(items),qualified=sum(p['qualified'] for p in items),speed_mps=[min(p['speed_mps'] for p in items),max(p['speed_mps'] for p in items)],maximum_absolute_yaw_deg=max(abs(p['yaw_change_deg']) for p in items),maximum_tail_1s_drift_m=max(p['last_1s_horizontal_drift_m'] for p in items),minimum_up_z=min(p['minimum_up_z'] for p in items))
    print(json.dumps(result,indent=2),flush=True)
