"""Verify retained alternating support, without claiming zero foot slip."""
import argparse
import json
from pathlib import Path
import numpy as np


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    rows=[]
    result=json.loads((a.folder/'results.json').read_text())
    for row in result['rows']:
        t=np.load(a.folder/f"seed_{row['seed']}.npz",allow_pickle=False)
        contacts=t['feet_contacts'][t['time']>=3].astype(bool)
        code=contacts[:,0].astype(int)+2*contacts[:,1].astype(int)
        counts=np.sum(contacts[:-1]&~contacts[1:],axis=0)
        boundaries=np.r_[0,np.flatnonzero(code[1:]!=code[:-1])+1,len(code)]
        sustained=[int(sum(code[start]==value and end-start>=2 for start,end in zip(boundaries[:-1],boundaries[1:]))) for value in (1,2)]
        rows.append(dict(seed=row['seed'],contact_proportions={name:float(np.mean(code==value)) for value,name in enumerate(('air','left_only','right_only','both'))},contact_loss_counts=counts.tolist(),single_support_segments_at_least_40ms=sustained))
    report=dict(rows=rows,all_have_alternating_support=all(min(r['single_support_segments_at_least_40ms'])>1 for r in rows),contact_loss_is_not_physical_step_count=True,contact_audit_is_not_a_slip_certificate=True)
    a.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(tested=len(rows),all_have_alternating_support=report['all_have_alternating_support'],minimum_sustained_left_support_segments=min(r['single_support_segments_at_least_40ms'][0] for r in rows),minimum_sustained_right_support_segments=min(r['single_support_segments_at_least_40ms'][1] for r in rows))),flush=True)
