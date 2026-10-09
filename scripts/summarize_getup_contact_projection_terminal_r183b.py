"""Closed result and contact eligibility counts only; no dynamic reconstruction."""
import json
import numpy as np
from diagnostics import probe_getup_contact_projection_r183b as run

result=json.loads((run.OUTPUT/'results.json').read_text())
rows=[]
for case,b,c in zip(run.CASES,result['baseline']['rows'],result['candidate']['rows']):
    with np.load(run.OUTPUT/'development_candidate'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as z:
        double=np.all(z['observations'][:529,48:50]>0,axis=1)
        extra=np.max(np.abs(z['projection_extra_rad'][:529]),axis=1)
        changed=np.flatnonzero(extra>0)
        rows.append(dict(case=case,baseline=b['success'],candidate=c['success'],valid=c['valid'],raw_double_contacts=int(double.sum()),active=int(np.count_nonzero(z['contact_projection_active'][:529])),maxextra=float(extra.max()),firstextra=int(changed[0]) if len(changed) else None))
print(json.dumps(dict(candidate_successes=result['candidate']['successes'],baseline_successes=result['baseline']['successes'],invalid=result['candidate']['physical_failures'],nominal=result['candidate']['nominal_success'],gate=result['original_development_gate'],rescued=[r['case'] for r in rows if r['candidate'] and not r['baseline']],regressed=[r['case'] for r in rows if r['baseline'] and not r['candidate']],rows=rows),indent=2))
