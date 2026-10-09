"""Read existing closed reports only, no simulation or controller decisions."""
import json
from diagnostics import probe_getup_phase_coordinate_r185b as run


def main():
    for root in (run.SMOKE,run.OUTPUT):
        if not root.exists():continue
        rows=[]
        for path in sorted(root.rglob('result.json')):
            row=json.loads(path.read_text())
            if 'case_seed' in row:
                rows.append(dict(group=path.parent.parent.name,case=row['case_seed'],success=row['success'],valid=row['valid'],controls=row['controls'],entry=row['entry_time_s'],tail=row['strict_tail_s'],maximum_shift=row['maximum_phase_reference_shift_rad'],maximum_offset=row['maximum_phase_offset_controls'],peaks=row['peaks']))
        print(json.dumps(dict(root=str(root),terminal=(root/'results.json').exists(),training_closed=(root/'training_closed.json').exists(),closed_cases=rows)),flush=True)
    if (run.OUTPUT/'results.json').exists():
        r=json.loads((run.OUTPUT/'results.json').read_text())
        b={x['case_seed']:x for x in r['baseline']['rows']};c={x['case_seed']:x for x in r['candidate']['rows']}
        print(json.dumps(dict(candidate_successes=r['candidate']['successes'],baseline_successes=r['baseline']['successes'],invalid=r['candidate']['physical_failures'],gate=r['original_development_gate'],rescued=[k for k in c if c[k]['success'] and not b[k]['success']],regressed=[k for k in c if b[k]['success'] and not c[k]['success']])),flush=True)


if __name__=='__main__':main()
