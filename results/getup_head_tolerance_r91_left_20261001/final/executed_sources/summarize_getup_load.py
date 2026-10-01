"""Compact, evidence-based summary of the bounded R7-R9 experiments."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--training',type=Path,default=Path('/data/shijinsheng/open_duck/training'))
    args=p.parse_args()
    rows=[]
    for case in ('getup_anatomical_r7_prone','getup_load_r8_supine','getup_load_r8_prone',
                 'getup_ik_r9_prone_verified','getup_ik_r9_supine'):
        r=json.loads((args.training/case/'results.json').read_text())
        row=dict(case=case,successes=r['successful_validation_runs'],runs=r['validation_runs'])
        progress=args.training/case/'search_progress.json'
        if progress.exists():
            states=json.loads(progress.read_text())
            m=states[-1]['metric']
            row.update(search_height_m=m['height_m'],search_up_z=m['up_z'],
                       foot_load_fraction=m.get('foot_load_fraction'),
                       foot_up_alignment=m['foot_up_alignment_to_home'],
                       knees_rad=[m['joint_positions_rad'][3],m['joint_positions_rad'][12]],
                       ready_nodes=sum(s['ready_nodes'] for s in states))
        else:
            row.update(valid_runs=sum(s['valid'] for s in r['results']),
                       entry_gate_reached_runs=sum(s['entry_gate_reached'] for s in r['results']))
        rows.append(row)
    (args.training/'getup_load_summary_r7_r9.json').write_text(json.dumps(dict(rows=rows),indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    main()
