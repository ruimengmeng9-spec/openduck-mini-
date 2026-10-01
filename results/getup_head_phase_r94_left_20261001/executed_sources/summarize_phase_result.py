"""Conservative, simulation-only acceptance summary for native phase controllers."""
import argparse
import json
from pathlib import Path


def summarize(path):
    data=json.loads(Path(path).read_text())
    rows=data['rows']
    if not rows:
        raise ValueError('empty validation')
    # A survival count is not a straight-backward gait certificate.
    qualified=[r for r in rows if not r['fallen'] and -.10 <= r['speed_mps'] <= -.05
               and abs(r['yaw_change_deg']) <= 15 and r['heading_rms_deg'] <= 10
               and abs(r['lateral_m']) <= .25 and r['minimum_up_z'] >= .94]
    return dict(file=str(path),simulation_only=True,tested=len(rows),survived=sum(not r['fallen'] for r in rows),
                qualified=len(qualified),failed_seeds=[r['seed'] for r in rows if r['fallen']],
                unqualified_seeds=[r['seed'] for r in rows if r not in qualified],
                duration_s=[min(r['duration_s'] for r in rows),max(r['duration_s'] for r in rows)],
                speed_mps=[min(r['speed_mps'] for r in rows),max(r['speed_mps'] for r in rows)],
                maximum_absolute_heading_deg=max(abs(r['yaw_change_deg']) for r in rows),
                maximum_heading_rms_deg=max(r['heading_rms_deg'] for r in rows),
                maximum_absolute_lateral_m=max(abs(r['lateral_m']) for r in rows),
                minimum_up_z=min(r['minimum_up_z'] for r in rows),
                thresholds='speed [-.10,-.05] m/s; |yaw|<=15 deg; RMS<=10 deg; |lateral|<=.25 m; min up>=.94; no fall',
                hardware_readiness=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('files',nargs='+',type=Path)
    args=p.parse_args()
    for path in args.files:
        print(json.dumps(summarize(path),ensure_ascii=False),flush=True)
