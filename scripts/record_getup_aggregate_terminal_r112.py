"""Append a verified terminal note, leaving all earlier snapshots unchanged."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
ROOT=Path('/data/shijinsheng/open_duck');REPO=ROOT/'github/openduck-mini-'


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    result=json.loads((ROOT/'outputs/getup_aggregate_r112_left_20261005/results.json').read_text())
    assert [r['successes'] for r in result['candidates']]==[7,7,3]
    assert [r['physical_failures'] for r in result['candidates']]==[1,1,3]
    assert all(r['nominal_success'] for r in result['candidates']) and not result['independent_qualification_run']
    snapshot=REPO/'results/getup_aggregate_r112_left_20261005/training_closed_04000/snapshot.json'
    assert json.loads(snapshot.read_text())['terminal_result_saved']
    doc='GETUP_AGGREGATION_TERMINAL_R112_20261005.md';assert not (REPO/doc).exists()
    shutil.copy2(Path(__file__).with_name(doc),REPO/doc);shutil.copy2(Path(__file__),REPO/'scripts'/Path(__file__).name)
    subprocess.run(['git','add',doc,'scripts/'+Path(__file__).name],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Record actual terminal R112 failure and next representation hypothesis'],cwd=REPO,check=True)
    bundle=ROOT/'tmp/getup_aggregate_terminal_note_r112_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=REPO,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())),flush=True)


if __name__=='__main__':main()
