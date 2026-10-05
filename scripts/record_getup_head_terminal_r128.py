import argparse
from pathlib import Path
import shutil
import subprocess
import json
from diagnostics.probe_getup_head_feedback_r128 import ROOT,OUTPUT


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--bundle-base',required=True);args=p.parse_args()
    repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.parent
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    result=json.loads((OUTPUT/'results.json').read_text());assert len(result['rows'])==15 and not result['smoke']
    for gain in (0.,1.,4.):
        rows=[r for r in result['rows'] if r['gain']==gain]
        assert len(rows)==5 and sum(r['success'] for r in rows if r['case_seed'] is not None)==0
        assert sum(not r['valid'] for r in rows)==1
    doc='GETUP_HEAD_FEEDBACK_TERMINAL_R128_20261005.md';assert not (repo/doc).exists()
    shutil.copy2(Path(__file__).with_name(doc),repo/doc)
    script=repo/'scripts'/Path(__file__).name;assert not script.exists();shutil.copy2(__file__,script)
    subprocess.run(['git','add',doc,str(script.relative_to(repo))],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Record R128 terminal: no strict recovery improvement, preserve paired physical rescue and regression'],cwd=repo,check=True)
    bundle=ROOT/'tmp/getup_head_feedback_r128_terminal_note_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.bundle_base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle))),flush=True)


if __name__=='__main__':main()
