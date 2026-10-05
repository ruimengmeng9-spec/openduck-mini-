import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.probe_getup_common_prefix_r136 import ROOT,OUTPUT
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args();repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    result=json.loads((OUTPUT/'results.json').read_text());assert result['complete_union_count']==21
    assert [r['successes'] for r in result['reports']['10']]==[15,6,6,5]
    assert [r['physical_failures'] for r in result['reports']['10']]==[0,0,1,4]
    target=repo/'results'/OUTPUT.name/'terminal_snapshot';assert not target.exists();shutil.copytree(OUTPUT,target)
    shutil.copy2(OUTPUT.with_suffix('.log'),target/OUTPUT.with_suffix('.log').name)
    tests=subprocess.run([str(ROOT/'projects/Open_Duck_Playground/.venv/bin/python'),'-m','unittest','diagnostics.test_getup_common_prefix_r136','-v'],cwd=ROOT/'projects/Open_Duck_Playground',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert tests.returncode==0;(target/'archive_regression_tests.log').write_text(tests.stdout)
    (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=True,closed_trials=len(list(target.rglob('result.json'))),
        reject_this_common_prefix_library=True,unified_candidate_promoted=False,hardware_readiness=False,full_task_completed=False),indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2))
    doc='GETUP_COMMON_PREFIX_TERMINAL_R136_20261006.md';assert not (repo/doc).exists();shutil.copy2(Path(__file__).with_name(doc),repo/doc)
    name=Path(__file__).name;assert not (repo/'scripts'/name).exists();shutil.copy2(__file__,repo/'scripts'/name)
    subprocess.run(['git','add',doc,'scripts/'+name],cwd=repo,check=True);subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve rejected common-prefix terminal R136'],cwd=repo,check=True)
    bundle=ROOT/'tmp/getup_common_prefix_terminal_r136_20261006.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle))),flush=True)


if __name__=='__main__':main()
