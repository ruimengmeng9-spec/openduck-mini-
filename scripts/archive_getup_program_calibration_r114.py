"""Append complete immutable R114 terminal evidence after natural completion."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest
from diagnostics.train_getup_program_calibration_r114 import ROOT,OUTPUT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    result=json.loads((OUTPUT/'results.json').read_text());assert result['development']['successes']==24
    target=repo/'results'/OUTPUT.name/'terminal_snapshot';assert not target.exists()
    shutil.copytree(OUTPUT,target);shutil.copy2(OUTPUT.with_suffix('.log'),target/'parent.log')
    cwd=ROOT/'projects/Open_Duck_Playground'
    test=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_program_calibration_r114','-v'],
        cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (target/'snapshot_regression_tests.log').write_text(test.stdout);assert test.returncode==0
    (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=True,full_task_completed=False,
        hardware_readiness=False,independent_qualification_run=result['independent_qualification_run']),indent=2))
    (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file()},indent=2))
    names=('train_getup_program_calibration_r114.py','test_getup_program_calibration_r114.py',
        'launch_getup_program_calibration_r114.py',Path(__file__).name)
    for name in names:
        dst=repo/'scripts'/name;assert not dst.exists();shutil.copy2(Path(__file__).with_name(name),dst)
    doc='GETUP_PROGRAM_CALIBRATION_R114_20261005.md';assert not (repo/doc).exists()
    shutil.copy2(Path(__file__).with_name(doc),repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve node-decoder calibration and full independent validation R114'],cwd=repo,check=True)
    bundle=ROOT/'tmp/getup_program_calibration_r114_terminal_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        development_successes=result['development']['successes'],left_stage_passed=result['left_stage_passed'])),flush=True)


if __name__=='__main__':main()
