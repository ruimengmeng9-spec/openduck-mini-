"""Append finished R108 and R109 smoke/startup, never overwrite old evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck');REPO=ROOT/'github/openduck-mini-'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    source=ROOT/'outputs/getup_initial_factors_r108_left_20261005'
    result=json.loads((source/'results.json').read_text());rows=result['rows']
    assert len(rows)==384 and len(list(source.rglob('trajectory.npz')))==388
    expected={'000':(24,0,24,0),'001':(11,1,8,0),'010':(7,4,6,2),'011':(9,1,7,2),
        '100':(6,2,7,1),'101':(7,2,4,2),'110':(9,1,11,0),'111':(12,0,13,0)}
    summary={}
    for mask,counts in expected.items():
        numbers=[]
        for profile in ('zero','r102'):
            group=[r for r in rows if ''.join(map(str,r['mask']))==mask and r['profile']==profile]
            assert len(group)==24
            numbers.extend((sum(r['success'] for r in group),sum(not r['valid'] for r in group)))
            if mask=='000':assert len({r['initial_hash'] for r in group})==1
            if mask=='111':assert all(r['original_full_path_bitwise_parity'] for r in group)
        assert tuple(numbers)==counts
        summary[mask]=numbers
    target=REPO/'results'/source.name/'terminal_snapshot'
    assert not target.exists();shutil.copytree(source,target)
    shutil.copy2(source.with_suffix('.log'),target/'parent.log')
    (target/'factor_summary.json').write_text(json.dumps(dict(counts=summary,
        columns=['zero_success','zero_invalid','r102_success','r102_invalid'],
        diagnostic_only=True,qualification_evidence=False),indent=2))
    smoke=ROOT/'outputs/getup_case_teachers_r109_smoke_20261005'
    assert json.loads((smoke/'results.json').read_text())['smoke']
    smoke_target=REPO/'results'/smoke.name
    assert not smoke_target.exists();shutil.copytree(smoke,smoke_target)
    shutil.copy2(ROOT/'outputs/getup_case_teachers_r109_tests_20261005.log',smoke_target/'tests.log')
    live=ROOT/'outputs/getup_case_teachers_r109_left_20261005'
    assert (live/'contract.json').exists() and (live/'parity.json').exists()
    live_target=REPO/'results'/live.name/'startup_snapshot';live_target.mkdir(parents=True,exist_ok=False)
    for name in ('contract.json','parity.json','frozen_profiles.npz'):shutil.copy2(live/name,live_target/name)
    shutil.copytree(live/'executed_sources',live_target/'executed_sources')
    (live_target/'archive_scope.json').write_text(json.dumps(dict(teacher_data_only=True,
        R109_terminal=False,unified_policy_success=False,full_task_completed=False,hardware_readiness=False),indent=2))
    names=['search_getup_case_teachers_r109.py','test_getup_case_teachers_r109.py',
        'launch_getup_case_teachers_r109.py',Path(__file__).name]
    for name in names:shutil.copy2(Path(__file__).with_name(name),REPO/'scripts'/name)
    doc='GETUP_CASE_TEACHERS_R108_R109_20261005.md'
    assert not (REPO/doc).exists();shutil.copy2(Path(__file__).with_name(doc),REPO/doc)
    paths=[target,smoke_target,live_target]
    for path in paths:
        (path/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(path)):digest(f)
            for f in path.rglob('*') if f.is_file()},indent=2))
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',*[str(p.relative_to(REPO)) for p in paths]],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','-m','Save R108 factorial failures and start complete-fall teacher search R109'],cwd=REPO,check=True)
    bundle=ROOT/'tmp/getup_factors_r108_teachers_r109_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=REPO,check=True)
    print('BUNDLE',bundle,subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),flush=True)


if __name__=='__main__':main()
