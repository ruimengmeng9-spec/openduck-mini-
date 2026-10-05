"""Append closed read-only audit and dynamic gate ablation evidence."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics import audit_getup_program_gate_r123 as audit
from diagnostics import probe_getup_continuous_nodes_r124 as trial
from diagnostics.getup_independent_native import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--snapshot',required=True)
    parser.add_argument('--document',required=True);args=parser.parse_args()
    assert args.snapshot.replace('_','').isalnum() and Path(args.document).name==args.document
    repo=audit.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    sources=[audit.OUTPUT,audit.ROOT/'outputs/getup_continuous_nodes_r124_smoke_20261005',
        audit.ROOT/'outputs/getup_continuous_nodes_r124_smoke_02_20261005',trial.OUTPUT]
    reports={};targets=[]
    for source in sources:
        assert source.exists()
        target=repo/'results'/source.name/args.snapshot;assert not target.exists()
        target.mkdir(parents=True);targets.append(target)
        for group in ('executed_sources','frozen_models'):
            if (source/group).exists():shutil.copytree(source/group,target/group)
        for name in ('results.json','progress.json','contract.json','prediction_parity.json',Path(audit.__file__).name):
            if (source/name).exists():shutil.copy2(source/name,target/name)
        count=0
        for marker in source.rglob('result.json'):
            if 'executed_sources' in marker.parts:continue
            shutil.copytree(marker.parent,target/marker.parent.relative_to(source));count+=1
        for path in (source.with_suffix('.log'),source.with_name(source.name+'_tests.log')):
            if path.exists():shutil.copy2(path,target/path.name)
        record=dict(terminal_result_saved=(target/'results.json').exists(),completed_trials=count,
            dynamic_integration=False if source==audit.OUTPUT else True,
            independent_qualification_run=False,hardware_readiness=False,full_task_completed=False)
        (target/'snapshot.json').write_text(json.dumps(record,indent=2));reports[source.name]=record
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file()},indent=2))
    cwd=audit.ROOT/'projects/Open_Duck_Playground'
    tests=subprocess.run([str(cwd/'.venv/bin/python'),'-m','unittest','diagnostics.test_getup_continuous_nodes_r124','-v'],cwd=cwd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert tests.returncode==0
    (targets[-1]/'archive_regression_tests.log').write_text(tests.stdout)
    for target in targets:
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f)
            for f in target.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2))
    names=['audit_getup_program_gate_r123.py','probe_getup_continuous_nodes_r124.py','test_getup_continuous_nodes_r124.py','launch_getup_continuous_nodes_r124.py',Path(__file__).name]
    for name in names:
        src=Path(__file__).with_name(name);dst=repo/'scripts'/name
        if dst.exists():assert digest(src)==digest(dst)
        else:shutil.copy2(src,dst)
    doc=repo/args.document;assert not doc.exists();shutil.copy2(Path(__file__).with_name(args.document),doc)
    subprocess.run(['git','add',args.document,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve program gate audit R123 and frozen ablation R124 '+args.snapshot],cwd=repo,check=True)
    bundle=audit.ROOT/'tmp'/('getup_program_gate_r123_r124_'+args.snapshot+'_20261005.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),reports=reports)),flush=True)


if __name__=='__main__':main()
