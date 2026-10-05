import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.probe_getup_passive_risk_r141b import ROOT,OUTPUT as R141
from diagnostics.audit_getup_distance_contract_r143 import OUTPUT as FAILED
from diagnostics.audit_getup_distance_contract_r143b import OUTPUT as R143
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args();repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    result=json.loads((R141/'results.json').read_text());assert len(result['rows'])==16 and result['all_full_trace_bitwise_equal']
    assert sum(not r['valid'] for r in result['rows'])==5
    metric=json.loads((R143/'results.json').read_text());assert len(metric['rows'])==23 and metric['deepest_contact_match_counts']==[0,0,0,0]
    targets=[]
    for source,snapshot in [(R141,'terminal_snapshot'),(R143,'terminal_snapshot'),(FAILED,'failed_interface_snapshot')]:
        target=repo/'results'/source.name/snapshot;assert not target.exists();shutil.copytree(source,target)
        if source==R141:shutil.copy2(source.with_suffix('.log'),target/source.with_suffix('.log').name)
        (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=(target/'results.json').exists(),
            not_new_strategy=True,original_labels_unchanged=True,full_task_completed=False),indent=2))
        if source==FAILED:(target/'failure_note.json').write_text(json.dumps(dict(error='AttributeError: mjtEnableBit has no mjENBL_MULTICCD',
            failed_before_metric_comparison=True,source_preserved=True),indent=2))
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2));targets.append(target)
    names=['audit_getup_distance_contract_r143.py','audit_getup_distance_contract_r143b.py',Path(__file__).name]
    for name in names:
        assert not (repo/'scripts'/name).exists();shutil.copy2(Path(__file__).with_name(name),repo/'scripts'/name)
    doc='GETUP_PASSIVE_GEOMETRY_TERMINAL_R141_R143_20261006.md';assert not (repo/doc).exists();shutil.copy2(Path(__file__).with_name(doc),repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve passive sensor parity terminal and distance contract R141-R143'],cwd=repo,check=True)
    bundle=ROOT/'tmp/getup_passive_terminal_r141_r143_20261006.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle))),flush=True)


if __name__=='__main__':main()
