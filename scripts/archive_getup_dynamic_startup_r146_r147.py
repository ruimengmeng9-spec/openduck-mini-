import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics.train_getup_dynamic_gain_r147 import ROOT,OUTPUT
from diagnostics.audit_getup_dynamic_feedback_r146 import OUTPUT as AUDIT
from diagnostics.getup_independent_native import digest


def hash_folder(target):
    (target/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args();repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    # All dependencies must be closed before creating any publication path.
    for folder in ('zero_parity','nonzero_smoke'):
        assert len(json.loads((OUTPUT/folder/'results.json').read_text())['rows'])==3
    for source,snapshot in ((AUDIT,'terminal_snapshot'),(ROOT/'outputs/getup_dynamic_gain_r147_smoke_20261006','terminal_snapshot'),(OUTPUT,'startup_closed_01')):
        assert not (repo/'results'/source.name/snapshot).exists()
    targets=[]
    audit=json.loads((AUDIT/'results.json').read_text());assert len(audit['rows'])==10
    target=repo/'results'/AUDIT.name/'terminal_snapshot';assert not target.exists();shutil.copytree(AUDIT,target)
    hash_folder(target);targets.append(target)
    smoke=ROOT/'outputs/getup_dynamic_gain_r147_smoke_20261006'
    result=json.loads((smoke/'results.json').read_text());assert result['smoke'] and len(result['parity']['rows'])==3 and len(result['nonzero']['rows'])==3
    assert (smoke/'tests.log').read_text().strip().endswith('OK')
    target=repo/'results'/smoke.name/'terminal_snapshot';assert not target.exists();shutil.copytree(smoke,target)
    # Preserve measured activation and motion change, not just interface shape.
    measurements=[]
    for row in result['nonzero']['rows']:
        case=row['case_seed'];path=target/'nonzero_smoke'/f'case_{case}'/'trajectory.npz'
        with np.load(path,allow_pickle=False) as z:
            maximum=float(np.abs(z['dynamic_activation']).max())
            gain_change=float(np.abs(z['dynamic_gains']-np.array(row['base_gains'])).max())
        assert (maximum==0 and gain_change==0) if case is None else (maximum>0 and gain_change>0)
        measurements.append(dict(case_seed=case,maximum_dynamic_activation=maximum,maximum_gain_change=gain_change,success=row['success'],valid=row['valid']))
    (target/'measured_dynamic_activation.json').write_text(json.dumps(measurements,indent=2))
    hash_folder(target);targets.append(target)
    formal=repo/'results'/OUTPUT.name/'startup_closed_01';assert not formal.exists();formal.mkdir(parents=True)
    for folder in ('executed_sources','frozen','zero_parity','nonzero_smoke'):
        shutil.copytree(OUTPUT/folder,formal/folder)
    for file in ('contract.json',):shutil.copy2(OUTPUT/file,formal/file)
    for folder in ('zero_parity','nonzero_smoke'):
        report=json.loads((formal/folder/'results.json').read_text());assert len(report['rows'])==3
    shutil.copy2(smoke/'tests.log',formal/'tests.log')
    shutil.copy2(OUTPUT.with_suffix('.log'),formal/'startup_log.log')
    (formal/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=False,
        training_generations_not_included=True,zero_and_nominal_full_parity_closed=True,
        no_expanded_or_independent_qualification=True,full_task_completed=False,hardware_readiness=False),indent=2))
    hash_folder(formal);targets.append(formal)
    names=['audit_getup_dynamic_feedback_r146.py','train_getup_dynamic_gain_r147.py','test_getup_dynamic_gain_r147.py',
        'launch_getup_dynamic_gain_r147.py',Path(__file__).name]
    for name in names:
        assert not (repo/'scripts'/name).exists();shutil.copy2(Path(__file__).with_name(name),repo/'scripts'/name)
    doc='GETUP_DYNAMIC_STATE_FEEDBACK_R146_R147_20261006.md';assert not (repo/doc).exists();shutil.copy2(Path(__file__).with_name(doc),repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R146 paired feedback evidence and R147 bounded dynamic startup'],cwd=repo,check=True)
    bundle=ROOT/'tmp/getup_dynamic_startup_r146_r147_20261006.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),smoke_measurements=measurements)),flush=True)


if __name__=='__main__':main()
