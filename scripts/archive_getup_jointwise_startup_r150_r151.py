"""Append immutable audit and closed full startup, not live training folders."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import train_getup_jointwise_dynamic_r151 as run
from diagnostics.audit_getup_rank_coupling_r150 import OUTPUT as AUDIT
from diagnostics.getup_independent_native import digest

def hash_folder(target):
    (target/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(target)):digest(p) for p in target.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args();repo=run.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    names=['audit_getup_rank_coupling_r150.py','train_getup_jointwise_dynamic_r151.py','test_getup_jointwise_dynamic_r151.py','launch_getup_jointwise_dynamic_r151.py',Path(__file__).name]
    doc='GETUP_JOINTWISE_DYNAMIC_R150_R151_20261006.md'
    assert all(not (repo/'scripts'/n).exists() for n in names) and not (repo/doc).exists()
    sources=((AUDIT,'terminal_snapshot'),(run.SMOKE,'terminal_snapshot'),(run.OUTPUT,'startup_closed_01'))
    assert all(not (repo/'results'/s.name/snapshot).exists() for s,snapshot in sources)
    audit=json.loads((AUDIT/'results.json').read_text());assert len(audit['rows'])==25 and all(r['scalar_actual_execution_exact'] for r in audit['rows'])
    smoke=json.loads((run.SMOKE/'results.json').read_text());assert smoke['smoke']
    assert (run.SMOKE/'tests.log').read_text().strip().endswith('OK')
    formal=json.loads((run.OUTPUT/'startup_closed.json').read_text())
    measurements=[]
    for label,result,path in [('independent',smoke,run.SMOKE),('formal',formal,run.OUTPUT)]:
        for folder in ('zero_parity','nonzero_smoke'):
            assert len(result['parity' if folder=='zero_parity' else 'nonzero']['rows'])==3
        for row in result['parity']['rows']:assert all(row['frozen_R134_trace_bitwise_equal'].values())
        for row in result['nonzero']['rows']:
            case=row['case_seed']
            with np.load(path/'nonzero_smoke'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as z:
                activation=float(np.abs(z['jointwise_activation']).max());change=float(np.abs(z['jointwise_gains']-row['base_gains']).max())
                assert (activation==0 and change==0) if case is None else (activation>0 and change>0)
                if label=='formal':
                    with np.load(run.SMOKE/'nonzero_smoke'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as old:
                        assert all(np.array_equal(z[k],old[k]) for k in z.files)
            measurements.append(dict(group=label,case_seed=case,maximum_channel_activation=activation,maximum_gain_change=change,success=row['success'],valid=row['valid']))
    targets=[]
    for source,snapshot in sources[:2]:
        target=repo/'results'/source.name/snapshot;shutil.copytree(source,target);hash_folder(target);targets.append(target)
    target=repo/'results'/run.OUTPUT.name/'startup_closed_01';target.mkdir(parents=True)
    for folder in ('executed_sources','frozen','zero_parity','nonzero_smoke'):shutil.copytree(run.OUTPUT/folder,target/folder)
    for file in ('contract.json','startup_closed.json'):shutil.copy2(run.OUTPUT/file,target/file)
    shutil.copy2(run.SMOKE/'tests.log',target/'tests.log')
    (target/'measurements.json').write_text(json.dumps(measurements,indent=2))
    (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=False,training_generations_not_included=True,
        formal_six_full_startup_closed=True,independent_six_full_smoke_closed=True,
        no_expanded_or_independent_qualification=True,full_task_completed=False,hardware_readiness=False),indent=2))
    hash_folder(target);targets.append(target)
    for name in names:shutil.copy2(Path(__file__).with_name(name),repo/'scripts'/name)
    shutil.copy2(Path(__file__).with_name(doc),repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R150 rank coupling audit and R151 independent causal joint startup'],cwd=repo,check=True)
    bundle=run.ROOT/'tmp/getup_jointwise_startup_r150_r151_20261006.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),bundle_sha256=digest(bundle),bundle_bytes=bundle.stat().st_size,measurements=measurements)),flush=True)

if __name__=='__main__':main()
