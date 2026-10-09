"""Unique closed terminal archive, only after previous publisher completed."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import train_getup_foot_task_r177 as run
from diagnostics import audit_getup_foot_task_terminal_r178 as audit


def hashes(root):return {str(p.relative_to(root)):run.prior.digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
    parser.add_argument('--publication-receipt',type=Path,default=run.ROOT/'tmp/getup_publication_r175_r177_20261009_result.json')
    args=parser.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_FOOT_TASK_TERMINAL_R177_R178_20261009.md'
    publication=json.loads(args.publication_receipt.read_text())
    assert publication['independent_remote_verification'] and publication['published_startup']==args.base
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    assert not (repo/doc).exists()
    terminal=json.loads((run.OUTPUT/'results.json').read_text());closed=json.loads((run.OUTPUT/'training_closed.json').read_text())
    report=json.loads((audit.OUTPUT/'results.json').read_text())
    assert terminal['terminal_result_saved'] and len(closed['history'])==2 and terminal['parameters']==[0.,0.]
    assert report['terminal_result_saved'] and report['scalar_programs']==10 and report['scalar_trajectories_audited']==250 and report['source_hashes_unchanged']
    assert len(report['terminal_pairing'])==25
    for case in (None,769002,773004):
        with np.load(audit.SMOKE/'probe_startup'/f'case_{case}'/'signals.npz',allow_pickle=False) as x,np.load(audit.OUTPUT/'probe_startup'/f'case_{case}'/'signals.npz',allow_pickle=False) as y:
            assert x.files==y.files
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    for source in (run.OUTPUT,audit.OUTPUT,audit.SMOKE):
        target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists();before=hashes(source);shutil.copytree(source,target)
        assert hashes(target)==before and hashes(source)==before;shutil.copy2(source.with_suffix('.log'),target/'process.log')
        if source==run.OUTPUT:
            for name,dest in [('getup_foot_task_r177_initial_regression_corrected_20261009.log','initial_regression_corrected.log'),('getup_foot_task_r177_formal_regression_20261009.log','formal_regression.log')]:shutil.copy2(run.ROOT/'tmp'/name,target/dest)
        run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,training_generations_saved=2 if source==run.OUTPUT else 0,dynamic_attempts=306 if source==run.OUTPUT else 0,
            independent_qualification_run=False,full_task_completed=False,hardware_readiness=False,overlapping_probe_signal_arrays_bitwise_equal=True))
        run.local.write_json(target/'artifact_hashes.json',hashes(target));subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    publication_target=repo/'results/getup_publication_r175_r177_resume_20261010';assert not publication_target.exists();publication_target.mkdir(parents=True)
    shutil.copy2(args.publication_receipt,publication_target/'verified_startup_receipt.json')
    for name in ('finish_getup_publication_r175_r177_20261009.py','getup_publication_r175_r177_20261009.log','resume_getup_publication_r175_r178_20261010.py'):
        shutil.copy2(run.ROOT/'tmp'/name,publication_target/name)
    subprocess.run(['git','add','-f',str(publication_target.relative_to(repo))],cwd=repo,check=True)
    for name in (Path(audit.__file__).name,Path(__file__).name):
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest);subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R177 foot task terminal failures and R178 canonical kinematic audit'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),terminal_result_saved=True,full_task_completed=False)),flush=True)


if __name__=='__main__':main()
