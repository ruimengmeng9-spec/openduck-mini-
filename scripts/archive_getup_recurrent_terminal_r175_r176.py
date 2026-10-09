"""Append immutable closed R175/R176 terminal evidence, direct server publication."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics import train_getup_recurrent_readout_r175 as run
from diagnostics import audit_getup_recurrent_terminal_r176 as audit
from diagnostics.getup_independent_native import digest


def hashes(root):
    return {str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    repo=run.ROOT/'github/openduck-mini-';doc='GETUP_RECURRENT_TERMINAL_R175_R176_20261009.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    assert not (repo/doc).exists()
    terminal=json.loads((run.OUTPUT/'results.json').read_text())
    closed=json.loads((run.OUTPUT/'training_closed.json').read_text())
    report=json.loads((audit.OUTPUT/'results.json').read_text())
    assert terminal['terminal_result_saved'] and len(closed['history'])==2
    assert report['terminal_result_saved'] and report['scalar_programs']==13 and report['scalar_trajectories_audited']==325 and report['source_hashes_unchanged']
    assert len(report['terminal_pairing'])==25
    # Independently compare the three overlapping saved PROBE signal arrays.
    for case in (None,769002,773004):
        paths=[p/'probe_startup'/f'case_{case}'/'signals.npz' for p in (audit.SMOKE,audit.OUTPUT)]
        with np.load(paths[0],allow_pickle=False) as x,np.load(paths[1],allow_pickle=False) as y:
            assert x.files==y.files
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    sources=[run.OUTPUT,audit.OUTPUT,audit.SMOKE]
    for source in sources:
        target=repo/'results'/source.name/'terminal_snapshot'
        assert not target.exists();before=hashes(source)
        shutil.copytree(source,target)
        assert hashes(target)==before and hashes(source)==before
        shutil.copy2(source.with_suffix('.log'),target/'process.log')
        if source==run.OUTPUT:
            for name,dest in [('getup_recurrent_readout_r175_initial_regression_20261009.log','initial_regression.log'),('getup_recurrent_readout_r175_formal_regression_20261009.log','formal_regression.log')]:
                shutil.copy2(run.ROOT/'tmp'/name,target/dest)
        run.local.write_json(target/'snapshot.json',dict(terminal_result_saved=True,training_generations_saved=2 if source==run.OUTPUT else 0,
            dynamic_attempts=381 if source==run.OUTPUT else 0,independent_qualification_run=False,full_task_completed=False,hardware_readiness=False,
            overlapping_probe_signal_arrays_bitwise_equal=True))
        run.local.write_json(target/'artifact_hashes.json',hashes(target))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in [Path(audit.__file__).name,Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists();shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc);subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R175 recurrent terminal failures and R176 scalar recurrence audit'],cwd=repo,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),terminal_result_saved=True,full_task_completed=False)),flush=True)


if __name__=='__main__':main()
