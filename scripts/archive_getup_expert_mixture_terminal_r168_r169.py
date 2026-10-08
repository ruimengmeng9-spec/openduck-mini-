"""Append closed R168 and read-only R169 without replacing prior snapshots."""
import argparse
from pathlib import Path
import json
import shutil
import subprocess
from diagnostics import audit_getup_expert_mixture_terminal_r169 as audit

def hashes(root):
    return {str(p.relative_to(root)):audit.digest(p) for p in root.rglob('*') if p.is_file()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args()
    run=audit.run;repo=run.ROOT/'github/openduck-mini-'
    doc='GETUP_EXPERT_MIXTURE_TERMINAL_R168_R169_20261008.md'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    result=audit.read(run.OUTPUT/'results.json');ar=audit.read(audit.OUTPUT/'results.json')
    closed=audit.read(run.OUTPUT/'training_closed.json')
    assert len(closed['history'])==4 and ar['distinct_actual_programs']==25 and ar['nonzero_programs']==24
    assert ar['actual_training_attempts']==ar['scalar_trajectories_audited']==625
    assert result['parameters']==[[0.]*12 for _ in range(4)]
    assert result['candidate']['successes']==result['baseline']['successes']==18
    assert result['candidate']['nominal_success'] and result['baseline']['nominal_success']
    assert result['candidate']['physical_failures']==result['baseline']['physical_failures']==0
    assert result['terminal_result_saved'] and not result['original_development_gate']
    assert not result['expanded_development_run'] and not result['independent_qualification_run']
    assert ar['read_only'] and not ar['smoke'] and len(ar['terminal_parity'])==25
    assert ar['source_hashes_unchanged'] and ar['independent_smoke_bitwise_equal']
    assert len(list(run.OUTPUT.rglob('trajectory.npz')))==681
    smoke=run.ROOT/'outputs/getup_expert_mixture_terminal_audit_r169_smoke_20261008'
    sr=audit.read(smoke/'results.json');assert sr['smoke'] and sr['scalar_trajectories_audited']==6
    bundle=run.ROOT/'tmp/getup_expert_mixture_terminal_r168_r169_20261008.bundle'
    assert not bundle.exists() and not (repo/doc).exists();saved=[]
    for source,readonly in [(run.OUTPUT,False),(audit.OUTPUT,True),(smoke,True)]:
        before=hashes(source);target=repo/'results'/source.name/'terminal_snapshot'
        assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copytree(source,target)
        log=source.with_suffix('.log')
        if log.exists():shutil.copy2(log,target/'process.log')
        if not readonly:shutil.copy2(run.ROOT/'tmp/getup_expert_mixture_r168_regression_20261008.log',target/'regression.log')
        assert before==hashes(source),'Closed source changed during archive'
        trajectories=len(list(source.rglob('trajectory.npz')))
        saved.append(dict(source=str(source),trajectories=trajectories))
        audit.write(target/'snapshot.json',dict(terminal_result_saved=True,read_only=readonly,source=str(source),
            saved_trajectories=trajectories,training_generations_saved=0 if readonly else 4,
            all_training_uses_original_full_path_acceptance=True,physical_invalidity_can_terminate_early=True,
            full_task_completed=False,hardware_readiness=False,independent_qualification_run=False))
        audit.write(target/'artifact_hashes.json',hashes(target))
        subprocess.run(['git','add','-f',str(target.relative_to(repo))],cwd=repo,check=True)
    for name in [Path(audit.__file__).name,Path(__file__).name]:
        dest=repo/'scripts'/name;assert not dest.exists()
        shutil.copy2(Path(__file__).with_name(name),dest)
        subprocess.run(['git','add','scripts/'+name],cwd=repo,check=True)
    shutil.copy2(run.ROOT/'tmp'/doc,repo/doc)
    subprocess.run(['git','add',doc],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m',
        'Preserve R168 four-generation complete-path terminal failures and R169 exact causal expert-output audit'],cwd=repo,check=True)
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        bundle=str(bundle),bundle_sha256=audit.digest(bundle),bytes=bundle.stat().st_size,saved=saved)),flush=True)

if __name__=='__main__':main()
