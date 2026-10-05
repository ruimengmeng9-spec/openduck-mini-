"""Archive R102 terminal evidence and closed R104 counterfactuals only."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck');PROJECT=ROOT/'projects/Open_Duck_Playground'
REPO=ROOT/'github/openduck-mini-'


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    source=ROOT/'outputs/getup_joint_anchor_r102_left_20261005'
    result=json.loads((source/'results.json').read_text());assert not result['candidate_promoted']
    target=REPO/'results'/source.name/'terminal_snapshot';target.mkdir(exist_ok=False)
    for name in ('results.json','contract.json','progress.json','history.json'):
        shutil.copy2(source/name,target/name)
    for file in source.glob('checkpoint_*.npz'):shutil.copy2(file,target/file.name)
    for name in ('development_candidate','development_baseline'):
        shutil.copytree(source/name,target/name)
    shutil.copy2(source.with_suffix('.log'),target/'parent.log')
    audit=ROOT/'outputs/getup_endpoint_audit_r103_20261005'
    shutil.copytree(audit,REPO/'results'/audit.name)
    probe=ROOT/'outputs/getup_temporal_gate_r104_left_20261005'
    snapshot=REPO/'results'/probe.name/'saved_snapshot';snapshot.mkdir(parents=True,exist_ok=False)
    for name in ('contract.json','frozen_controller.npz'):
        shutil.copy2(probe/name,snapshot/name)
    shutil.copytree(probe/'executed_sources',snapshot/'executed_sources')
    for name in ('early','late','local','prefix','independent_candidate','independent_baseline'):
        if (probe/name/'results.json').exists():shutil.copytree(probe/name,snapshot/name)
    for name in ('partial_results.json','results.json','frozen_selection.json'):
        if (probe/name).exists():
            data=json.loads((probe/name).read_text());(snapshot/name).write_text(json.dumps(data,indent=2))
    shutil.copy2(probe.with_suffix('.log'),snapshot/'parent.log')
    scope=dict(simulation_only=True,hardware_readiness=False,full_task_completed=False,
        r102_terminal_complete=True,r104_terminal_complete=(snapshot/'results.json').exists(),
        r104_closed_groups=[d.name for d in snapshot.iterdir() if d.is_dir() and (d/'results.json').exists()])
    (snapshot/'archive_scope.json').write_text(json.dumps(scope,indent=2))
    paths=[target,REPO/'results'/audit.name,snapshot]
    for name in ('getup_alignment_audit_r105_20261005','getup_contact_audit_r106_20261005','getup_state_mixture_r107_smoke_20261005'):
        source_dir=ROOT/'outputs'/name
        assert (source_dir/'results.json').exists()
        destination=REPO/'results'/name;shutil.copytree(source_dir,destination);paths.append(destination)
    live=ROOT/'outputs/getup_state_mixture_r107_left_20261005'
    startup=REPO/'results'/live.name/'startup_snapshot';startup.mkdir(parents=True,exist_ok=False)
    for name in ('contract.json','frozen_profiles.npz'):
        shutil.copy2(live/name,startup/name)
    shutil.copytree(live/'executed_sources',startup/'executed_sources')
    for name in ('parity_zero','parity_one'):
        assert (live/name/'results.json').exists()
        shutil.copytree(live/name,startup/name)
    shutil.copy2(live.with_suffix('.log'),startup/'parent_at_archive.log')
    (startup/'archive_scope.json').write_text(json.dumps(dict(startup_only=True,terminal_result_saved=False,
        formal_profile_parity_complete=True,simulation_only=True,hardware_readiness=False),indent=2))
    paths.append(startup)
    (snapshot/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(REPO)):digest(f)
        for directory in paths for f in directory.rglob('*') if f.is_file()},indent=2))
    names=['audit_getup_endpoint_r103.py','probe_getup_temporal_gate_r104.py','test_getup_temporal_gate_r104.py',
        'audit_getup_alignment_r105.py','audit_getup_contacts_r106.py','train_getup_state_mixture_r107.py',
        'test_getup_state_mixture_r107.py',Path(__file__).name]
    for name in names:shutil.copy2(PROJECT/'diagnostics'/name,REPO/'scripts'/name)
    launch='launch_getup_temporal_gate_r104.sh';shutil.copy2(PROJECT/'scripts'/launch,REPO/'scripts'/launch)
    next_launch='launch_getup_state_mixture_r107.sh';shutil.copy2(PROJECT/'scripts'/next_launch,REPO/'scripts'/next_launch)
    doc='GETUP_TEMPORAL_FEEDBACK_R103_R104_20261005.md';shutil.copy2(PROJECT/doc,REPO/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in names],'scripts/'+launch,'scripts/'+next_launch],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',*[str(path.relative_to(REPO)) for path in paths]],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','-m','Save R102 terminal, R103-R106 counterfactual audits and R107 state mixture startup'],cwd=REPO,check=True)
    bundle=ROOT/'tmp/getup_temporal_r104_20261005.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=REPO,check=True)
    print('R104_BUNDLE',bundle,subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),flush=True)


if __name__=='__main__':main()
