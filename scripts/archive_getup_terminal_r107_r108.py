"""Append R107 terminal evidence and R108 launch evidence to clean publish repo."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from diagnostics.getup_independent_native import digest

ROOT=Path('/data/shijinsheng/open_duck')
REPO=ROOT/'github/openduck-mini-'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
    args=parser.parse_args()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    run=ROOT/'outputs/getup_state_mixture_r107_left_20261005'
    result=json.loads((run/'results.json').read_text())
    assert result['selected_parameters']==[1.,0.,0.,0.,0.]
    assert result['independent'] is None and not result['candidate_promoted']
    assert json.loads((run/'progress.json').read_text())['completed_generations']==24
    for name,count,folder in [('development',13,'development_candidate'),
        ('baseline',12,'development_baseline'),('previous_profile',13,'development_r102')]:
        group=result[name]
        assert group['successes']==count and group['nominal_success'] and group['physical_failures']==0
        assert len(group['rows'])==25
        for row in group['rows']:
            seed=row['case_seed']
            assert row['initial_hash']==next(r['initial_hash'] for r in result['baseline']['rows'] if r['case_seed']==seed)
            with np.load(run/folder/f'case_{seed}.npz') as data:
                assert len(data['time'])==2279
                if name=='development':
                    with np.load(run/'development_r102'/f'case_{seed}.npz') as old:
                        for key in ('qpos','qvel','applied','strict','normalized_residual'):
                            np.testing.assert_array_equal(data[key],old[key])
    destination=REPO/'results'/run.name/'terminal_snapshot'
    assert not destination.exists()
    shutil.copytree(run,destination)
    shutil.copy2(run.with_suffix('.log'),destination/'parent.log')
    for dirname in ('getup_initial_factors_r108_smoke_20261005',):
        source=ROOT/'outputs'/dirname
        target=REPO/'results'/dirname
        assert not target.exists();shutil.copytree(source,target)
    live=ROOT/'outputs/getup_initial_factors_r108_left_20261005'
    assert (live/'contract.json').exists()
    live_dest=REPO/'results'/live.name/'startup_snapshot'
    live_dest.mkdir(parents=True,exist_ok=False)
    shutil.copy2(live/'contract.json',live_dest/'contract.json')
    shutil.copytree(live/'executed_sources',live_dest/'executed_sources')
    if (live/'parity.json').exists():shutil.copy2(live/'parity.json',live_dest/'parity.json')
    write_scope=dict(R107_terminal=True,R108_terminal=False,R108_diagnostic_only=True,
        independent_seeds_unused=True,full_task_completed=False,hardware_readiness=False)
    (live_dest/'archive_scope.json').write_text(json.dumps(write_scope,indent=2))
    sources=Path(__file__).parent
    filenames=['probe_getup_initial_factors_r108.py','test_getup_initial_factors_r108.py',
        'launch_getup_initial_factors_r108.py',Path(__file__).name]
    for name in filenames:shutil.copy2(sources/name,REPO/'scripts'/name)
    doc='GETUP_INITIAL_FACTORS_R107_R108_20261005.md'
    assert not (REPO/doc).exists();shutil.copy2(sources/doc,REPO/doc)
    paths=[destination,REPO/'results/getup_initial_factors_r108_smoke_20261005',live_dest]
    for target in paths:
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f)
            for f in target.rglob('*') if f.is_file()},indent=2))
    subprocess.run(['git','add',doc,*['scripts/'+n for n in filenames]],cwd=REPO,check=True)
    subprocess.run(['git','add','-f',*[str(p.relative_to(REPO)) for p in paths]],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','-m','Preserve R107 terminal failure and launch R108 initial-state factorial diagnosis'],cwd=REPO,check=True)
    bundle=ROOT/'tmp/getup_terminal_r107_r108_20261005.bundle'
    assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=REPO,check=True)
    print('BUNDLE',bundle,subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),flush=True)


if __name__=='__main__':main()
