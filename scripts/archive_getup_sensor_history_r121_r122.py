"""Append immutable history audit and matched training snapshots to publisher."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics import probe_getup_sensor_history_r121 as r121
from diagnostics import train_getup_history_program_r122 as r122
from diagnostics.getup_independent_native import digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True)
    parser.add_argument('--snapshot',required=True);parser.add_argument('--document',required=True)
    args=parser.parse_args();assert args.snapshot.replace('_','').isalnum()
    doc=Path(args.document);assert doc.name==args.document and doc.suffix=='.md'
    repo=r121.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    targets=[];reports={}
    sources=[r121.OUTPUT,r121.ROOT/'outputs/getup_sensor_history_r121_smoke_20261005',
        r121.ROOT/'outputs/getup_history_program_r122_smoke_20261005',
        r121.ROOT/'outputs/getup_history_program_r122_smoke_02_20261005',r122.OUTPUT]
    assert (r121.OUTPUT/'results.json').exists() and (r122.OUTPUT/'training/rng.json').exists()
    for source in sources:
        if not source.exists():continue
        target=repo/'results'/source.name/args.snapshot;assert not target.exists()
        target.mkdir(parents=True);targets.append(target)
        if (source/'executed_sources').exists():shutil.copytree(source/'executed_sources',target/'executed_sources')
        for name in ('contract.json','causal_sensor_dataset.npz','manifest.json','progress.json','results.json'):
            if (source/name).exists():shutil.copy2(source/name,target/name)
        if (source/'training/rng.json').exists():shutil.copytree(source/'training',target/'training')
        elif (source/'training/fit_results.json').exists():
            shutil.copytree(source/'training',target/'training_closed_before_path_error')
        count=0
        for marker in source.rglob('result.json'):
            if 'executed_sources' in marker.parts:continue
            shutil.copytree(marker.parent,target/marker.parent.relative_to(source));count+=1
        for suffix in ('.log','_tests.log'):
            path=source.with_suffix(suffix) if suffix=='.log' else source.with_name(source.name+suffix)
            if path.exists():shutil.copy2(path,target/path.name)
        record=dict(terminal_result_saved=(target/'results.json').exists(),completed_trials=count,
            hardware_readiness=False,full_task_completed=False,independent_qualification_run=False)
        if source.name=='getup_history_program_r122_smoke_20261005':
            record.update(aborted_before_any_rollout=True,error="TypeError: unsupported operand type(s) for /: 'str' and 'str' at original line 180; fixed in smoke_02, original executed source retained")
        if source==r122.OUTPUT and (target/'results.json').exists():
            terminal=json.loads((target/'results.json').read_text())
            old=json.loads((r121.ROOT/'outputs/getup_program_r113_left_20261005/results.json').read_text())['baseline_reused_from_R110']['rows']
            expanded=json.loads((r121.ROOT/'outputs/getup_program_calibration_r114_left_20261005/results.json').read_text())['independent']['baseline']['rows']
            former=json.loads((r121.FORMER/'results.json').read_text())['independent']['baseline']['rows']
            fixed={r['case_seed']:r for r in [*old,*expanded,*former]}
            paired={}
            for mode,groups in terminal['summaries'].items():
                rows=groups['all']['rows']
                assert all(r['initial_hash']==fixed[r['case_seed']]['initial_hash'] for r in rows)
                paired[mode]=dict(paired_hashes_equal=True,
                    rescued_vs_zero=[r['case_seed'] for r in rows if r['success'] and not fixed[r['case_seed']]['success']],
                    regressed_vs_zero=[r['case_seed'] for r in rows if not r['success'] and fixed[r['case_seed']]['success']],
                    development_gate=bool(groups['original']['nominal_success'] and groups['original']['successes']>=22
                        and groups['expanded']['successes']>=36 and groups['all']['physical_failures']==0),
                    fixed_comparison_counts=dict(original_zero=12,original_r102=13,expanded_zero=16,expanded_r102=9,former_zero=8),
                    independent_qualification_run=False,hardware_readiness=False)
            (target/'paired_development_audit.json').write_text(json.dumps(paired,indent=2))
        (target/'snapshot.json').write_text(json.dumps(record,indent=2));reports[source.name]=record
        (target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f)
            for f in target.rglob('*') if f.is_file()},indent=2))
    names=['probe_getup_sensor_history_r121.py','test_getup_sensor_history_r121.py','launch_getup_sensor_history_r121.py',
        'train_getup_history_program_r122.py','test_getup_history_program_r122.py','launch_getup_history_program_r122.py',Path(__file__).name]
    for name in names:
        src=Path(__file__).with_name(name);dst=repo/'scripts'/name
        if dst.exists():assert digest(src)==digest(dst)
        else:shutil.copy2(src,dst)
    destination=repo/doc.name;assert not destination.exists();shutil.copy2(Path(__file__).with_name(doc.name),destination)
    subprocess.run(['git','add',doc.name,*['scripts/'+n for n in names]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in targets]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve causal sensor history audit R121 and matched program experiment R122 '+args.snapshot],cwd=repo,check=True)
    bundle=r121.ROOT/'tmp'/('getup_history_r121_r122_'+args.snapshot+'_20261005.bundle');assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(bundle=str(bundle),head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),reports=reports)),flush=True)


if __name__=='__main__':main()
