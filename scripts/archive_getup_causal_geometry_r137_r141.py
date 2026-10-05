import argparse
import json
from pathlib import Path
import shutil
import subprocess
from diagnostics.audit_getup_sensor_geometry_r137 import ROOT,OUTPUT as R137
from diagnostics.audit_getup_geometry_prediction_r138 import OUTPUT as R138
from diagnostics.audit_getup_nominal_geometry_r139 import OUTPUT as R139
from diagnostics.probe_getup_causal_acceleration_r140 import OUTPUT as R140
from diagnostics.audit_getup_pair_distance_r142 import OUTPUT as R142
from diagnostics.probe_getup_passive_risk_r141b import OUTPUT as R141
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);args=p.parse_args();repo=ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==args.base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    smoke=ROOT/'outputs/getup_passive_risk_r141b_smoke_20261006'
    sr=json.loads((smoke/'results.json').read_text());assert sr['all_full_trace_bitwise_equal'] and len(sr['rows'])==2
    copies=[]
    failed=ROOT/'outputs/getup_passive_risk_r141_smoke_20261006'
    failure_target=repo/'results'/failed.name/'failed_interface_snapshot'
    assert not failure_target.exists();shutil.copytree(failed,failure_target)
    (failure_target/'failure_note.json').write_text(json.dumps(dict(terminal_result_saved=False,
        failure='native50 equality failed at history-target fields because mj_step entry is after step_target updated prev',
        original_complete_trace_and_physical_peak_checks_passed_before_sensor_assertion=True,
        original_source_and_data_preserved=True,no_assertion_weakened=True),indent=2));copies.append(failure_target)
    for source in (R137,R138,R139,R140,R142,ROOT/'outputs/getup_sensor_geometry_r137_smoke_20261006',smoke):
        assert (source/'results.json').exists()
        target=repo/'results'/source.name/'terminal_snapshot';assert not target.exists();shutil.copytree(source,target)
        (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=True,diagnostic_not_new_strategy=True),indent=2));copies.append(target)
    target=repo/'results'/R141.name/'startup_closed_01';assert not target.exists();target.mkdir(parents=True)
    shutil.copytree(R141/'executed_sources',target/'executed_sources')
    terminal_before_copy=(R141/'results.json').exists()
    for closed in R141.iterdir():
        if closed.is_dir() and (closed/'result.json').exists():
            row=json.loads((closed/'result.json').read_text())
            if 'full_trace_bitwise_equal' not in row:continue
            assert all(row['full_trace_bitwise_equal'].values()) and row['original_peaks_identical']
            shutil.copytree(closed,target/closed.name)
    if terminal_before_copy:
        terminal=json.loads((R141/'results.json').read_text())
        assert len(list(target.glob('*/result.json')))==len(terminal['rows'])
        shutil.copy2(R141/'results.json',target/'results.json')
    if (R141/'progress.json').exists():shutil.copy2(R141/'progress.json',target/'progress.json')
    if R141.with_suffix('.log').exists():shutil.copy2(R141.with_suffix('.log'),target/R141.with_suffix('.log').name)
    (target/'snapshot.json').write_text(json.dumps(dict(terminal_result_saved=(target/'results.json').exists(),
        closed_replay_folders=len(list(target.glob('*/result.json'))),not_new_model_training=True,full_task_completed=False),indent=2));copies.append(target)
    test=subprocess.run([str(ROOT/'projects/Open_Duck_Playground/.venv/bin/python'),'-m','unittest','diagnostics.test_getup_sensor_geometry_r137','-v'],cwd=ROOT/'projects/Open_Duck_Playground',stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    assert test.returncode==0;(copies[1]/'archive_regression_tests.log').write_text(test.stdout)
    for target in copies:(target/'artifact_hashes.json').write_text(json.dumps({str(f.relative_to(target)):digest(f) for f in target.rglob('*') if f.is_file() and f.name!='artifact_hashes.json'},indent=2))
    files=['audit_getup_sensor_geometry_r137.py','test_getup_sensor_geometry_r137.py','audit_getup_geometry_prediction_r138.py','audit_getup_nominal_geometry_r139.py',
        'probe_getup_causal_acceleration_r140.py','probe_getup_passive_risk_r141.py','launch_getup_passive_risk_r141.py',
        'probe_getup_passive_risk_r141b.py','launch_getup_passive_risk_r141b.py','audit_getup_pair_distance_r142.py',Path(__file__).name]
    for name in files:
        assert not (repo/'scripts'/name).exists();shutil.copy2(Path(__file__).with_name(name),repo/'scripts'/name)
    doc='GETUP_CAUSAL_GEOMETRY_R137_R142_20261006.md';assert not (repo/doc).exists();shutil.copy2(Path(__file__).with_name(doc),repo/doc)
    subprocess.run(['git','add',doc,*['scripts/'+n for n in files]],cwd=repo,check=True)
    subprocess.run(['git','add','-f',*[str(t.relative_to(repo)) for t in copies]],cwd=repo,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Save causal sensor geometry diagnostics R137-R140 and passive replay R141'],cwd=repo,check=True)
    bundle=ROOT/'tmp/getup_causal_geometry_r137_r141_20261006.bundle';assert not bundle.exists()
    subprocess.run(['git','bundle','create',str(bundle),'HEAD','^'+args.base],cwd=repo,check=True)
    print(json.dumps(dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),bundle=str(bundle),r141_terminal_saved=(copies[-1]/'results.json').exists())),flush=True)


if __name__=='__main__':main()
