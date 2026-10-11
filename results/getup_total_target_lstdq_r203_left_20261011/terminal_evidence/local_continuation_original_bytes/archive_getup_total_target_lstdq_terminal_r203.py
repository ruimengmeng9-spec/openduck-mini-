"""Append-only R203 completed no-improvement archive; bind completed proof, never rerun decisions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
REPO=ROOT/'github/openduck-mini-'
BASE='8f19fcfb0ac6922818a4079f058d23cecd565e5c'
PROOF=ROOT/'tmp/getup_total_target_lstdq_r203_terminal_verification_20261011.json'
PROOF_SHA='5b4e3ae3f4311fa9a5a89b958e98fd1f8ed3c9b6f2a611308d72e553822d697c'
EXIT=ROOT/'tmp/getup_total_target_lstdq_r203_terminal_verification_exit_20261011.json'
VERIFY=ROOT/'outputs/getup_total_target_lstdq_r203_terminal_verification_20261011'
CONT=ROOT/'tmp/getup_total_target_lstdq_r203_continuation_20261011'
TERMINAL='GETUP_TOTAL_TARGET_LSTDQ_TERMINAL_R203_20261011.json'
PLAN='GETUP_TOTAL_TARGET_LSTDQ_PLAN_R203_20261011.json'
RESULT=ROOT/'tmp/getup_total_target_lstdq_r203_archive_result_20261011.json'
ARCHIVE_LOG=ROOT/'tmp/getup_total_target_lstdq_r203_archive_20261011.log'
PUBLISHER=ROOT/'tmp/publish_getup_total_target_lstdq_terminal_r203_20261011.py'
NUMSRC=ROOT/'tmp/getup_total_target_lstdq_numeric_r203_20261011'
FORMAL=ROOT/'outputs/getup_total_target_lstdq_r203_left_20261011'
LAUNCH=ROOT/'outputs/getup_total_target_lstdq_launcher_r203b_20261011'
PROTECTED=2664890
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def inventory(p):return {str(f.relative_to(p)):digest(f) for f in sorted(p.rglob('*')) if f.is_file()}
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def git(*args):return subprocess.check_output(['git','-c','gc.auto=0',*args],cwd=REPO,text=True).strip()
def idle():
    assert git('rev-parse','HEAD')==BASE
    assert not git('status','--porcelain')
    assert (Path('/proc')/str(PROTECTED)/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()=='.venv/bin/python -u -m open_duck_agent.live_server --host 127.0.0.1 --port 18766'
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines():
        first=Path(line.split()[0]).name if line.split() else ''
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in line and any(t in line for t in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_','verify_getup_'))),line
        assert 'publish_getup' not in line and 'git push' not in line,line
def check_sources(manifest,folder):
    for p,v in manifest.items():assert digest(p)==v['sha256']==digest(folder/v['copy']),p
def completed_proof():
    assert digest(PROOF)==PROOF_SHA
    proof=read(PROOF);e=read(EXIT)
    assert proof['audit_complete'] and proof['closed_saved_paths']==162 and proof['controls']==365248 and proof['new_dynamic_attempts']==0
    assert proof['formal_and_launcher_actual_natural_exit_zero'] and proof['dynamic_budget162_closed']
    assert proof['final_fallback_full_25_arrays_old_fields_peaks_exact'] and proof['candidate_R157_fallback_successes']==18
    assert not proof['additional_dynamic_budget_allowed'] and not proof['full_task_completed']
    assert e['natural_exit'] and e['exit_code']==0 and e['source_unchanged'] and e['proof_saved'] and e['proof_sha256']==PROOF_SHA
    assert e['source_sha256']==proof['verifier_source_sha256']==digest(PROJECT/'diagnostics/verify_getup_total_target_lstdq_terminal_r203.py')
    for p,items in proof['source_inventory'].items():assert inventory(Path(p))==items,p
    for p,count in proof['source_manifest_counts']:
        p=Path(p)
        if p.name.startswith('worker_source_manifest_'):
            folder=p.with_name(p.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''));manifest=read(p)
        elif (p/'source_manifest.json').exists():manifest=read(p/'source_manifest.json');folder=p/'executed_sources'
        else:manifest=read(p/'contract.json')['sources'];folder=p/'executed_sources'
        assert len(manifest)==count;check_sources(manifest,folder)
    check_sources(read(VERIFY/'source_manifest.json'),VERIFY/'executed_sources')
    for folder in (ROOT/'outputs/getup_total_target_lstdq_r203_smoke_20261011',FORMAL):
        c=read(folder/'contract.json')
        for p,sha in c['model_files'].items():assert digest(p)==sha
        for p,sha in c['hashes'].items():assert digest(p)==sha
        assert read(folder/'input_hashes_after.json')['source_and_model_files_unchanged']
    for p in (LAUNCH/'smoke_natural_exit.json',LAUNCH/'formal_natural_exit.json',ROOT/'tmp/getup_total_target_lstdq_r203b_launcher_exit_20261011.json'):
        e=read(p);assert e['natural_exit'] and e['exit_code']==0
    return proof
def original_sources():
    paths=sorted(p for p in (PROJECT/'diagnostics').iterdir() if p.is_file() and 'r203' in p.name.lower() and p.suffix=='.py')
    assert Path(__file__) in paths
    return paths
def all_live():
    files=set()
    for directory in (ROOT/'outputs',ROOT/'tmp'):
        for p in directory.iterdir():
            if 'r203' in p.name.lower():
                if p.is_file():files.add(p)
                elif p.is_dir():files.update(f for f in p.rglob('*') if f.is_file())
    files.update(original_sources())
    return sum(f.stat().st_size for f in files)
def prepare():
    idle();assert not (ROOT/'tmp'/TERMINAL).exists() and not RESULT.exists() and not CONT.exists()
    proof=completed_proof();dirs=[Path(p) for p in proof['source_inventory']]+[VERIFY]
    live=all_live();assert live+(640+640+128)*1024**2+6*1024**2<2*1024**3
    assert shutil.disk_usage(ROOT).free>=10*1024**3
    results=read(FORMAL/'results.json')
    assert results['terminal_result_saved'] and results['actual_dynamic_attempts']==156 and results['selected_R157_fallback'] and results['selected_checkpoint'] is None
    assert len(proof['two_complete_current_round_LSTDQ_solves_independently_recomputed'])==2
    document=dict(date='2026-10-11',terminal=True,stage='R203_completed_no_improvement_independently_verified_archive_preparation',
        original_formal_and_launcher_natural_exit_zero=True,learning_completed=True,overall_acceptance_passed=False,
        saved_paths=162,controls=365248,independent_smoke_saved_paths=6,formal_saved_paths=156,budget_cap=162,budget_closed=True,
        fresh_zero_parameters1122=True,LSTDQ_completed_solves=2,checkpoint0_1_2_full_theta_state_counts_rng_present=True,
        training_successes=[8,7],training_standard_successes=[True,False],training_invalid_counts=[0,1],
        deterministic_check_successes=[4,4],deterministic_standard_successes=[False,False],deterministic_invalid_counts=[1,0],
        common_eligible=[False,False],selected_checkpoint=None,final_R157_fallback_successes=18,final_standard_success=True,final_invalid=0,
        best='R157 18/24',new_improvement=False,
        original_invalids=[read(FORMAL/'check_0001/case_773009/result.json'),read(FORMAL/'training_0002/case_773011/result.json')],
        original_launcher_a_recording_failure_preserved=True,original_launch_request_AttributeError_preserved=True,
        original_request_a_exit_receipt_missing_not_fabricated=True,original_failure_is_not_training_physics_failure=True,
        independent_proof={k:v for k,v in proof.items() if k!='source_inventory'},
        independent_proof_file=str(PROOF),independent_proof_sha256=PROOF_SHA,independent_verification_exit_sha256=digest(EXIT),
        source_inventory_sha256={str(p):hashlib.sha256(json.dumps(inventory(p),sort_keys=True).encode()).hexdigest() for p in dirs},
        archive_preparation_source_sha256={str(p):digest(p) for p in original_sources()},
        original_before_request_full_IMU_double_gap_preserved=True,first_joint_slew_planning_independent_reconstruction_claim=False,
        original_invalid_labels_500Hz_peaks_RNG_source_arrays_raw_bytes_preserved=True,qpos_qvel_only_isolated_equality=True,
        full_recomputed_matrices_compared_and_byte_hashes_saved_no_duplicate_large_NPZ=True,
        physical_safety_or_cross_implementation_bitwise_claim=False,
        full_task_completed=False,hardware_readiness=False,qualification=False,new_dynamic_attempts=0,
        unread320_not_read=True,protected_live_server_not_operated=True,
        immutable_preparation_archive_complete=False,immutable_preparation_publication_complete=False,
        publication_requires_actual_exit_independent_readonly_and_full_GitHub_terminal_match=True,
        resource_preparation_snapshot=dict(live_all_R203_bytes=live,forecast_bytes=(640+640+128)*1024**2,free_bytes=shutil.disk_usage(ROOT).free))
    write(ROOT/'tmp'/TERMINAL,document)
    print(json.dumps(dict(preparation_saved=True,terminal=str(ROOT/'tmp'/TERMINAL),terminal_sha256=digest(ROOT/'tmp'/TERMINAL),characters=len((ROOT/'tmp'/TERMINAL).read_text()),new_dynamic_attempts=0)),flush=True)
def copy(source,target):
    assert not target.exists();shutil.copy2(source,target);assert digest(source)==digest(target)
def add(p):subprocess.run(['git','-c','gc.auto=0','add','-f','--',str(p.relative_to(REPO))],cwd=REPO,check=True)
def archive():
    idle();assert not RESULT.exists() and CONT.is_dir()
    proof=completed_proof();doc=read(ROOT/'tmp'/TERMINAL)
    assert doc['independent_proof']=={k:v for k,v in proof.items() if k!='source_inventory'} and doc['independent_proof_sha256']==PROOF_SHA
    assert digest(CONT/TERMINAL)==digest(ROOT/'tmp'/TERMINAL)
    assert digest(CONT/'terminal_verification_proof_20261011.json')==PROOF_SHA
    for p,sha in doc['archive_preparation_source_sha256'].items():assert digest(p)==sha
    dirs=[Path(p) for p in proof['source_inventory']]+[VERIFY];before={str(d):inventory(d) for d in dirs}
    for d in dirs:assert hashlib.sha256(json.dumps(before[str(d)],sort_keys=True).encode()).hexdigest()==doc['source_inventory_sha256'][str(d)]
    outputs=[d for d in dirs if d!=NUMSRC]
    assert len(outputs)==8
    for d in outputs:assert not (REPO/'results'/d.name).exists()
    names=original_sources()
    for p in names:assert not (REPO/'scripts'/p.name).exists()
    for name in (TERMINAL,PLAN,PUBLISHER.name):assert not (REPO/('scripts/'+name if name==PUBLISHER.name else name)).exists()
    record=REPO/'results/getup_total_target_lstdq_r203_left_20261011/terminal_evidence'
    tmpfiles=sorted(p for p in (ROOT/'tmp').iterdir() if p.is_file() and 'r203' in p.name.lower() and p!=ARCHIVE_LOG)
    logs=sorted(p for p in (ROOT/'outputs').iterdir() if p.is_file() and 'r203' in p.name.lower())
    external={str(p):digest(p) for p in [*tmpfiles,*logs]};continuation=inventory(CONT)
    live=all_live();forecast=(640+640+128)*1024**2
    assert live+forecast<2*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
    git_before=size(REPO/'.git');print('R203_TERMINAL_ARCHIVE_BEGIN',live,flush=True)
    for d in outputs:
        target=REPO/'results'/d.name/'terminal_snapshot';shutil.copytree(d,target);assert inventory(target)==before[str(d)]
        write(target/'snapshot_metadata.json',dict(original_R203_completed_no_improvement=True,new_dynamic_attempts=0,
            original_formal_natural_exit_zero=True,completed_LSTDQ_solves=2,original_failures_source_gaps_and_invalids_preserved=True,
            full_task_completed=False,hardware_readiness=False))
        add(target)
    target=REPO/'results'/outputs[0].name/'numeric_source_terminal_snapshot'
    shutil.copytree(NUMSRC,target);assert inventory(target)==before[str(NUMSRC)];add(target)
    record.mkdir()
    for p in [*tmpfiles,*logs]:copy(p,record/p.name)
    shutil.copytree(CONT,record/'local_continuation_original_bytes')
    assert inventory(record/'local_continuation_original_bytes')==continuation
    write(record/'artifact_manifest.json',inventory(record));add(record)
    for p in names:target=REPO/'scripts'/p.name;copy(p,target);add(target)
    target=REPO/'scripts'/PUBLISHER.name;copy(PUBLISHER,target);add(target)
    for name in (TERMINAL,PLAN):target=REPO/name;copy(ROOT/'tmp'/name,target);add(target)
    assert before=={str(d):inventory(d) for d in dirs}
    assert all(digest(p)==sha for p,sha in external.items()) and inventory(CONT)==continuation
    assert git('rev-parse','HEAD')==BASE
    staged=git('diff','--cached','--name-only').splitlines()
    allowed=[str((REPO/'results'/d.name).relative_to(REPO))+'/' for d in outputs]
    allowed.extend('scripts/'+p.name for p in names);allowed.extend([TERMINAL,PLAN,'scripts/'+PUBLISHER.name])
    assert staged and all(any(p.startswith(a) if a.endswith('/') else p==a for a in allowed) for p in staged)
    archive_bytes=sum(size(REPO/'results'/d.name) for d in outputs)+sum((REPO/'scripts'/p.name).stat().st_size for p in names)+(REPO/'scripts'/PUBLISHER.name).stat().st_size+sum((REPO/n).stat().st_size for n in (PLAN,TERMINAL))
    assert archive_bytes<640*1024**2 and live+archive_bytes+640*1024**2+128*1024**2<2*1024**3
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve completed R203 LSTD-Q, 162 paths, two solves, full independent audit and R157 fallback'],cwd=REPO,check=True)
    head=git('rev-parse','HEAD');parent=git('rev-parse','HEAD^');assert parent==BASE and not git('status','--porcelain')
    git_increment=size(REPO/'.git')-git_before;live=all_live();total=live+archive_bytes+max(0,git_increment)
    assert git_increment<640*1024**2 and total<2*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
    assert before=={str(d):inventory(d) for d in dirs} and all(digest(p)==sha for p,sha in external.items())
    result=dict(commit=head,parent=parent,repository_clean=True,completed_proof_sha256=PROOF_SHA,
        independent_verifier_not_reexecuted=True,source_hashes_unchanged=True,new_dynamic_attempts=0,
        original_formal_launcher_natural_exit_zero=True,original_recording_failures_source_gaps_invalids_preserved=True,
        saved_paths=162,controls=365248,completed_LSTDQ_solves=2,final_R157_fallback_successes=18,new_improvement=False,
        terminal_sha256=digest(ROOT/'tmp'/TERMINAL),archive_source_sha256=digest(__file__),
        live_bytes=live,archive_bytes=archive_bytes,git_increment_bytes=git_increment,total_bytes=total,free_bytes=shutil.disk_usage(ROOT).free,
        full_task_completed=False,hardware_readiness=False,qualification=False,published=False)
    write(RESULT,result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    prepare() if args.prepare else archive()

