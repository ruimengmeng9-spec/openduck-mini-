"""Append-only R202 user-interruption archive; binds completed proof without reexecution."""
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
BASE='9b4edd92c86f0c689f15f904c0298cae52f17b82'
PROOF=ROOT/'tmp/getup_total_target_vmpo_r202_interruption_verification_20261011.json'
PROOF_SHA='f10e540779ed4366a719b50176f34883b4fe3a05df203c076633484ea37d51d6'
EXIT=ROOT/'tmp/getup_total_target_vmpo_r202_interruption_verification_exit_20261011.json'
VERIFY=ROOT/'outputs/getup_total_target_vmpo_r202_interruption_verification_20261011'
CONT=ROOT/'tmp/getup_total_target_vmpo_r202_interruption_continuation_20261011'
TERMINAL='GETUP_TOTAL_TARGET_VMPO_INTERRUPTED_TERMINAL_R202_20261011.json'
PLAN='GETUP_TOTAL_TARGET_VMPO_PLAN_R202_20261011.json'
RESULT=ROOT/'tmp/getup_total_target_vmpo_r202_interruption_archive_result_20261011.json'
ARCHIVE_LOG=ROOT/'tmp/getup_total_target_vmpo_r202_interruption_archive_20261011.log'
PUBLISHER=ROOT/'tmp/publish_getup_total_target_vmpo_interruption_r202_20261011.py'
NUMSRC=ROOT/'tmp/getup_total_target_vmpo_numeric_r202_20261011'
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
    proof=read(PROOF);exit=read(EXIT)
    assert proof['audit_complete'] and proof['closed_saved_paths']==62 and proof['controls']==139323 and proof['new_dynamic_attempts']==0
    assert proof['original_formal_interrupted_by_user_authorization'] and not proof['formal_natural_exit_zero']
    assert proof['original_missing_MC_actor_value_and_final_not_filled']
    assert exit['natural_exit'] and exit['exit_code']==0 and exit['source_hash_unchanged'] and exit['proof_saved']
    assert exit['source_sha256']==proof['verifier_source_sha256']==digest(PROJECT/'diagnostics/verify_getup_total_target_vmpo_interruption_r202.py')
    for p,items in proof['source_inventory'].items():assert inventory(Path(p))==items,p
    for p,count in proof['source_manifest_counts']:
        p=Path(p)
        if p.name.startswith('worker_source_manifest_'):
            folder=p.with_name(p.name.replace('worker_source_manifest_','worker_executed_sources_').replace('.json',''))
            manifest=read(p)
        elif (p/'source_manifest.json').exists():
            manifest=read(p/'source_manifest.json');folder=p/'executed_sources'
        else:
            manifest=read(p/'contract.json')['sources'];folder=p/'executed_sources'
        assert len(manifest)==count;check_sources(manifest,folder)
    check_sources(read(VERIFY/'source_manifest.json'),VERIFY/'executed_sources')
    assert digest(VERIFY/'independent_saved_reward_MC_reference.npz')==proof['initial_state_and_independent_reference_MC']['independent_reference_MC_sha256']
    return proof
def original_sources():
    paths=sorted(p for p in (PROJECT/'diagnostics').iterdir() if p.is_file() and 'r202' in p.name.lower() and p.suffix=='.py')
    assert Path(__file__) in paths
    return paths
def all_live():
    files=set()
    for directory in (ROOT/'outputs',ROOT/'tmp'):
        for p in directory.iterdir():
            if 'r202' in p.name.lower():
                if p.is_file():files.add(p)
                elif p.is_dir():files.update(f for f in p.rglob('*') if f.is_file())
    files.update(original_sources())
    return sum(f.stat().st_size for f in files)
def prepare():
    idle();assert not (ROOT/'tmp'/TERMINAL).exists() and not RESULT.exists() and not CONT.exists()
    proof=completed_proof()
    dirs=[Path(p) for p in proof['source_inventory']]+[VERIFY]
    live=all_live();assert live+(640+640+128)*1024**2<2*1024**3
    assert shutil.disk_usage(ROOT).free>=10*1024**3
    missing=['checkpoint_0001.npz','checkpoint_0001.learner.msgpack','checkpoint_0002.npz',
        'training_0001/complete_original_MC_batch.npz','training_0001/learner_rng_before.json',
        'training_0001/actor_full_decisions.npz','training_0001/actor_update.json',
        'training_0001/value_full_updates.npz','training_0001/value_update.json',
        'check_0001','training_0002','check_0002','candidate','results.json','training_closed_history.json','input_hashes_after.json']
    formal=ROOT/'outputs/getup_total_target_vmpo_r202_left_20261011'
    assert all(not (formal/p).exists() for p in missing)
    document=dict(date='2026-10-11',terminal=True,stage='R202_user_interrupted_completed_saved_decision_audit_archive_preparation',
        original_formal_user_authorized_interruption=True,original_formal_natural_exit_zero=False,
        original_launcher_natural_exit_receipt_fabricated=False,overall_training_success=False,
        saved_paths=62,controls=139323,independent_smoke_saved_paths=6,formal_saved_paths=56,
        budget_cap=162,remaining100_budget_closed=True,best='R157 18/24',
        new_candidate=False,R202_final_or_fallback_exists=False,
        persisted_actor_value_updates=[0,0],unobserved_in_memory_progress_not_claimed=True,
        missing_original_evidence_not_filled=missing,
        original_numeric_plan_path_failure_preserved=True,original_launcher_initial_formal_14pass2skip_each_preserved=True,
        corrected_gate16pass0skip_not_retroactive_formal_permission=True,
        original_invalid_769002=read(formal/'training_0001/case_769002/result.json'),
        independent_proof={k:v for k,v in proof.items() if k!='source_inventory'},
        independent_proof_file=str(PROOF),independent_proof_sha256=PROOF_SHA,
        independent_verification_exit_sha256=digest(EXIT),
        source_inventory_sha256={str(p):hashlib.sha256(json.dumps(inventory(p),sort_keys=True).encode()).hexdigest() for p in dirs},
        archive_preparation_source_sha256={str(p):digest(p) for p in original_sources()},
        reference_MC_is_new_audit_reference_not_original_missing_trace=True,
        original_before_request_full_IMU_double_gap_preserved=True,
        first_joint_slew_planning_independent_reconstruction_claim=False,
        original_failure_source_labels_peaks_RNG_raw_bytes_preserved=True,
        qpos_qvel_only_isolated_equality=True,
        physical_safety_or_cross_implementation_bitwise_claim=False,
        full_task_completed=False,hardware_readiness=False,qualification=False,new_dynamic_attempts=0,
        unread320_not_read=True,protected_live_server_not_operated=True,
        immutable_preparation_archive_complete=False,immutable_preparation_publication_complete=False,
        publication_requires_actual_exit_independent_readonly_and_full_GitHub_terminal_match=True,
        resource_preparation_snapshot=dict(live_all_R202_bytes=live,forecast_bytes=(640+640+128)*1024**2,free_bytes=shutil.disk_usage(ROOT).free))
    write(ROOT/'tmp'/TERMINAL,document)
    print(json.dumps(dict(preparation_saved=True,terminal=str(ROOT/'tmp'/TERMINAL),terminal_sha256=digest(ROOT/'tmp'/TERMINAL),characters=len((ROOT/'tmp'/TERMINAL).read_text()),new_dynamic_attempts=0)),flush=True)
def copy(source,target):
    assert not target.exists();shutil.copy2(source,target);assert digest(source)==digest(target)
def add(p):subprocess.run(['git','-c','gc.auto=0','add','-f','--',str(p.relative_to(REPO))],cwd=REPO,check=True)
def archive():
    idle();assert not RESULT.exists() and CONT.is_dir()
    proof=completed_proof();doc=read(ROOT/'tmp'/TERMINAL)
    assert doc['independent_proof']=={k:v for k,v in proof.items() if k!='source_inventory'}
    assert doc['independent_proof_sha256']==PROOF_SHA
    assert digest(CONT/TERMINAL)==digest(ROOT/'tmp'/TERMINAL)
    assert digest(CONT/'interruption_verification_proof_20261011.json')==PROOF_SHA
    for p,sha in doc['archive_preparation_source_sha256'].items():assert digest(p)==sha
    dirs=[Path(p) for p in proof['source_inventory']]+[VERIFY]
    before={str(d):inventory(d) for d in dirs}
    for d in dirs:assert hashlib.sha256(json.dumps(before[str(d)],sort_keys=True).encode()).hexdigest()==doc['source_inventory_sha256'][str(d)]
    outputs=[d for d in dirs if d!=NUMSRC]
    for d in outputs:assert not (REPO/'results'/d.name).exists()
    names=original_sources()
    for p in names:assert not (REPO/'scripts'/p.name).exists()
    for name in (TERMINAL,PLAN,PUBLISHER.name):assert not (REPO/('scripts/'+name if name==PUBLISHER.name else name)).exists()
    record=REPO/'results/getup_total_target_vmpo_r202_left_20261011/interruption_evidence'
    tmpfiles=sorted(p for p in (ROOT/'tmp').iterdir() if p.is_file() and 'r202' in p.name.lower() and p!=ARCHIVE_LOG)
    logs=sorted(p for p in (ROOT/'outputs').iterdir() if p.is_file() and 'r202' in p.name.lower())
    external={str(p):digest(p) for p in [*tmpfiles,*logs]}
    continuation=inventory(CONT)
    live=all_live();forecast=(640+640+128)*1024**2
    assert live+forecast<2*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
    git_before=size(REPO/'.git');print('R202_INTERRUPTION_ARCHIVE_BEGIN',live,flush=True)
    for d in outputs:
        target=REPO/'results'/d.name/'terminal_snapshot';shutil.copytree(d,target);assert inventory(target)==before[str(d)]
        write(target/'snapshot_metadata.json',dict(original_R202_interruption_preserved=True,new_dynamic_attempts=0,
            original_formal_natural_exit_zero=False,completed_persisted_actor_value_updates=[0,0],
            missing_original_learning_and_final_evidence_not_recreated=True,full_task_completed=False,hardware_readiness=False))
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
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve user-interrupted R202 V-MPO, 62 saved paths, gate repair and independent saved-decision audit'],cwd=REPO,check=True)
    head=git('rev-parse','HEAD');parent=git('rev-parse','HEAD^');assert parent==BASE and not git('status','--porcelain')
    git_increment=size(REPO/'.git')-git_before;live=all_live();total=live+archive_bytes+max(0,git_increment)
    assert git_increment<640*1024**2 and total<2*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
    assert before=={str(d):inventory(d) for d in dirs} and all(digest(p)==sha for p,sha in external.items())
    result=dict(commit=head,parent=parent,repository_clean=True,completed_proof_sha256=PROOF_SHA,
        independent_verifier_not_reexecuted=True,source_hashes_unchanged=True,new_dynamic_attempts=0,
        original_formal_user_interrupted_not_natural_exit_zero=True,original_skips_invalid_and_missing_evidence_preserved=True,
        saved_paths=62,controls=139323,completed_persisted_actor_value_updates=[0,0],
        terminal_sha256=digest(ROOT/'tmp'/TERMINAL),archive_source_sha256=digest(__file__),
        live_bytes=live,archive_bytes=archive_bytes,git_increment_bytes=git_increment,total_bytes=total,free_bytes=shutil.disk_usage(ROOT).free,
        full_task_completed=False,hardware_readiness=False,qualification=False,published=False)
    write(RESULT,result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    prepare() if args.prepare else archive()

