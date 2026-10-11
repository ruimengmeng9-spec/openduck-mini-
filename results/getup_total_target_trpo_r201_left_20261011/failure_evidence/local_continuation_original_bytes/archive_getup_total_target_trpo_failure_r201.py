"""Append-only R201 failure archival bound to the already completed independent proof."""
import json
import os
from pathlib import Path
import shutil
import subprocess
from diagnostics import verify_getup_total_target_trpo_failure_r201b as evidence
ROOT=evidence.ROOT;REPO=evidence.REPO;BASE=evidence.BASE
A=ROOT/'outputs/getup_total_target_trpo_r201_failure_verification_20261011'
B=evidence.DEST
CONT=ROOT/'tmp/getup_total_target_trpo_r201_failure_continuation_20261011'
RESULT=ROOT/'tmp/getup_total_target_trpo_r201_failure_archive_result_20261011.json'
TERMINAL='GETUP_TOTAL_TARGET_TRPO_FAILURE_TERMINAL_R201_20261011.json'
PLAN='GETUP_TOTAL_TARGET_TRPO_PLAN_R201_20261011.json'
NAMES=('total_target_trpo_kernel_r201.py','train_getup_total_target_trpo_r201.py','test_getup_total_target_trpo_r201.py','launch_getup_total_target_trpo_r201.py','verify_getup_total_target_trpo_failure_r201.py','verify_getup_total_target_trpo_failure_r201b.py','archive_getup_total_target_trpo_failure_r201.py')
TMPFILES=('getup_total_target_trpo_r201_preflight_exit_20261011.json','getup_total_target_trpo_r201_failure_verification_exit_20261011.json','getup_total_target_trpo_r201b_failure_verification_exit_20261011.json','getup_total_target_trpo_r201_failure_verification_20261011.log','getup_total_target_trpo_r201b_failure_verification_20261011.log','getup_total_target_trpo_r201_failure_verification_wrapper_20261011.py','getup_total_target_trpo_r201b_failure_verification_wrapper_20261011.py','getup_total_target_trpo_r201_interface_preparation_20261011.json','publish_getup_total_target_trpo_failure_r201_20261011.py','getup_total_target_trpo_r201_failure_archive_wrapper_20261011.py')
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def add(p):subprocess.run(['git','add','-f',str(p.relative_to(REPO))],cwd=REPO,check=True)
def size(d):return sum(f.stat().st_size for f in d.rglob('*') if f.is_file())
def copy(source,target):
    assert not target.exists();shutil.copy2(source,target);assert evidence.digest(source)==evidence.digest(target)
def main():
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6]);evidence.idle();assert not RESULT.exists()
    proof=evidence.read(evidence.PROOF);assert evidence.digest(evidence.PROOF)=='6b10b1a29ad279a29e6f3298d2b80a49682eba0f5b05fe86b039e4f73d897b14'
    assert proof['audit_complete'] and proof['closed_saved_paths']==112 and proof['controls']==251297 and proof['new_dynamic_attempts']==0
    assert proof['verifier_source_sha256']==evidence.digest(evidence.__file__)
    receipt=evidence.read(ROOT/'tmp/getup_total_target_trpo_r201b_failure_verification_exit_20261011.json')
    assert receipt['natural_exit'] and receipt['exit_code']==0 and receipt['source_hash_unchanged'] and receipt['proof_saved']
    assert receipt['source_sha256']==proof['verifier_source_sha256']
    first=evidence.read(ROOT/'tmp/getup_total_target_trpo_r201_failure_verification_exit_20261011.json')
    assert first['natural_exit'] and first['exit_code']==1 and not first['proof_saved'] and first['new_dynamic_attempts']==0
    assert first['source_sha256']==evidence.digest(Path(__file__).with_name('verify_getup_total_target_trpo_failure_r201.py'))
    document=evidence.read(ROOT/'tmp'/TERMINAL)
    assert document['independent_proof']=={k:v for k,v in proof.items() if k!='source_inventory'}
    # No repeated learner/scalar verifier invocation: exact original inventories and source hashes bind the completed proof.
    for p,items in proof['source_inventory'].items():assert evidence.inventory(Path(p))==items,p
    for folder in (evidence.run.LAUNCH,evidence.run.SMOKE,evidence.run.OUTPUT):
        c=evidence.read(folder/'contract.json');evidence.sources(c['sources'],folder/'executed_sources')
        for p,sha in c.get('model_files',{}).items():assert evidence.digest(p)==sha
        for p,sha in c.get('hashes',{}).items():assert evidence.digest(p)==sha
    for folder in (A,B):evidence.sources(evidence.read(folder/'source_manifest.json'),folder/'executed_sources')
    dirs=(*evidence.DIRECTORIES,A,B);before={str(d):evidence.inventory(d) for d in dirs}
    outputs=[d for d in dirs if d!=evidence.run.NUMERIC_SOURCE]
    for d in outputs:assert not (REPO/'results'/d.name).exists()
    for name in NAMES:assert not (REPO/'scripts'/name).exists()
    for name in (TERMINAL,PLAN):assert not (REPO/name).exists()
    assert not (REPO/'scripts/publish_getup_total_target_trpo_failure_r201_20261011.py').exists()
    live=sum(size(d) for d in dirs)+size(CONT)+evidence.PROOF.stat().st_size
    live+=sum((ROOT/'tmp'/n).stat().st_size for n in TMPFILES)+sum(Path(__file__).with_name(n).stat().st_size for n in NAMES)
    logs=[Path(str(evidence.run.SMOKE)+'.log'),Path(str(evidence.run.OUTPUT)+'.log'),ROOT/'tmp/getup_total_target_trpo_r201_launcher_20261011.log']
    live+=sum(p.stat().st_size for p in logs)+sum((ROOT/'tmp'/n).stat().st_size for n in (PLAN,TERMINAL))
    assert 3*live<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    git_before=size(REPO/'.git');print('R201_FAILURE_ARCHIVE_BEGIN',live,flush=True)
    for d in outputs:
        target=REPO/'results'/d.name/'terminal_snapshot';shutil.copytree(d,target);assert evidence.inventory(target)==before[str(d)]
        write(target/'snapshot_metadata.json',dict(saved_failure_stage=True,new_dynamic_attempts=0,overall_training_success=False,completed_actor_updates=1,completed_value_updates=64,original_failure_and_missing_trace_preserved=True,full_task_completed=False,hardware_readiness=False))
        add(target)
    numeric_target=REPO/'results'/evidence.run.NUMERIC.name/'numeric_source_terminal_snapshot';shutil.copytree(evidence.run.NUMERIC_SOURCE,numeric_target);assert evidence.inventory(numeric_target)==before[str(evidence.run.NUMERIC_SOURCE)];add(numeric_target)
    record=REPO/'results'/evidence.run.OUTPUT.name/'failure_evidence';record.mkdir()
    for source in [evidence.PROOF,*[ROOT/'tmp'/n for n in TMPFILES],*logs]:copy(source,record/source.name)
    shutil.copytree(CONT,record/'local_continuation_original_bytes');assert evidence.inventory(record/'local_continuation_original_bytes')==evidence.inventory(CONT)
    write(record/'artifact_manifest.json',evidence.inventory(record));add(record)
    for name in NAMES:target=REPO/'scripts'/name;copy(Path(__file__).with_name(name),target);add(target)
    publisher=REPO/'scripts/publish_getup_total_target_trpo_failure_r201_20261011.py';copy(ROOT/'tmp'/publisher.name,publisher);add(publisher)
    for name in (PLAN,TERMINAL):target=REPO/name;copy(ROOT/'tmp'/name,target);add(target)
    assert before=={str(d):evidence.inventory(d) for d in dirs}
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==BASE
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve finite R201 trust-region formal failure, 112 saved paths and independent completed-update audit'],cwd=REPO,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();parent=subprocess.check_output(['git','rev-parse','HEAD^'],cwd=REPO,text=True).strip();assert parent==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    archive=sum(size(REPO/'results'/d.name) for d in outputs)+sum((REPO/'scripts'/n).stat().st_size for n in NAMES)+(REPO/'scripts'/publisher.name).stat().st_size+sum((REPO/n).stat().st_size for n in (PLAN,TERMINAL))
    git_increment=size(REPO/'.git')-git_before;free=shutil.disk_usage(ROOT).free;total=live+archive+max(0,git_increment)
    assert total<2*1024**3 and free>10*1024**3
    result=dict(commit=head,parent=parent,repository_clean=True,completed_independent_proof_bound_by_hash_and_full_inventory=True,independent_verifier_not_reexecuted=True,source_hashes_unchanged=True,new_dynamic_attempts=0,saved_paths=112,controls=251297,completed_actor_updates=1,completed_value_updates=64,live_bytes=live,archive_bytes=archive,git_increment_bytes=git_increment,total_bytes=total,free_bytes=free,full_task_completed=False,hardware_readiness=False,qualification=False,published=False)
    write(RESULT,result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
