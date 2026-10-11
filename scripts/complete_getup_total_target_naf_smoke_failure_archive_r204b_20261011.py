"""One-shot append-only completion of the already staged R204 archive after original path failure.
Does not import or execute original archive, verifier, kernel, learner, environment or dynamics.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
ROOT=Path('/data/shijinsheng/open_duck')
REPO=ROOT/'github/openduck-mini-'
PROJECT=ROOT/'projects/Open_Duck_Playground'
TMP=ROOT/'tmp'
BASE='3b2ae6dc6d745cf02f2ba57c7552251305c1c179'
TERMINAL='GETUP_TOTAL_TARGET_NAF_SMOKE_FAILED_TERMINAL_R204_20261011.json'
PLAN='GETUP_TOTAL_TARGET_NAF_PLAN_R204_20261011.json'
PROOF=TMP/'getup_total_target_naf_r204_smoke_failure_verification_20261011.json'
PROOF_SHA='82ebd45907458619ab17adf53835132cdd383cfffec553643c1e34a27a519c20'
NUMSRC=TMP/'getup_total_target_naf_numeric_r204_20261011'
VERIFY=ROOT/'outputs/getup_total_target_naf_r204_smoke_failure_verification_20261011'
SMOKE=ROOT/'outputs/getup_total_target_naf_r204_smoke_20261011'
RECORD=REPO/'results'/SMOKE.name/'terminal_evidence'
CONT=TMP/'getup_total_target_naf_r204_continuation_20261011'
EXIT_A=TMP/'getup_total_target_naf_r204_smoke_failure_archive_exit_20261011.json'
LOG_A=TMP/'getup_total_target_naf_r204_smoke_failure_archive_20261011.log'
FAILURE=TMP/'smoke_failure_archive_a_failure_record_20261011.json'
RESULT=TMP/'getup_total_target_naf_r204_smoke_failure_archive_completion_b_result_20261011.json'
NEW_NAMES=(
 'complete_getup_total_target_naf_smoke_failure_archive_r204b_20261011.py',
 'smoke_failure_archive_completion_wrapper_r204b_20261011.py',
 'publish_getup_total_target_naf_smoke_failure_r204b_20261011.py',
 'smoke_failure_publication_wrapper_r204b_20261011.py',
 'smoke_failure_publication_readonly_wrapper_r204b_20261011.py')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def inv(p):return {str(f.relative_to(p)):digest(f) for f in sorted(p.rglob('*')) if f.is_file()}
def size(p):return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
def git(*a):return subprocess.check_output(['git','-c','gc.auto=0',*a],cwd=REPO,text=True).strip()
def write(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def add(p):subprocess.run(['git','-c','gc.auto=0','add','-f','--',str(p.relative_to(REPO))],cwd=REPO,check=True)
def copy(p,t):
    assert not t.exists();shutil.copy2(p,t);assert digest(p)==digest(t)
def live():
    files=set()
    for d in (ROOT/'outputs',TMP):
        for p in d.iterdir():
            if 'r204' in p.name.lower():
                if p.is_file():files.add(p)
                elif p.is_dir():files.update(f for f in p.rglob('*') if f.is_file())
    files.update(p for p in (PROJECT/'diagnostics').iterdir() if p.is_file() and 'r204' in p.name.lower())
    files.add(FAILURE)
    return sum(p.stat().st_size for p in files)
def quiet():
    assert (Path('/proc/2664890/cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        =='.venv/bin/python -u -m open_duck_agent.live_server --host 127.0.0.1 --port 18766')
    for s in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines():
        first=Path(s.split()[0]).name if s.split() else ''
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in s and any(t in s for t in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_','verify_getup_'))),s
        assert 'publish_getup' not in s and 'git push' not in s,s
def index_exact(expected):
    status=git('status','--porcelain').splitlines()
    assert len(status)==len(expected) and all(s.startswith('A  ') for s in status)
    assert {s[3:] for s in status}==expected
    assert not git('diff','--name-only')
    entries={}
    for row in subprocess.check_output(['git','ls-files','-s','-z'],cwd=REPO).split(b'\0'):
        if row:
            meta,name=row.split(b'\t',1);mode,oid,stage=meta.split()
            if name.decode() in expected:entries[name.decode()]=(mode,oid,stage)
    assert set(entries)==expected
    for name,(mode,oid,stage) in entries.items():
        data=(REPO/name).read_bytes()
        assert mode in (b'100644',b'100755') and stage==b'0'
        assert (mode==b'100755')==bool((REPO/name).stat().st_mode & 0o111)
        assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest().encode()==oid,name
def main():
    quiet();assert git('rev-parse','HEAD')==BASE and not RESULT.exists()
    status=git('status','--porcelain')
    assert len(status.splitlines())==10105 and hashlib.sha256(status.encode()).hexdigest()=='5223fe812c886c065af526f9007618a90336117024da169b5fe64811a7b45420'
    assert digest(EXIT_A)=='1f8a0a2f956a662dc3328f7b5d34b9b92193a0d2eea5c94df2e0bed9be00ca8f'
    assert digest(LOG_A)=='8b59b116a192c4f71e28ac178044fcd53ade7899ab648a424aa4a3eac0aadc89'
    e=read(EXIT_A);assert e['actual_subprocess_return_observed'] and e['natural_exit'] and e['exit_code']==1 and e['source_hash_unchanged'] and not e['archive_result_saved']
    assert digest(PROOF)==PROOF_SHA
    proof=read(PROOF);doc=read(TMP/TERMINAL)
    assert proof['audit_complete'] and proof['closed_saved_paths']==6 and proof['controls']==11441
    assert doc['independent_proof']=={k:v for k,v in proof.items() if k!='source_inventory'}
    assert digest(TMP/TERMINAL)==digest(REPO/TERMINAL)==digest(CONT/TERMINAL)=='761cb6a82e59c70ac9e21da2cf7a184a247653c8c5fd16733401b17385b6df60'
    assert not doc['smoke_physical_gate_passed'] and doc['formal_not_run'] and doc['production_joint_Q_updates']==0 and doc['unexecuted_formal156_closed']
    assert not (ROOT/'outputs/getup_total_target_naf_r204_left_20261011').exists()
    dirs=[Path(p) for p in proof['source_inventory']]+[VERIFY]
    before={str(d):inv(d) for d in dirs}
    for d in dirs:
        assert hashlib.sha256(json.dumps(before[str(d)],sort_keys=True).encode()).hexdigest()==doc['source_inventory_sha256'][str(d)]
        if str(d) in proof['source_inventory']:assert before[str(d)]==proof['source_inventory'][str(d)]
    outputs=[d for d in dirs if d!=NUMSRC]
    expected={TERMINAL}
    for d in outputs:
        t=REPO/'results'/d.name/'terminal_snapshot';stored=inv(t);meta=stored.pop('snapshot_metadata.json')
        assert stored==before[str(d)]
        assert read(t/'snapshot_metadata.json')['original_failures_source_gaps_missing_raw_checkpoint_and_invalids_preserved']
        expected.update(str(p.relative_to(REPO)) for p in t.rglob('*') if p.is_file())
    numericcopy=REPO/'results'/outputs[0].name/'numeric_source_terminal_snapshot'
    assert inv(numericcopy)==before[str(NUMSRC)]
    expected.update(str(p.relative_to(REPO)) for p in numericcopy.rglob('*') if p.is_file())
    artifact=read(RECORD/'artifact_manifest.json')
    stored=inv(RECORD);stored.pop('artifact_manifest.json');assert stored==artifact
    for name,sha in artifact.items():
        if '/' not in name:
            orig=TMP/name if (TMP/name).is_file() else ROOT/'outputs'/name
            assert digest(orig)==sha,name
    assets=read(RECORD/'external_original_assets/manifest.json')
    assert {p:v['sha256'] for p,v in assets.items()}==doc['external_original_XML_mesh_ONNX_reference_sha256']
    for p,v in assets.items():assert digest(p)==v['sha256']==digest(RECORD/'external_original_assets'/v['copy'])
    continuation=inv(CONT);assert inv(RECORD/'local_continuation_original_bytes')==continuation and len(continuation)==82
    expected.update(str(p.relative_to(REPO)) for p in RECORD.rglob('*') if p.is_file())
    for p,sha in doc['archive_preparation_source_sha256'].items():
        assert digest(p)==sha==digest(REPO/'scripts'/Path(p).name)
        expected.add('scripts/'+Path(p).name)
    oldpublisher=TMP/'publish_getup_total_target_naf_smoke_failure_r204_20261011.py'
    assert digest(oldpublisher)==digest(REPO/'scripts'/oldpublisher.name)
    expected.add('scripts/'+oldpublisher.name)
    index_exact(expected)
    assert not (REPO/PLAN).exists()
    assert digest(NUMSRC/PLAN)=='7265ba9ab27334a0ec36f3c36f0e943b65ce574816f40e353a57eecfd1aa6db2'
    assert shutil.disk_usage(ROOT).free>=12*1024**3
    archive_before=sum((REPO/p).stat().st_size for p in expected)
    assert live()+archive_before+640*1024**2+128*1024**2+8*1024**2<2*1024**3
    print('R204_ARCHIVE_COMPLETION_B_EXISTING_10105_STAGED_FILES_EXACT',flush=True)
    completion=RECORD/'archive_completion_b';completion.mkdir()
    for p in (EXIT_A,LOG_A,FAILURE):
        t=completion/p.name;copy(p,t);add(t);expected.add(str(t.relative_to(REPO)))
    newsha={}
    for name in NEW_NAMES:
        p=TMP/name;t=REPO/'scripts'/name;copy(p,t);add(t);expected.add(str(t.relative_to(REPO)));newsha[str(p)]=digest(p)
    t=REPO/PLAN;copy(NUMSRC/PLAN,t);add(t);expected.add(PLAN)
    manifest=dict(original_archive_exit_code=1,original_archive_source_immutable=True,original_failure_receipt_sha256=digest(EXIT_A),
        original_failure_full_log_sha256=digest(LOG_A),original_partial_archive_not_reexecuted=True,
        completion_only_missing_PLAN_and_append_failure_completion_sources=True,existing_staged_files_exact=10105,
        original_prepared_terminal_unchanged=True,completed_proof_sha256=PROOF_SHA,new_sources=newsha,
        new_files={str((REPO/p).relative_to(REPO)):digest(REPO/p) for p in expected if p not in {s[3:] for s in status.splitlines()}},
        new_dynamic_attempts=0,old_verifier_not_reexecuted=True,original_missing_raw_checkpoint_invalid_and_closed156_preserved=True)
    m=completion/'completion_manifest.json';write(m,manifest);add(m);expected.add(str(m.relative_to(REPO)))
    index_exact(expected);quiet();assert git('rev-parse','HEAD')==BASE
    assert before=={str(d):inv(d) for d in dirs}
    assert all(digest(p)==sha for p,sha in doc['archive_preparation_source_sha256'].items())
    assert all(digest(p)==v['sha256']==digest(RECORD/'external_original_assets'/v['copy']) for p,v in assets.items())
    archive_bytes=sum((REPO/p).stat().st_size for p in expected)
    assert archive_bytes<640*1024**2 and live()+archive_bytes+(640+128)*1024**2<2*1024**3
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R204 NAF smoke failure and independent audit; retain archive path failure and append-only completion'],cwd=REPO,check=True)
    head=git('rev-parse','HEAD');assert git('rev-parse','HEAD^')==BASE and not git('status','--porcelain')
    objects=[s.split()[0] for s in git('rev-list','--objects',BASE+'..'+head).splitlines()]
    loose=[REPO/'.git/objects'/s[:2]/s[2:] for s in objects]
    assert all(p.is_file() for p in loose)
    git_new_object_bytes=sum(p.stat().st_size for p in loose)
    git_accounted=git_new_object_bytes+(REPO/'.git/index').stat().st_size+1024**2
    total=live()+archive_bytes+git_accounted
    assert git_accounted<640*1024**2 and total<2*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
    assert before=={str(d):inv(d) for d in dirs}
    result=dict(commit=head,parent=BASE,repository_clean=True,archive_completion_b_success=True,
        original_archive_a_exit_code=1,original_failure_receipt_sha256=digest(EXIT_A),original_failure_log_sha256=digest(LOG_A),
        original_archive_not_reexecuted=True,completed_proof_sha256=PROOF_SHA,independent_verifier_not_reexecuted=True,
        source_hashes_unchanged=True,new_dynamic_attempts=0,smoke_physical_gate_passed=False,formal_not_run=True,
        production_joint_Q_updates=0,saved_paths=6,controls=11441,unexecuted_formal156_closed=True,
        original_source_gaps_missing_raw_checkpoint_invalids_preserved=True,new_improvement=False,
        local_continuation_original_bytes_count=82,local_continuation_sha256=continuation,
        external_original_asset_copies_count=len(assets),external_original_asset_copies_SHA_exact=True,
        terminal_sha256=digest(TMP/TERMINAL),completion_source_sha256=digest(__file__),new_sources_sha256=newsha,
        live_bytes=live(),archive_bytes=archive_bytes,git_new_reachable_loose_object_bytes=git_new_object_bytes,
        git_conservative_accounted_bytes=git_accounted,git_accounting='new reachable loose objects plus entire current index and one MiB ref-log overhead bound; not historical pre-stage delta',
        total_conservatively_accounted_bytes=total,free_bytes=shutil.disk_usage(ROOT).free,
        full_task_completed=False,hardware_readiness=False,qualification=False,published=False)
    write(RESULT,result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
