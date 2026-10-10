"""Append-only R200 terminal archive after independent readonly proof; no new dynamic."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
from diagnostics import verify_getup_total_target_c51_terminal_r200b as evidence
ROOT=evidence.ROOT;REPO=evidence.REPO
NAMES=('total_target_c51_kernel_r200.py','train_getup_total_target_c51_r200.py','test_getup_total_target_c51_r200.py','launch_getup_total_target_c51_r200.py','verify_getup_total_target_c51_terminal_r200.py','archive_getup_total_target_c51_terminal_r200.py','verify_getup_total_target_c51_terminal_r200b.py','archive_getup_total_target_c51_terminal_r200b.py')
TMPNAMES=('GETUP_TOTAL_TARGET_C51_PLAN_R200_20261011.json','GETUP_TOTAL_TARGET_C51_TERMINAL_R200_20261011.json','publish_getup_total_target_c51_terminal_r200_20261011.py')
def write(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def add(path):subprocess.run(['git','add','-f',str(path.relative_to(REPO))],cwd=REPO,check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);a=p.parse_args()
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
    evidence.idle(a.base)
    for d in evidence.DIRECTORIES:assert not (REPO/'results'/d.name).exists()
    for name in NAMES:assert not (REPO/'scripts'/name).exists()
    for name in TMPNAMES:assert not (REPO/('scripts/'+name if name.endswith('.py') else name)).exists()
    assert not (ROOT/'tmp/getup_total_target_c51_r200_archive_result_20261011.json').exists()
    proof=evidence.read(evidence.PROOF)
    assert proof['verifier_source_sha256']==evidence.digest(Path(evidence.__file__))
    print('R200_ARCHIVE_RECONSTRUCTION_BEGIN',flush=True)
    recomputed=evidence.verify()
    assert recomputed==proof
    document=evidence.read(ROOT/'tmp/GETUP_TOTAL_TARGET_C51_TERMINAL_R200_20261011.json')
    first_failure=evidence.read(ROOT/'tmp/verification_exit_r200_20261011.json')
    assert first_failure['observed_exit_code']==1 and not first_failure['proof_exists'] and first_failure['new_dynamic']==0
    assert evidence.digest(Path(__file__).with_name('verify_getup_total_target_c51_terminal_r200.py'))==first_failure['verifier_source_sha256']
    assert document['independent_proof']=={k:v for k,v in proof.items() if k not in ('source_inventory','rows')}
    assert document['saved_paths']==162 and document['controls']==358549
    assert document['meta']['selected_R157_fallback'] and not document['meta']['original_development_gate']
    live=proof['live_bytes'];assert 3*live<2*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    gitbefore=sum(f.stat().st_size for f in (REPO/'.git').rglob('*') if f.is_file())
    for d in evidence.DIRECTORIES:
        target=REPO/'results'/d.name/'terminal_snapshot';shutil.copytree(d,target)
        assert evidence.inventory(target)==proof['source_inventory'][d.name]
        log=d.with_suffix('.log') if d in (evidence.run.SMOKE,evidence.run.OUTPUT) else ROOT/'tmp/getup_total_target_c51_r200_launcher_20261011.log' if d==evidence.run.LAUNCH else None
        if log and log.exists():shutil.copy2(log,target/'process.log')
        write(target/'snapshot_metadata.json',dict(terminal_result_saved=True,natural_exit=True,new_dynamic=0,closed_paths=162,full_task_completed=False,hardware_readiness=False,qualification=False))
        write(target/'artifact_manifest.json',evidence.inventory(target));add(target)
    for name in NAMES:
        source=Path(__file__).with_name(name);target=REPO/'scripts'/name;shutil.copy2(source,target)
        assert evidence.digest(source)==evidence.digest(target);add(target)
    for name in TMPNAMES:
        source=ROOT/'tmp'/name;target=REPO/('scripts/'+name if name.endswith('.py') else name);shutil.copy2(source,target)
        assert evidence.digest(source)==evidence.digest(target);add(target)
    for source,target in [
        (ROOT/'tmp/getup_total_target_c51_r200_preflight_exit_20261011.json',REPO/'results'/evidence.PREFLIGHT.name/'preflight_exit_receipt.json'),
        (evidence.PROOF,REPO/'results'/evidence.run.OUTPUT.name/'terminal_verification.json'),
        (ROOT/'tmp/verification_exit_r200b_20261011.json',REPO/'results'/evidence.run.OUTPUT.name/'verification_exit_receipt.json')]:
        assert not target.exists();shutil.copy2(source,target);add(target)
    for source,filename in [
        (ROOT/'tmp/verification_exit_r200_20261011.json','verification_failure_01_exit.json'),
        (ROOT/'tmp/getup_total_target_c51_r200_verification_20261011.log','verification_failure_01.log'),
        (ROOT/'tmp/getup_total_target_c51_r200b_verification_20261011.log','independent_verification.log'),
        (ROOT/'tmp/getup_total_target_c51_r200_continuation_evidence_20261011.json','continuation_evidence.json')]:
        target=REPO/'results'/evidence.run.OUTPUT.name/filename
        assert not target.exists();shutil.copy2(source,target);add(target)
    assert proof['source_inventory']=={d.name:evidence.inventory(d) for d in evidence.DIRECTORIES}
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve finite R200 categorical distributional total-target C51 and all failed checkpoints'],cwd=REPO,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    assert subprocess.check_output(['git','rev-parse','HEAD^'],cwd=REPO,text=True).strip()==a.base
    gitafter=sum(f.stat().st_size for f in (REPO/'.git').rglob('*') if f.is_file())
    archive=sum(f.stat().st_size for d in evidence.DIRECTORIES for f in (REPO/'results'/d.name).rglob('*') if f.is_file())
    total=live+archive+max(0,gitafter-gitbefore);free=shutil.disk_usage(ROOT).free
    assert total<2*1024**3 and free>10*1024**3
    result=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),parent=a.base,repository_clean=True,proof_equal_to_independent_recomputation=True,new_dynamic=0,live_bytes=live,archive_bytes=archive,git_increment_bytes=gitafter-gitbefore,total_bytes=total,free_bytes=free,full_task_completed=False,hardware_readiness=False,qualification=False,published=False)
    write(ROOT/'tmp/getup_total_target_c51_r200_archive_result_20261011.json',result)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()

