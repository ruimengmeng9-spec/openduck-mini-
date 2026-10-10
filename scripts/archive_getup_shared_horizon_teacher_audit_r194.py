"""Append-only R194 terminal verification and archive; no old dynamics copied."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
import numpy as np
from diagnostics import audit_getup_shared_horizon_teacher_terminal_r194 as audit
ROOT=audit.ROOT
REPO=ROOT/'github/openduck-mini-'
L=ROOT/'outputs/getup_shared_horizon_teacher_audit_launcher_r194_20261010'
PROOF=ROOT/'tmp/getup_shared_horizon_teacher_audit_r194_terminal_verification_20261010.json'
def inventory(root):return {str(p.relative_to(root)):audit.digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
def idle(base):
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()==base
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines():
        first=Path(line.split()[0]).name if line.split() else ''
        if 'python' not in first and first!='git':continue
        assert not (' -m diagnostics.' in line and any(w in line for w in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_')))
        assert not ('publish_getup' in line or 'git push' in line)
def sources(manifest,folder):
    for p,v in manifest.items():assert audit.digest(p)==v['sha256']==audit.digest(folder/v['copy'])
def verify():
    launch=audit.read(L/'result.json')
    assert launch['natural_exit'] and launch['smoke_exit_code']==launch['formal_exit_code']==0 and launch['new_dynamic_trajectories']==0 and launch['source_hashes_unchanged']
    formal=audit.read(audit.OUTPUT/'results.json');smoke=audit.read(audit.SMOKE/'results.json')
    for root,result,n in ((audit.OUTPUT,formal,187),(audit.SMOKE,smoke,6)):
        assert result['terminal_result_saved'] and result['all_scalar_and_limits_bitwise'] and result['source_evidence_unchanged']
        assert result['existing_trajectories_audited']==n==len(result['rows']) and len(result['terminal_pairs'])==25
        assert result['R157_old_field_parity_count']==87 and result['new_dynamic_trajectories']==0 and not result['original_IMU_double_request_reconstructed']
        assert not any(result[k] for k in ('full_task_completed','hardware_readiness','qualification'))
        assert all(r['teacher_and_original_limits_independently_bitwise'] and r['original_right_hip_feedback_bitwise'] and not r['original_double_IMU_request_reconstructed'] for r in result['rows'])
        tracked=audit.read(root/'source_hashes.json');assert tracked['unchanged'] and tracked['compiled_model_unchanged'] and tracked['before']==tracked['after']
        for p,sha in tracked['before'].items():assert audit.digest(p)==sha
        sources(audit.read(root/'executed_sources_manifest.json'),root/'executed_sources')
    assert formal['controls_audited']==418014 and smoke['controls_audited']==11440
    assert formal['independent_smoke_matches'] and formal['terminal_pairs']==smoke['terminal_pairs']
    assert formal['common_selection']==smoke['common_selection']==audit.common_selection()
    assert formal['R157_old_field_parity_count']==audit.old_field_equalities(audit.jobs())
    assert formal['terminal_pairs']==audit.terminal_pairing()
    for row in smoke['rows']:
        assert row==next(r for r in formal['rows'] if r['identity']==row['identity'])
        identity=row['identity']
        with np.load(audit.OUTPUT/'signals'/identity/'signals.npz',allow_pickle=False) as x,np.load(audit.SMOKE/'signals'/identity/'signals.npz',allow_pickle=False) as y:
            assert x.files==y.files
            for key in x.files:np.testing.assert_array_equal(x[key],y[key])
    invalid=[r for r in formal['rows'] if not r['valid']]
    assert len(invalid)==4 and sorted(r['controls'] for r in invalid)==[45,304,304,304]
    for row in invalid:assert (audit.OUTPUT/'signals'/row['identity']/'signals.npz').is_file()
    for manifest in (L/'executed_sources_manifest.json',L/'initial/test_source_manifest.json',L/'formal/test_source_manifest.json'):
        sources(audit.read(manifest),manifest.parent/'executed_sources')
    for label in ('initial','formal'):
        log=(L/label/'regression.log').read_text();assert 'Ran 12 tests' in log and '\nOK\n' in log
        assert audit.read(L/label/'result.json')==dict(exit_code=0,natural_exit=True,new_dynamic_trajectories=0)
    live=audit.budget()
    return dict(terminal_result_saved=True,natural_exit=True,formal_existing_trajectories=187,independent_existing_trajectories=6,formal_controls=418014,independent_controls=11440,new_dynamic_trajectories=0,original_IMU_double_request_reconstructed=False,full_source_evidence_unchanged=True,independent_smoke_bitwise_equal=True,terminal_pairs=25,R157_old_fields_parity=87,selection=formal['common_selection'],invalid=invalid,live_bytes=live,source_inventory={r.name:inventory(r) for r in (audit.SMOKE,audit.OUTPUT,L)},full_task_completed=False,hardware_readiness=False,qualification=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--verify',action='store_true');a=p.parse_args()
    idle(a.base);proof=verify()
    if a.verify:
        audit.write(PROOF,proof)
        print(json.dumps({k:v for k,v in proof.items() if k!='source_inventory'},indent=2));return
    assert audit.read(PROOF)==proof
    before_git=sum(p.stat().st_size for p in (REPO/'.git').rglob('*') if p.is_file())
    for root in (audit.SMOKE,audit.OUTPUT,L):
        target=REPO/'results'/root.name/'terminal_snapshot';assert not target.exists()
        shutil.copytree(root,target);assert inventory(root)==proof['source_inventory'][root.name]==inventory(target)
        log=ROOT/'tmp/getup_shared_horizon_teacher_audit_r194_launcher_20261010.log' if root==L else root.with_suffix('.log')
        shutil.copy2(log,target/'process.log')
        audit.write(target/'snapshot_metadata.json',dict(terminal_result_saved=True,natural_exit=True,read_only=True,new_dynamic_trajectories=0,historical_gaps_recovered=False,full_task_completed=False,hardware_readiness=False,qualification=False))
        audit.write(target/'artifact_manifest.json',inventory(target))
        subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    target=REPO/'results'/audit.OUTPUT.name/'terminal_verification.json';assert not target.exists();shutil.copy2(PROOF,target)
    subprocess.run(['git','add','-f',str(target.relative_to(REPO))],cwd=REPO,check=True)
    for name in ('audit_getup_shared_horizon_teacher_terminal_r194.py','test_getup_shared_horizon_teacher_audit_r194.py','launch_getup_shared_horizon_teacher_audit_r194.py',Path(__file__).name):
        target=REPO/'scripts'/name;assert not target.exists();shutil.copy2(Path(__file__).with_name(name),target)
        assert audit.digest(target)==audit.digest(Path(__file__).with_name(name))
        subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
    wrapper='publish_getup_shared_horizon_teacher_audit_r194_20261010.py'
    target=REPO/'scripts'/wrapper;assert not target.exists();shutil.copy2(ROOT/'tmp'/wrapper,target)
    subprocess.run(['git','add',str(target.relative_to(REPO))],cwd=REPO,check=True)
    for name in ('GETUP_SHARED_HORIZON_TEACHER_AUDIT_PLAN_R194_20261010.md','GETUP_SHARED_HORIZON_TEACHER_AUDIT_R194_20261010.md'):
        target=REPO/name;assert not target.exists();shutil.copy2(ROOT/'tmp'/name,target)
        subprocess.run(['git','add',name],cwd=REPO,check=True)
    subprocess.run(['git','-c','gc.auto=0','commit','--quiet','-m','Preserve R194 independent closed R193 teacher scalar selection and target audit'],cwd=REPO,check=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    after_git=sum(p.stat().st_size for p in (REPO/'.git').rglob('*') if p.is_file())
    archive=sum(p.stat().st_size for root in (audit.SMOKE,audit.OUTPUT,L) for p in (REPO/'results'/root.name).rglob('*') if p.is_file())
    assert proof['live_bytes']+archive+max(0,after_git-before_git)<.25*1024**3 and shutil.disk_usage(ROOT).free>10*1024**3
    print(json.dumps(dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),archive_bytes=archive,git_increment_bytes=after_git-before_git,live_bytes=proof['live_bytes'],free_bytes=shutil.disk_usage(ROOT).free)),flush=True)
if __name__=='__main__':main()
