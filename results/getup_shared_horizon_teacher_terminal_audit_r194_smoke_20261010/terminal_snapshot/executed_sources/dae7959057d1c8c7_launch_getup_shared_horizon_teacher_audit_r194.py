"""Unique R194 read-only audit, complete independent smoke natural exit before formal."""
import json,os,subprocess,sys,traceback
from pathlib import Path
from diagnostics import audit_getup_shared_horizon_teacher_terminal_r194 as run
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
FOLDER=run.ROOT/'outputs/getup_shared_horizon_teacher_audit_launcher_r194_20261010'
BASE='13c583ec2fddca0dafc9ea280da1f5e8cf0b4de1'
def idle():
    repo=run.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()==BASE
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip()
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
        pid,args=line.strip().split(None,1)
        if int(pid)==os.getpid():continue
        first=Path(args.split()[0]).name
        if 'python' not in first and first!='git':continue
        if int(pid)==os.getppid() and '-m diagnostics.launch_getup_shared_horizon_teacher_audit_r194' in args:continue
        assert not ('-m diagnostics.' in args and any(w in args for w in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_'))),'Existing task; wait without signals'
        assert not ('publish_getup' in args or 'git push' in args),'Existing publisher'
def tests(label):
    folder=FOLDER/label;folder.mkdir()
    env={**ENV,'OPEN_DUCK_R194_TEST_CAPTURE':str(folder/'executed_sources')}
    with (folder/'regression.log').open('x') as f:
        child=subprocess.run([sys.executable,'-m','diagnostics.test_getup_shared_horizon_teacher_audit_r194'],env=env,stdout=f,stderr=subprocess.STDOUT)
    run.write(folder/'result.json',dict(exit_code=child.returncode,natural_exit=True,new_dynamic_trajectories=0))
    assert child.returncode==0
def main():
    idle();assert not any(p.exists() for p in (run.OUTPUT,run.SMOKE,FOLDER))
    run.budget();FOLDER.mkdir()
    run.write(FOLDER/'executed_sources_manifest.json',run.capture_sources(FOLDER/'executed_sources'))
    before={str(p):run.digest(p) for p in run.python_sources()}
    try:
        tests('initial');run.budget()
        with run.SMOKE.with_suffix('.log').open('x') as f:
            child=subprocess.run([sys.executable,'-u','-m','diagnostics.audit_getup_shared_horizon_teacher_terminal_r194','--smoke'],env=ENV,stdout=f,stderr=subprocess.STDOUT)
        run.write(FOLDER/'smoke_exit.json',dict(natural_exit=True,exit_code=child.returncode))
        assert child.returncode==0;idle()
        report=run.read(run.SMOKE/'results.json');assert report['terminal_result_saved'] and report['all_scalar_and_limits_bitwise'] and report['existing_trajectories_audited']==6
        tests('formal');idle();run.budget()
        with run.OUTPUT.with_suffix('.log').open('x') as f:
            child=subprocess.run([sys.executable,'-u','-m','diagnostics.audit_getup_shared_horizon_teacher_terminal_r194'],env=ENV,stdout=f,stderr=subprocess.STDOUT)
        run.write(FOLDER/'formal_exit.json',dict(natural_exit=True,exit_code=child.returncode))
        assert child.returncode==0
        assert before=={str(p):run.digest(p) for p in run.python_sources()}
        result=dict(natural_exit=True,smoke_exit_code=0,formal_exit_code=0,new_dynamic_trajectories=0,source_hashes_unchanged=True,total_live_bytes=run.budget(),full_task_completed=False,hardware_readiness=False,qualification=False)
        run.write(FOLDER/'result.json',result);print(json.dumps(result),flush=True)
    except BaseException:
        run.write(FOLDER/'failure.json',dict(traceback=traceback.format_exc(),new_dynamic_trajectories=0,full_task_completed=False));raise
if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle();assert not any(p.exists() for p in (run.OUTPUT,run.SMOKE,FOLDER))
        run.budget()
        with (run.ROOT/'tmp/getup_shared_horizon_teacher_audit_r194_launcher_20261010.log').open('x') as f:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_shared_horizon_teacher_audit_r194'],env=ENV,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,read_only=True,new_dynamic_trajectories=0)),flush=True)
    else:assert not sys.argv[1:];main()
