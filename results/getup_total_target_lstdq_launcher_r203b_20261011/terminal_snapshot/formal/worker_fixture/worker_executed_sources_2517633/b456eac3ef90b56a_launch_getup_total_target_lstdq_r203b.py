"""Strict zero-skip R203 gates and actual post-child-exit supervision."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from diagnostics import train_getup_total_target_lstdq_r203b as run

ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
OUTPUT=run.LAUNCH
EXIT=run.ROOT/'tmp/getup_total_target_lstdq_r203b_launcher_exit_20261011.json'
LOG=run.ROOT/'tmp/getup_total_target_lstdq_r203b_launcher_20261011.log'

def idle():
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
        pid,args=line.strip().split(None,1)
        if int(pid)==os.getpid():continue
        first=Path(args.split()[0]).name
        if 'python' not in first and first!='git':continue
        if int(pid)==os.getppid() and '-m diagnostics.launch_getup_total_target_lstdq_r203b --supervise' in args:continue
        assert not ('-m diagnostics.' in args and any(s in args for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_','verify_getup_'))),'Existing task; no signals or restart'
        assert 'publish_getup' not in args and 'git push' not in args

def tests(label):
    directory=OUTPUT/label;directory.mkdir()
    env={**ENV,'OPEN_DUCK_R203_TEST_CAPTURE':str(directory/'executed_sources')}
    started=time.monotonic()
    with (directory/'regression.log').open('x') as stream:
        child=subprocess.run([sys.executable,'-m','diagnostics.test_getup_total_target_lstdq_r203b'],env=env,stdout=stream,stderr=subprocess.STDOUT)
    result=json.loads((directory/'interface_result.json').read_text())
    run.local.write_json(directory/'regression_natural_exit.json',dict(
        **result,natural_exit=True,exit_code=child.returncode,subprocess_elapsed_seconds=time.monotonic()-started))
    assert child.returncode==0 and result['tests_run']==16
    assert result['skipped']==result['failures']==result['errors']==0
    assert result['source_hashes_unchanged'] and result['new_dynamic_attempts']==0
    return result

def prerequisites():
    numeric=json.loads((run.NUMERIC/'result.json').read_text())
    ne=json.loads((run.ROOT/'tmp/getup_total_target_lstdq_r203_numeric_exit_20261011.json').read_text())
    assert numeric['passed'] and numeric['tests_run']==24 and numeric['skipped']==numeric['errors']==numeric['failures']==0
    assert ne['natural_exit'] and ne['exit_code']==0 and ne['source_hashes_unchanged']
    pre=json.loads((run.ROOT/'tmp/getup_total_target_lstdq_r203b_preflight_exit_20261011.json').read_text())
    pr=json.loads((run.PREFLIGHT/'interface_result.json').read_text())
    assert pre['natural_exit'] and pre['exit_code']==0 and pre['source_hashes_unchanged']
    assert pr['tests_run']==16 and pr['skipped']==pr['failures']==pr['errors']==0 and pr['source_hashes_unchanged']

def main():
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:6])
    idle();assert not any(p.exists() for p in (OUTPUT,run.SMOKE,run.OUTPUT));run.storage_gate(start=True)
    repo=run.ROOT/'github/openduck-mini-'
    assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==run.EXPECTED_HEAD
    assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
    prerequisites()
    run.kernel.Learner(run.kernel.initial_weights())
    OUTPUT.mkdir();sources=run.capture_sources(OUTPUT/'executed_sources')
    run.local.write_json(OUTPUT/'contract.json',dict(cpu_affinity=sorted(os.sched_getaffinity(0)),total_cap=162,smoke_cap=6,formal_cap=156,strict_tests=16,skipped_allowed=0,storage_budget_gib=2,reserve_gib=10,sources=sources,full_task_completed=False,hardware_readiness=False))
    smoke_exit=None;formal_exit=None
    try:
        run.storage_gate(start=True);tests('initial');run.storage_gate()
        started=time.monotonic()
        with run.SMOKE.with_suffix('.log').open('x') as stream:
            smoke=subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_total_target_lstdq_r203b','--smoke'],env=ENV,stdout=stream,stderr=subprocess.STDOUT)
        smoke_exit=smoke.returncode
        run.local.write_json(OUTPUT/'smoke_natural_exit.json',dict(natural_exit=True,exit_code=smoke_exit,elapsed_seconds=time.monotonic()-started,actual_subprocess_return_observed=True))
        assert smoke_exit==0
        idle();report=json.loads((run.SMOKE/'results.json').read_text());passed=run.smoke_gate(report['reports'])
        if passed:
            tests('formal');run.storage_gate(start=True);idle();assert not run.OUTPUT.exists()
            started=time.monotonic()
            with run.OUTPUT.with_suffix('.log').open('x') as stream:
                formal=subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_total_target_lstdq_r203b'],env=ENV,stdout=stream,stderr=subprocess.STDOUT)
            formal_exit=formal.returncode
            run.local.write_json(OUTPUT/'formal_natural_exit.json',dict(natural_exit=True,exit_code=formal_exit,elapsed_seconds=time.monotonic()-started,actual_subprocess_return_observed=True))
            assert formal_exit==0
        run.local.write_json(OUTPUT/'result.json',dict(terminal_result_saved=True,smoke_exit_code=smoke_exit,formal_exit_code=formal_exit,formal_not_run=not passed,smoke_gate_passed=passed,source_hashes_unchanged=all(run.prior.digest(p)==v['sha256'] for p,v in sources.items()),launcher_natural_exit_not_yet_observed_here=True,full_task_completed=False,hardware_readiness=False))
    except Exception:
        run.local.write_json(OUTPUT/'failure.json',dict(smoke_exit_code=smoke_exit,formal_exit_code=formal_exit,traceback=traceback.format_exc(),full_task_completed=False,hardware_readiness=False));raise
    finally:assert all(run.prior.digest(p)==v['sha256'] for p,v in sources.items())

def supervise():
    assert not EXIT.exists()
    hashes={str(Path(__file__)):run.prior.digest(__file__),str(Path(run.__file__)):run.prior.digest(run.__file__),str(Path(run.kernel.__file__)):run.prior.digest(run.kernel.__file__)}
    started=time.monotonic()
    child=subprocess.run([sys.executable,'-u','-m','diagnostics.launch_getup_total_target_lstdq_r203b'],env=ENV)
    unchanged=all(run.prior.digest(p)==sha for p,sha in hashes.items())
    run.local.write_json(EXIT,dict(natural_exit=True,exit_code=child.returncode,elapsed_seconds=time.monotonic()-started,actual_launcher_subprocess_return_observed=True,source_hashes_unchanged=unchanged,source_hashes=hashes,protected_live_server_not_operated=True))
    assert unchanged
    sys.exit(child.returncode)

if __name__=='__main__':
    if sys.argv[1:]==['--supervise']:supervise()
    elif sys.argv[1:]==['--launch']:
        idle();prerequisites();run.storage_gate(start=True)
        assert not any(p.exists() for p in (OUTPUT,run.SMOKE,run.OUTPUT,EXIT,LOG))
        with LOG.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_total_target_lstdq_r203b','--supervise'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(supervisor_pid=child.pid,only_locator_not_natural_exit=True,new_controller_not_yet_verified=True)),flush=True)
    else:assert not sys.argv[1:];main()

