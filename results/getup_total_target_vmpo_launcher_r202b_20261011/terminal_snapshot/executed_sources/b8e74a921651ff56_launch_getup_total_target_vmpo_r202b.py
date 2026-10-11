"""Corrected zero-skip launcher; interrupted R202 budget is never replayed."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from diagnostics import train_getup_total_target_vmpo_r202 as run

ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
OUTPUT=run.ROOT/'outputs/getup_total_target_vmpo_launcher_r202b_20261011'


def idle():
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
        pid,args=line.strip().split(None,1)
        if int(pid)==os.getpid():continue
        first=Path(args.split()[0]).name
        if 'python' not in first and first!='git':continue
        if int(pid)==os.getppid() and '-m diagnostics.launch_getup_total_target_vmpo_r202b' in args:continue
        assert not ('-m diagnostics.' in args and any(s in args for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_'))),'Existing task; no signals or restart'
        assert 'publish_getup' not in args and 'git push' not in args


def tests(label):
    directory=OUTPUT/label;directory.mkdir()
    env={**ENV,'OPEN_DUCK_R202_TEST_CAPTURE':str(directory/'executed_sources')}
    with (directory/'regression.log').open('x') as stream:
        subprocess.run([sys.executable,'-m','diagnostics.test_getup_total_target_vmpo_r202'],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    text=(directory/'regression.log').read_text()
    assert 'Ran 16 tests in ' in text and '\nOK\n' in text and 'skipped' not in text, 'Incomplete full gate: skipped or missing tests'


def main():
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:6])
    idle();assert not any(p.exists() for p in (OUTPUT,run.SMOKE,run.OUTPUT));run.storage_gate(start=True)
    assert subprocess.check_output(['git','-C',str(run.ROOT/'github/openduck-mini-'),'rev-parse','HEAD'],text=True).strip()=='9b4edd92c86f0c689f15f904c0298cae52f17b82'
    assert not subprocess.check_output(['git','-C',str(run.ROOT/'github/openduck-mini-'),'status','--porcelain'],text=True).strip()
    run.kernel.Learner(run.kernel.initial_weights()) # Runtime imported before explicit/imported Python capture.
    OUTPUT.mkdir();sources=run.capture_sources(OUTPUT/'executed_sources')
    run.local.write_json(OUTPUT/'contract.json',dict(cpu_affinity=sorted(os.sched_getaffinity(0)),total_cap=162,smoke_cap=6,formal_cap=156,storage_budget_gib=2,reserve_gib=10,sources=sources,full_task_completed=False,hardware_readiness=False))
    smoke_exit=None;formal_exit=None
    try:
        assert json.loads((run.ROOT/'tmp/getup_total_target_vmpo_r202_preflight_exit_20261011.json').read_text())['exit_code']==0
        numeric=json.loads((run.NUMERIC_B/'result.json').read_text())
        numeric_exit=json.loads((run.ROOT/'tmp/getup_total_target_vmpo_r202b_numeric_exit_20261011.json').read_text())
        assert numeric['passed'] and numeric['tests_run']==24 and numeric_exit['natural_exit'] and numeric_exit['exit_code']==0
        run.storage_gate(start=True);tests('initial');run.storage_gate()
        with run.SMOKE.with_suffix('.log').open('x') as stream:
            smoke=subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_total_target_vmpo_r202','--smoke'],env=ENV,stdout=stream,stderr=subprocess.STDOUT)
        smoke_exit=smoke.returncode;assert smoke_exit==0
        idle();report=json.loads((run.SMOKE/'results.json').read_text());passed=run.smoke_gate(report['reports'])
        if passed:
            tests('formal');run.storage_gate();idle();assert not run.OUTPUT.exists()
            with run.OUTPUT.with_suffix('.log').open('x') as stream:
                formal=subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_total_target_vmpo_r202'],env=ENV,stdout=stream,stderr=subprocess.STDOUT)
            formal_exit=formal.returncode;assert formal_exit==0
        run.local.write_json(OUTPUT/'result.json',dict(natural_exit=True,smoke_exit_code=smoke_exit,formal_exit_code=formal_exit,formal_not_run=not passed,smoke_gate_passed=passed,source_hashes_unchanged=all(run.prior.digest(p)==v['sha256'] for p,v in sources.items()),full_task_completed=False,hardware_readiness=False))
    except Exception:
        run.local.write_json(OUTPUT/'failure.json',dict(smoke_exit_code=smoke_exit,formal_exit_code=formal_exit,traceback=traceback.format_exc(),full_task_completed=False,hardware_readiness=False));raise
    finally:assert all(run.prior.digest(p)==v['sha256'] for p,v in sources.items())



def gate_only():
    """Post-stop offline repair check. Existing formal budget cannot be replayed."""
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:6])
    idle();assert not OUTPUT.exists()
    assert run.OUTPUT.exists(), 'This gate-only repair binds the actually interrupted formal output'
    stop=run.ROOT/'tmp/getup_total_target_vmpo_r202_user_authorized_task_stop_20261011.json'
    stopped=json.loads(stop.read_text());assert stopped['user_authorized_task_stop']
    assert stopped['formal_budget_consumed_not_restarted']
    run.storage_gate()
    run.kernel.Learner(run.kernel.initial_weights())
    OUTPUT.mkdir()
    sources=run.capture_sources(OUTPUT/'executed_sources')
    before={str(p):run.prior.digest(p) for d in (run.SMOKE,run.OUTPUT,run.LAUNCH) for p in d.rglob('*') if p.is_file()}
    run.local.write_json(OUTPUT/'original_dynamic_evidence_inventory_before.json',before)
    try:
        tests('complete_gate')
        # Include the newly separate repair directory explicitly; old main's storage guard
        # knows only the original launcher, so it is not falsely claimed to count this one.
        repaired_bytes=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())
        budget=run.storage_gate()
        assert repaired_bytes <= 128*1024**2
        assert all(run.prior.digest(p)==sha for p,sha in before.items())
        assert all(run.prior.digest(p)==entry['sha256'] for p,entry in sources.items())
        run.local.write_json(OUTPUT/'result.json',dict(
            repair_gate_only=True,tests_run=16,skipped=0,passed=True,new_dynamic_attempts=0,
            original_dynamic_evidence_hashes_unchanged=True,
            source_hashes_unchanged=True,repair_directory_bytes=repaired_bytes,
            repair_directory_separately_counted_in_128MiB_extra_reservation=True,
            original_formal_interrupted_by_user_authorization=True,
            original_formal_budget_closed=True,remaining_budget_not_replayed=True,
            formal_restarted=False,smoke_replayed=False,full_task_completed=False,
            hardware_readiness=False))
    except Exception:
        run.local.write_json(OUTPUT/'failure.json',dict(
            traceback=traceback.format_exc(),repair_gate_only=True,new_dynamic_attempts=0,
            original_formal_budget_closed=True,formal_restarted=False));raise


if __name__=='__main__':
    if sys.argv[1:]==['--gate-only']:
        gate_only()
    elif sys.argv[1:]==['--launch']:
        idle();assert not any(p.exists() for p in (OUTPUT,run.SMOKE,run.OUTPUT))
        with (run.ROOT/'tmp/getup_total_target_vmpo_r202b_launcher_20261011.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_total_target_vmpo_r202b'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,new_controller_not_yet_verified=True)),flush=True)
    else:assert not sys.argv[1:];main()

