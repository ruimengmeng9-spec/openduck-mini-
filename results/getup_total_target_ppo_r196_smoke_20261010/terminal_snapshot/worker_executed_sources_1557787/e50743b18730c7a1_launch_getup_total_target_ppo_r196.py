"""Independent natural-exit smoke, then finite formal PPO; no hardware."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from diagnostics import train_getup_total_target_ppo_r196 as run

ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
OUTPUT=run.LAUNCH


def idle():
    for line in subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines():
        pid,args=line.strip().split(None,1)
        if int(pid)==os.getpid():continue
        first=Path(args.split()[0]).name
        if 'python' not in first and first!='git':continue
        if int(pid)==os.getppid() and '-m diagnostics.launch_getup_total_target_ppo_r196' in args:continue
        assert not ('-m diagnostics.' in args and any(s in args for s in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_'))),'Existing task; no signals or restart'
        assert 'publish_getup' not in args and 'git push' not in args


def tests(label):
    directory=OUTPUT/label;directory.mkdir()
    env={**ENV,'OPEN_DUCK_R196_TEST_CAPTURE':str(directory/'executed_sources')}
    with (directory/'regression.log').open('x') as stream:
        subprocess.run([sys.executable,'-m','diagnostics.test_getup_total_target_ppo_r196'],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:6])
    idle();assert not any(p.exists() for p in (OUTPUT,run.SMOKE,run.OUTPUT));assert shutil.disk_usage(run.ROOT).free>12*1024**3
    OUTPUT.mkdir();sources=run.capture_sources(OUTPUT/'executed_sources')
    run.local.write_json(OUTPUT/'contract.json',dict(cpu_affinity=sorted(os.sched_getaffinity(0)),total_cap=162,smoke_cap=6,formal_cap=156,storage_budget_gib=2,reserve_gib=10,sources=sources,full_task_completed=False,hardware_readiness=False))
    smoke_exit=None;formal_exit=None
    try:
        tests('initial')
        with run.SMOKE.with_suffix('.log').open('x') as stream:
            smoke=subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_total_target_ppo_r196','--smoke'],env=ENV,stdout=stream,stderr=subprocess.STDOUT)
        smoke_exit=smoke.returncode;assert smoke_exit==0
        idle();report=json.loads((run.SMOKE/'results.json').read_text());passed=run.smoke_gate(report['reports'])
        if passed:
            tests('formal');idle();assert not run.OUTPUT.exists()
            with run.OUTPUT.with_suffix('.log').open('x') as stream:
                formal=subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_total_target_ppo_r196'],env=ENV,stdout=stream,stderr=subprocess.STDOUT)
            formal_exit=formal.returncode;assert formal_exit==0
        run.local.write_json(OUTPUT/'result.json',dict(natural_exit=True,smoke_exit_code=smoke_exit,formal_exit_code=formal_exit,formal_not_run=not passed,smoke_gate_passed=passed,source_hashes_unchanged=all(run.prior.digest(p)==v['sha256'] for p,v in sources.items()),full_task_completed=False,hardware_readiness=False))
    except Exception:
        run.local.write_json(OUTPUT/'failure.json',dict(smoke_exit_code=smoke_exit,formal_exit_code=formal_exit,traceback=traceback.format_exc(),full_task_completed=False,hardware_readiness=False));raise
    finally:assert all(run.prior.digest(p)==v['sha256'] for p,v in sources.items())


if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle();assert not any(p.exists() for p in (OUTPUT,run.SMOKE,run.OUTPUT))
        with (run.ROOT/'tmp/getup_total_target_ppo_r196_launcher_20261010.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_total_target_ppo_r196'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,new_controller_not_yet_verified=True)),flush=True)
    else:assert not sys.argv[1:];main()
