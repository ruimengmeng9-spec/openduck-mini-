"""Fixed-budget orchestration; natural smoke exit before original formal paths."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from diagnostics import probe_getup_velocity_bias_r191 as run
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
OUTPUT=run.ROOT/'outputs/getup_velocity_bias_launcher_r191_20261010'


def idle():
    lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines()
    for line in lines:
        pid,args=line.strip().split(None,1)
        if int(pid)==os.getpid():continue
        first=args.split()[0]
        if 'python' not in Path(first).name and Path(first).name!='git':continue
        if int(pid)==os.getppid() and '-m diagnostics.launch_getup_velocity_bias_r191' in args:continue
        assert not ('-m diagnostics.' in args and any(word in args for word in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_'))) ,'Existing task; wait without signals'
        assert not ('publish_getup' in args or 'git push' in args),'Existing publisher; no competing mutation'


def smoke_gate(report):
    assert report['terminal_result_saved'] and report['smoke']
    for name in ('parity','bias'):
        assert report[name]['nominal_success'] and not report[name]['physical_failures']
    assert all(all(r['complete_R157_parity'].values()) for r in report['parity']['rows'])
    standard=next(r for r in report['bias']['rows'] if r['case_seed'] is None)
    assert all(standard['complete_R157_parity'].values())
    assert standard['maximum_raw_velocity_bias_request_rad']==0.


def tests(label):
    destination=OUTPUT/label
    destination.mkdir()
    env={**ENV,'OPEN_DUCK_R191_TEST_CAPTURE':str(destination/'executed_sources')}
    with (destination/'regression.log').open('x') as stream:
        subprocess.run([sys.executable,'-m','diagnostics.test_getup_velocity_bias_r191'],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    idle();assert not OUTPUT.exists() and not run.OUTPUT.exists() and not run.SMOKE.exists()
    assert shutil.disk_usage(run.ROOT).free>12*1024**3
    OUTPUT.mkdir();sources=run.capture_sources(OUTPUT/'executed_sources')
    run.local.write_json(OUTPUT/'contract.json',dict(sources=sources,independent_smoke=6,formal_budget=56,storage_budget_gb=2,reserve_gb=10,full_task_completed=False,hardware_readiness=False))
    try:
        tests('initial')
        with run.SMOKE.with_suffix('.log').open('x') as stream:
            subprocess.run([sys.executable,'-u','-m','diagnostics.probe_getup_velocity_bias_r191','--smoke'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
        idle();smoke_gate(json.loads((run.SMOKE/'results.json').read_text()))
        tests('formal');idle();assert not run.OUTPUT.exists()
        with run.OUTPUT.with_suffix('.log').open('x') as stream:
            subprocess.run([sys.executable,'-u','-m','diagnostics.probe_getup_velocity_bias_r191'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
        run.local.write_json(OUTPUT/'result.json',dict(natural_exit=True,smoke_exit_code=0,formal_exit_code=0,source_hashes_unchanged=all(run.prior.digest(p)==v['sha256'] for p,v in sources.items()),full_task_completed=False,hardware_readiness=False))
    except Exception:
        run.local.write_json(OUTPUT/'failure.json',dict(traceback=traceback.format_exc(),full_task_completed=False,hardware_readiness=False))
        raise
    finally:
        assert all(run.prior.digest(p)==v['sha256'] for p,v in sources.items())
    print(json.dumps(dict(natural_exit=True,full_task_completed=False)),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle();assert not OUTPUT.exists() and not run.OUTPUT.exists() and not run.SMOKE.exists()
        with (run.ROOT/'tmp/getup_velocity_bias_r191_launcher_20261010.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_velocity_bias_r191'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,new_controller_not_yet_verified=True)),flush=True)
    else:assert not sys.argv[1:];main()

