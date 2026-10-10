"""Finite orchestration with natural smoke exit and causal model gates."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from diagnostics import train_getup_adaptive_sensor_mpc_r195 as run
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
OUTPUT=run.ROOT/'outputs/getup_adaptive_sensor_mpc_launcher_r195_20261010'

def idle():
    lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','pid=,args='],text=True).splitlines()
    for line in lines:
        pid,args=line.strip().split(None,1)
        if int(pid)==os.getpid():continue
        first=args.split()[0]
        if 'python' not in Path(first).name and Path(first).name!='git':continue
        if int(pid)==os.getppid() and '-m diagnostics.launch_getup_adaptive_sensor_mpc_r195' in args:continue
        assert not ('-m diagnostics.' in args and any(word in args for word in ('train_getup_','probe_getup_','audit_getup_','launch_getup_','validate_'))),'Existing task; wait without signals'
        assert not ('publish_getup' in args or 'git push' in args),'Existing publisher; no competing mutation'

def smoke_gate(report):
    assert report['terminal_result_saved'] and report['smoke']
    for value in report['reports'].values():
        if not value['nominal_success'] or value['physical_failures'] or any(r['adaptive_fallback_controls'] for r in value['rows']):return False
    assert all(all(r['complete_R157_parity'].values()) for r in report['reports']['zero_parity']['rows'])
    standard=report['reports']['calibration_standard']['rows'][0]
    assert standard['case_seed'] is None and all(standard['complete_R157_parity'].values()) and standard['maximum_raw_adaptive_request_rad']==0
    return bool(report['causal_effect']['passed'])

def tests(label):
    destination=OUTPUT/label;destination.mkdir()
    env={**ENV,'OPEN_DUCK_R195_TEST_CAPTURE':str(destination/'executed_sources')}
    with (destination/'regression.log').open('x') as stream:
        subprocess.run([sys.executable,'-m','diagnostics.test_getup_adaptive_sensor_mpc_r195'],env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    idle();assert not OUTPUT.exists() and not run.OUTPUT.exists() and not run.SMOKE.exists()
    assert shutil.disk_usage(run.ROOT).free>12*1024**3
    OUTPUT.mkdir();sources=run.capture_sources(OUTPUT/'executed_sources')
    run.local.write_json(OUTPUT/'contract.json',dict(sources=sources,independent_identification_smoke_cap=8,formal_identification_cap=80,remaining_candidate_cap=28,total_cap=116,storage_budget_gib=2,reserve_gib=10,full_task_completed=False,hardware_readiness=False))
    try:
        tests('initial')
        with run.SMOKE.with_suffix('.log').open('x') as stream:
            subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_adaptive_sensor_mpc_r195','--smoke'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
        idle();report=json.loads((run.SMOKE/'results.json').read_text())
        if not smoke_gate(report):
            run.local.write_json(OUTPUT/'result.json',dict(natural_exit=True,smoke_exit_code=0,formal_exit_code=None,formal_not_run=True,stopped_at_causal_or_physical_smoke_gate=True,causal_effect=report['causal_effect'],source_hashes_unchanged=all(run.prior.digest(p)==v['sha256'] for p,v in sources.items()),full_task_completed=False,hardware_readiness=False))
            print('R195_SMOKE_GATE_NOT_PASSED_NO_FORMAL',flush=True);return
        tests('formal');idle();assert not run.OUTPUT.exists()
        with run.OUTPUT.with_suffix('.log').open('x') as stream:
            subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_adaptive_sensor_mpc_r195'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
        run.local.write_json(OUTPUT/'result.json',dict(natural_exit=True,smoke_exit_code=0,formal_exit_code=0,formal_not_run=False,source_hashes_unchanged=all(run.prior.digest(p)==v['sha256'] for p,v in sources.items()),full_task_completed=False,hardware_readiness=False))
    except Exception:
        run.local.write_json(OUTPUT/'failure.json',dict(traceback=traceback.format_exc(),full_task_completed=False,hardware_readiness=False));raise
    finally:
        assert all(run.prior.digest(p)==v['sha256'] for p,v in sources.items())

if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle();assert not OUTPUT.exists() and not run.OUTPUT.exists() and not run.SMOKE.exists()
        with (run.ROOT/'tmp/getup_adaptive_sensor_mpc_r195_launcher_20261010.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_adaptive_sensor_mpc_r195'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,new_controller_not_yet_verified=True)),flush=True)
    else:assert not sys.argv[1:];main()
