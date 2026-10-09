"""Unique finite simulator-only pipeline; independent smoke exits naturally."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from diagnostics import train_getup_foot_orientation_r179 as run

ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}

def idle():
    lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(s in line for s in ('train_getup_','probe_getup_','audit_getup_')) for line in lines),'Existing task active; no duplicate and no signals'

def tests(name):
    path=run.ROOT/'tmp'/name
    with path.open('x') as stream:subprocess.run([sys.executable,'-m','diagnostics.test_getup_foot_orientation_r179'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    idle();assert not run.OUTPUT.exists() and not run.SMOKE.exists()
    assert shutil.disk_usage(run.ROOT).free>12*1024**3
    tests('getup_foot_orientation_r179_initial_regression_20261010.log')
    with run.SMOKE.with_suffix('.log').open('x') as stream:
        subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_foot_orientation_r179','--smoke'],stdout=stream,stderr=subprocess.STDOUT,env=ENV,check=True)
    idle();report=json.loads((run.SMOKE/'results.json').read_text())
    assert report['smoke'] and report['terminal_result_saved']
    assert report['parity']['nominal_success'] and not report['parity']['physical_failures'] and report['nonzero']['nominal_success']
    assert all(all(row['complete_R157_parity'].values()) for row in report['parity']['rows'])
    assert all(row['physics_unchanged'] for name in ('parity','nonzero') for row in report[name]['rows'])
    tests('getup_foot_orientation_r179_formal_regression_20261010.log')
    idle();assert not run.OUTPUT.exists()
    with run.OUTPUT.with_suffix('.log').open('x') as stream:
        subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_foot_orientation_r179'],stdout=stream,stderr=subprocess.STDOUT,env=ENV,check=True)
    print(json.dumps(dict(natural_exit=True,terminal_result_saved=True,full_task_completed=False)),flush=True)

if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle();log=run.ROOT/'tmp/getup_foot_orientation_r179_launcher_20261010.log'
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_foot_orientation_r179'],stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=ENV,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,log=str(log),new_controller_not_yet_verified=True)),flush=True)
    else:
        assert not sys.argv[1:];main()
