"""Unique read-only audit, independent nonzero and invalid smoke then natural exit."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from diagnostics import audit_getup_phase_coordinate_terminal_r186 as run

ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}

def idle():
    lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(name in line for name in ('train_getup_','probe_getup_','audit_getup_')) for line in lines),'Existing task; no duplicate or signals'

def tests(label):
    log=run.ROOT/'tmp'/('getup_phase_coordinate_audit_r186_'+label+'_regression_20261010.log')
    with log.open('x') as stream:
        subprocess.run([sys.executable,'-m','diagnostics.test_getup_phase_coordinate_audit_r186'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    idle();assert not run.OUTPUT.exists() and not run.SMOKE.exists()
    assert shutil.disk_usage(run.ROOT).free>10*1024**3
    tests('initial')
    with run.SMOKE.with_suffix('.log').open('x') as stream:
        subprocess.run([sys.executable,'-u','-m','diagnostics.audit_getup_phase_coordinate_terminal_r186','--smoke'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    idle();assert json.loads((run.SMOKE/'results.json').read_text())['all_scalar_and_limits_bitwise']
    tests('formal');idle()
    with run.OUTPUT.with_suffix('.log').open('x') as stream:
        subprocess.run([sys.executable,'-u','-m','diagnostics.audit_getup_phase_coordinate_terminal_r186'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(natural_exit=True,new_dynamic_trajectories=0,full_task_completed=False)),flush=True)

if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle()
        with (run.ROOT/'tmp/getup_phase_coordinate_audit_r186_launcher_20261010.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_phase_coordinate_audit_r186'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,read_only=True,new_dynamic_trajectories=0)),flush=True)
    else:assert not sys.argv[1:];main()



