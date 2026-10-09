"""Finite simulation pipeline; never signals or resumes old tasks."""
import json
import os
import shutil
import subprocess
import sys
from diagnostics import train_getup_full_episode_pg_r181 as run

ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}

def idle():
    lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(s in line for s in ('train_getup_','probe_getup_','audit_getup_')) for line in lines),'Existing task active; no duplicate or signals'

def tests(name):
    with (run.ROOT/'tmp'/name).open('x') as stream:subprocess.run([sys.executable,'-m','diagnostics.test_getup_full_episode_pg_r181'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    idle();assert not run.OUTPUT.exists() and not run.SMOKE.exists();assert shutil.disk_usage(run.ROOT).free>12*1024**3
    tests('getup_full_episode_pg_r181_initial_regression_20261010.log')
    with run.SMOKE.with_suffix('.log').open('x') as stream:subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_full_episode_pg_r181','--smoke'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    idle();r=json.loads((run.SMOKE/'results.json').read_text());assert r['smoke'] and r['terminal_result_saved'] and r['parity']['nominal_success'] and not r['parity']['physical_failures'] and r['nonzero']['nominal_success']
    assert all(all(v['complete_R157_parity'].values()) for v in r['parity']['rows']);assert all(v['physics_unchanged'] for key in ('parity','nonzero') for v in r[key]['rows'])
    tests('getup_full_episode_pg_r181_formal_regression_20261010.log');idle();assert not run.OUTPUT.exists()
    with run.OUTPUT.with_suffix('.log').open('x') as stream:subprocess.run([sys.executable,'-u','-m','diagnostics.train_getup_full_episode_pg_r181'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(natural_exit=True,terminal_result_saved=True,full_task_completed=False)),flush=True)

if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle()
        with (run.ROOT/'tmp/getup_full_episode_pg_r181_launcher_20261010.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_full_episode_pg_r181'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,new_controller_not_yet_verified=True)),flush=True)
    else:assert not sys.argv[1:];main()
