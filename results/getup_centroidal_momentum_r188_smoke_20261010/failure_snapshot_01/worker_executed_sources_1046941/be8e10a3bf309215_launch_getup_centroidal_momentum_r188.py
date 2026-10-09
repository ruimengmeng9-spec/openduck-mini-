"""No duplicate task or signals; smoke natural exit precedes fixed comparison."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from diagnostics import probe_getup_centroidal_momentum_r188 as run
ENV={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}


def idle():
    lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in line and any(s in line for s in ('train_getup_','probe_getup_','audit_getup_')) for line in lines),'Existing task; wait without signals'


def tests(label):
    log=run.ROOT/'tmp'/f'getup_centroidal_momentum_r188_{label}_regression_20261010.log'
    try:
        with log.open('x') as stream:subprocess.run([sys.executable,'-m','diagnostics.test_getup_centroidal_momentum_r188'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    except subprocess.CalledProcessError:
        failure=run.ROOT/'outputs'/f'getup_centroidal_momentum_r188_{label}_regression_failure_20261010';failure.mkdir(exist_ok=False)
        for name in ('probe_getup_centroidal_momentum_r188.py','test_getup_centroidal_momentum_r188.py','launch_getup_centroidal_momentum_r188.py'):shutil.copy2(Path(__file__).with_name(name),failure/name)
        shutil.copy2(log,failure/log.name)
        run.local.write_json(failure/'result.json',dict(regression_passed=False,dynamic_replay_started=False,full_task_completed=False,hashes={f.name:run.prior.digest(f) for f in failure.iterdir()}))
        raise


def main():
    idle();assert not run.OUTPUT.exists() and not run.SMOKE.exists()
    assert shutil.disk_usage(run.ROOT).free>12*1024**3
    tests('initial')
    with run.SMOKE.with_suffix('.log').open('x') as stream:subprocess.run([sys.executable,'-u','-m','diagnostics.probe_getup_centroidal_momentum_r188','--smoke'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    idle();report=json.loads((run.SMOKE/'results.json').read_text())
    assert report['terminal_result_saved'] and report['smoke'] and report['parity']['nominal_success'] and not report['parity']['physical_failures'] and report['momentum']['nominal_success'] and not report['phase']['physical_failures']
    assert all(all(r['complete_R157_parity'].values()) for r in report['parity']['rows'])
    tests('formal');idle();assert not run.OUTPUT.exists()
    with run.OUTPUT.with_suffix('.log').open('x') as stream:subprocess.run([sys.executable,'-u','-m','diagnostics.probe_getup_centroidal_momentum_r188'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(dict(natural_exit=True,full_task_completed=False)),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle()
        with (run.ROOT/'tmp/getup_centroidal_momentum_r188_launcher_20261010.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_centroidal_momentum_r188'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,new_controller_not_yet_verified=True)),flush=True)
    else:assert not sys.argv[1:];main()


