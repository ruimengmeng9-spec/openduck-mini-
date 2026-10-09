"""Correct only the smoke gate field; reuse immutable R188b completed smoke."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
from diagnostics import probe_getup_centroidal_momentum_r188b as run
from diagnostics.launch_getup_centroidal_momentum_r188b import idle,ENV

OUTPUT=run.ROOT/'outputs/getup_centroidal_momentum_launcher_r188c_20261010'


def main():
    idle();assert not OUTPUT.exists() and not run.OUTPUT.exists()
    assert shutil.disk_usage(run.ROOT).free>12*1024**3
    report=json.loads((run.SMOKE/'results.json').read_text())
    assert report['terminal_result_saved'] and report['smoke']
    for name in ('parity','momentum'):
        assert report[name]['nominal_success'] and not report[name]['physical_failures']
    assert all(all(r['complete_R157_parity'].values()) for r in report['parity']['rows'])
    standard=next(r for r in report['momentum']['rows'] if r['case_seed'] is None)
    assert all(standard['complete_R157_parity'].values())
    assert standard['maximum_raw_momentum_request_rad']==0.
    OUTPUT.mkdir();sources=run.capture_sources(OUTPUT/'executed_sources')
    run.local.write_json(OUTPUT/'contract.json',dict(reuses_completed_independent_smoke=str(run.SMOKE),no_additional_smoke_attempts=True,new_controller_source=False,sources=sources,formal_budget=56,full_task_completed=False,hardware_readiness=False))
    log=OUTPUT/'formal_regression.log'
    with log.open('x') as stream:subprocess.run([sys.executable,'-m','diagnostics.test_getup_centroidal_momentum_r188b'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    idle();assert not run.OUTPUT.exists()
    with run.OUTPUT.with_suffix('.log').open('x') as stream:subprocess.run([sys.executable,'-u','-m','diagnostics.probe_getup_centroidal_momentum_r188b'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,check=True)
    assert all(run.prior.digest(p)==info['sha256'] for p,info in sources.items())
    run.local.write_json(OUTPUT/'result.json',dict(natural_exit=True,formal_process_exit_code=0,additional_smoke_attempts=0,source_hashes_unchanged=True,full_task_completed=False,hardware_readiness=False))
    print(json.dumps(dict(natural_exit=True,full_task_completed=False)),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        idle();assert not OUTPUT.exists() and not run.OUTPUT.exists()
        with (run.ROOT/'tmp/getup_centroidal_momentum_r188c_launcher_20261010.log').open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u','-m','diagnostics.launch_getup_centroidal_momentum_r188c'],env=ENV,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,reuses_closed_independent_smoke=True)),flush=True)
    else:assert not sys.argv[1:];main()
