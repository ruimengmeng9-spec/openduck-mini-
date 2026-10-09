"""Unique publication-only recovery; preserve failed waiting evidence."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path('/data/shijinsheng/open_duck')
BASE='dcf18833724b8ddbd52194d81b42c328a52e86fe'
RECEIPT=ROOT/'tmp/getup_publication_r175_r177_resume_20261010_result.json'
DONE=ROOT/'tmp/getup_publication_r177_r178_terminal_20261010_result.json'


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,indent=2)


def main():
    assert not RECEIPT.exists() and not DONE.exists()
    spec=importlib.util.spec_from_file_location('old_publication',ROOT/'tmp/finish_getup_publication_r175_r177_20261009.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    assert not old.existing_push(),'Do not duplicate or signal an active transfer'
    # Old transport exited with parent failure, but independent remote checks
    # already show BASE. Original helper skips a push when remote equals HEAD.
    old.publish(BASE,large_retry=True)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    def archive(module,base,*extra):
        output=subprocess.check_output([sys.executable,'-u','-m',module,'--base',base,*extra],cwd=ROOT/'projects/Open_Duck_Playground',env=env,text=True)
        print(output,flush=True);return json.loads(output.strip().splitlines()[-1])
    startup=archive('diagnostics.archive_getup_foot_task_startup_r177',BASE)
    assert startup['startup_only'] and not startup['terminal_result_saved']
    old.publish(startup['head'])
    save(RECEIPT,dict(published_terminal=BASE,published_startup=startup['head'],independent_remote_verification=True,
        previous_waiter_failed=True,old_result_file_not_created_or_overwritten=True,full_task_completed=False))
    terminal=archive('diagnostics.archive_getup_foot_task_terminal_r177_r178',startup['head'],'--publication-receipt',str(RECEIPT))
    assert terminal['terminal_result_saved'] and not terminal['full_task_completed']
    old.publish(terminal['head'],large_retry=True)
    save(DONE,dict(published_startup=startup['head'],published_terminal=terminal['head'],independent_remote_verification=True,
        upload_only_timeout_seconds=3600,training_budget_changed=False,full_task_completed=False,hardware_readiness=False))
    print(json.dumps(dict(publication_complete=True,**json.loads(DONE.read_text()))),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        log=ROOT/'tmp/getup_publication_r175_r178_resume_20261010.log'
        with log.open('x') as stream:
            child=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=child.pid,log=str(log),publication_not_yet_confirmed=True)),flush=True)
    else:
        assert not sys.argv[1:];main()
