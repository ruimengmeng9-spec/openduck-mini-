"""Finite publication continuation: wait for existing push, verify, retry once.

Does not signal processes, change training, copy credentials or alter old helpers.
Only git push transport gets a 3600s deadline for the one large terminal retry.
"""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/data/shijinsheng/open_duck')
REPO=ROOT/'github/openduck-mini-'
BASE='dcf18833724b8ddbd52194d81b42c328a52e86fe'
HELPER=ROOT/'tmp/push_openduck_server_20261009.py'
DONE=ROOT/'tmp/getup_publication_r175_r177_20261009_result.json'


def existing_push():
    lines=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    return any((str(HELPER) in line and '--expected '+BASE in line and '--push' in line)
        or 'git -c gc.auto=0 push --progress server-publish HEAD:refs/heads/main' in line
        or ('git-receive-pack' in line and 'ruimengmeng9-spec/openduck-mini-' in line) for line in lines)


def publish(commit,large_retry=False):
    spec=importlib.util.spec_from_file_location('scoped_publisher',HELPER)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_run=subprocess.run;original_argv=sys.argv
    def bounded_run(*args,**kwargs):
        command=args[0] if args else kwargs.get('args')
        if large_retry and command==['git','-c','gc.auto=0','push','--progress','server-publish','HEAD:refs/heads/main']:
            assert kwargs.get('timeout')==900
            kwargs['timeout']=3600
        return original_run(*args,**kwargs)
    try:
        subprocess.run=bounded_run;sys.argv=[str(HELPER),'--expected',commit,'--push'];module.main()
    finally:subprocess.run=original_run;sys.argv=original_argv
    # Independently re-query with the original unmodified helper, read-only.
    original_run([sys.executable,'-u',str(HELPER),'--expected',commit],check=True)


def main():
    assert not DONE.exists()
    started=time.monotonic()
    while existing_push():
        assert time.monotonic()-started<900,'Existing transfer still active; do not duplicate or signal'
        time.sleep(5)
    print('EXISTING_TRANSFER_EXITED_VERIFY_BEFORE_ANY_DEPENDENCY',flush=True)
    publish(BASE,large_retry=True)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    archive=subprocess.check_output([sys.executable,'-u','-m','diagnostics.archive_getup_foot_task_startup_r177','--base',BASE],cwd=ROOT/'projects/Open_Duck_Playground',env=env,text=True)
    report=json.loads(archive.strip().splitlines()[-1]);commit=report['head'];assert report['startup_only'] and not report['terminal_result_saved']
    publish(commit)
    with DONE.open('x') as f:json.dump(dict(published_terminal=BASE,published_startup=commit,independent_remote_verification=True,full_task_completed=False),f,indent=2)
    print(json.dumps(dict(publication_complete=True,terminal=BASE,startup=commit,training_status_not_inferred=True)),flush=True)


if __name__=='__main__':
    if sys.argv[1:]==['--launch']:
        log=ROOT/'tmp/getup_publication_r175_r177_20261009.log'
        with log.open('x') as stream:
            p=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve())],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(pid=p.pid,log=str(log),publication_not_yet_confirmed=True)),flush=True)
    else:
        assert not sys.argv[1:];main()
