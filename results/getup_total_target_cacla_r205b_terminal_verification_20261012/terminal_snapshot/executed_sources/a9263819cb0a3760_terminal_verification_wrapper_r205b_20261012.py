"""One new standalone readonly R205 B child; actual natural exit, immutable A retained."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
SOURCE=PROJECT/'diagnostics/verify_getup_total_target_cacla_terminal_r205b.py'
EXPECTED_SOURCE='7cdfc7f2615389c57200dea8f73fb4ea98fc0797705fc71409f617e0f1e2ae2c'
AUTH=ROOT/'tmp/r205_storage_cap_authorized_4gib_20261012.json'
AUTH_SHA='b737435de94f9e7248241b4a6e0e5e7895c6ec52d1101ea72d072f3b0aae587d'
EXIT=ROOT/'tmp/getup_total_target_cacla_r205b_terminal_verification_exit_20261012.json'
LOG=ROOT/'tmp/getup_total_target_cacla_r205b_terminal_verification_20261012.log'
PROOF=ROOT/'tmp/getup_total_target_cacla_r205b_terminal_verification_20261012.json'
DEST=ROOT/'outputs/getup_total_target_cacla_r205b_terminal_verification_20261012'
CAP=4294967296
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def resources(additional=0,start=False):
    files=set()
    for d in (ROOT/'tmp',ROOT/'outputs',PROJECT/'diagnostics'):
        for p in d.iterdir():
            if 'r205' in p.name.lower():
                files.update([p] if p.is_file() else [q for q in p.rglob('*') if q.is_file()])
    live=sum(p.stat().st_size for p in files)
    local_bound=16*1024**2;new_bound=32*1024**2
    archive_bound=max(640*1024**2,live+local_bound+new_bound+16*1024**2)
    git_bound=max(640*1024**2,2*archive_bound+64*1024**2)
    total=live+archive_bound+git_bound+128*1024**2+local_bound+additional
    free=shutil.disk_usage(ROOT).free
    assert total<=CAP and free>=(12 if start else 10)*1024**3,'Full R205 4GiB gate, no bypass'
    return dict(live_bytes=live,future_archive_bytes=archive_bound,
        future_git_including_temporary_pack_index_ref_bytes=git_bound,future_extra_bytes=128*1024**2,
        local_continuation_future_upload_bound_bytes=local_bound,additional_bound_bytes=additional,
        total_bytes=total,cap_bytes=CAP,free_bytes=free)
assert not EXIT.exists() and not LOG.exists() and not PROOF.exists() and not DEST.exists()
assert digest(AUTH)==AUTH_SHA
before=digest(SOURCE);assert before==EXPECTED_SOURCE
wrapper=digest(__file__)
gate=resources(48*1024**2,start=True) # Full fresh source capture, proof/manifests/log/exit reserve.
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
    CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1')
started=time.monotonic()
with LOG.open('xb') as log:
    result=subprocess.run([str(PROJECT/'.venv/bin/python'),'-B',str(SOURCE)],
        cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
elapsed=time.monotonic()-started
assert digest(SOURCE)==before and digest(__file__)==wrapper and digest(AUTH)==AUTH_SHA
receipt=dict(actual_subprocess_return_observed=True,natural_exit=True,exit_code=result.returncode,
    elapsed_seconds=elapsed,source_unchanged=True,source_sha256=before,wrapper_sha256=wrapper,
    proof_saved=PROOF.exists(),new_dynamic_attempts=0,log_sha256=digest(LOG),
    protected_live_server_not_operated=True,failed_A_not_modified_or_reexecuted=True,
    storage_cap_bytes=CAP,authorization_sha256=AUTH_SHA,resources_before=gate,resources_after=resources(8192))
if PROOF.exists():receipt['proof_sha256']=digest(PROOF)
with EXIT.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(result.returncode)
