"""Single synchronous fresh R205 readonly verifier child; preserve actual natural return."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
SOURCE=PROJECT/'diagnostics/verify_getup_total_target_cacla_terminal_r205.py'
EXIT=ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_exit_20261011.json'
LOG=ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_20261011.log'
PROOF=ROOT/'tmp/getup_total_target_cacla_r205_terminal_verification_20261011.json'
DEST=ROOT/'outputs/getup_total_target_cacla_r205_terminal_verification_20261011'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not EXIT.exists() and not LOG.exists() and not PROOF.exists() and not DEST.exists()
before=digest(SOURCE);wrapper=digest(__file__)
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu',PYTHONDONTWRITEBYTECODE='1')
started=time.monotonic()
with LOG.open('xb') as log:
    result=subprocess.run([str(PROJECT/'.venv/bin/python'),'-B',str(SOURCE)],cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT)
elapsed=time.monotonic()-started
assert digest(SOURCE)==before and digest(__file__)==wrapper
receipt=dict(actual_subprocess_return_observed=True,natural_exit=True,exit_code=result.returncode,elapsed_seconds=elapsed,
    source_unchanged=True,source_sha256=before,wrapper_sha256=wrapper,proof_saved=PROOF.exists(),new_dynamic_attempts=0,
    log_sha256=digest(LOG),protected_live_server_not_operated=True)
if PROOF.exists():receipt['proof_sha256']=digest(PROOF)
with EXIT.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
raise SystemExit(result.returncode)
