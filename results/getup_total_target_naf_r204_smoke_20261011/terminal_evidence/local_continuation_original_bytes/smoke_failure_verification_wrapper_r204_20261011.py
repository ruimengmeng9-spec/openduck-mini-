"""One new readonly R204 full-verification subprocess and real exit receipt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import shutil
ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
SOURCE=PROJECT/'diagnostics/verify_getup_total_target_naf_smoke_failure_r204.py'
SHA='36681b28130cde79632aab2cc81adf48d5d6493e166dd0b08dac99f18f9db130'
LOG=ROOT/'tmp/getup_total_target_naf_r204_smoke_failure_verification_20261011.log'
EXIT=ROOT/'tmp/getup_total_target_naf_r204_smoke_failure_verification_exit_20261011.json'
PROOF=ROOT/'tmp/getup_total_target_naf_r204_smoke_failure_verification_20261011.json'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert shutil.disk_usage(ROOT).free>=12*1024**3
assert digest(SOURCE)==SHA
assert not LOG.exists() and not EXIT.exists() and not PROOF.exists()
started=time.monotonic()
with LOG.open('x') as stream:
    child=subprocess.run([str(PROJECT/'.venv/bin/python'),'-u','-m','diagnostics.verify_getup_total_target_naf_smoke_failure_r204'],
        cwd=PROJECT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'},stdout=stream,stderr=subprocess.STDOUT)
result=dict(natural_exit=True,actual_subprocess_return_observed=True,exit_code=child.returncode,
    elapsed_seconds=time.monotonic()-started,source_unchanged=digest(SOURCE)==SHA,source_sha256=SHA,
    new_dynamic_attempts=0,proof_saved=PROOF.exists(),proof_sha256=digest(PROOF) if PROOF.exists() else None,
    log=str(LOG),log_sha256=digest(LOG),protected_live_server_not_operated=True)
with EXIT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result),flush=True)
assert result['source_unchanged']
raise SystemExit(child.returncode)
