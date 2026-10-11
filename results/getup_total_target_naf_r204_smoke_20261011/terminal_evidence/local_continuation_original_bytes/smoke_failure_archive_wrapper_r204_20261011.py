"""Observe unique new R204 archive natural exit, retaining complete output."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
source=PROJECT/'diagnostics/archive_getup_total_target_naf_smoke_failure_r204.py'
log=ROOT/'tmp/getup_total_target_naf_r204_smoke_failure_archive_20261011.log'
receipt=ROOT/'tmp/getup_total_target_naf_r204_smoke_failure_archive_exit_20261011.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not log.exists() and not receipt.exists()
before=sha(source);start=time.monotonic()
with log.open('x') as stream:
    proc=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-u','-m','diagnostics.archive_getup_total_target_naf_smoke_failure_r204'],cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
    code=proc.wait()
result=dict(natural_exit=True,actual_subprocess_return_observed=True,exit_code=code,elapsed_s=time.monotonic()-start,source_sha256=before,source_hash_unchanged=before==sha(source),new_dynamic_attempts=0,archive_result_saved=(ROOT/'tmp/getup_total_target_naf_r204_smoke_failure_archive_result_20261011.json').exists())
with receipt.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result),flush=True)
raise SystemExit(code)
