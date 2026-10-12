"""Observe real unique R205 archive preparation or commit exit; preserve full log."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('/data/shijinsheng/open_duck')
PROJECT=ROOT/'projects/Open_Duck_Playground'
parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
source=PROJECT/'diagnostics/archive_getup_total_target_cacla_terminal_r205.py'
stem='getup_total_target_cacla_r205_archive'+('_preparation' if args.prepare else '')
log=ROOT/'tmp'/(stem+'_20261012.log')
receipt=ROOT/'tmp'/(stem+'_exit_20261012.json')
result=ROOT/'tmp/getup_total_target_cacla_r205_archive_result_20261012.json'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
before=digest(source);wrapper_before=digest(__file__)
assert not log.exists() and not receipt.exists() and not result.exists()
start=time.monotonic()
command=[str(PROJECT/'.venv/bin/python'),'-B','-u',str(source)]+(['--prepare'] if args.prepare else [])
with log.open('x') as f:
    proc=subprocess.Popen(command,cwd=PROJECT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:
        f.write(line);f.flush();print(line,end='',flush=True)
    code=proc.wait()
out=dict(actual_subprocess_return_observed=True,natural_exit=True,exit_code=code,elapsed_seconds=time.monotonic()-start,stage='prepare' if args.prepare else 'archive',source_sha256=before,wrapper_sha256=wrapper_before,source_hashes_unchanged=digest(source)==before and digest(__file__)==wrapper_before,log_sha256=digest(log),archive_result_saved=result.exists(),new_dynamic_attempts=0,verifier_A_or_B_not_executed=True,protected_live_server_not_operated=True,storage_cap_bytes=4294967296)
with receipt.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out),flush=True)
sys.exit(code)

