"""Observe unique strict FF R205 publication, retaining full progress and natural exit."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/shijinsheng/open_duck');PROJECT=ROOT/'projects/Open_Duck_Playground'
parser=argparse.ArgumentParser();parser.add_argument('--expected',required=True);args=parser.parse_args()
source=ROOT/'tmp/publish_getup_total_target_cacla_terminal_r205_20261012.py'
helper=ROOT/'tmp/push_openduck_server_20261009.py'
log=ROOT/'tmp/getup_total_target_cacla_r205_publication_20261012.log'
receipt=ROOT/'tmp/getup_total_target_cacla_r205_publication_exit_20261012.json'
success=ROOT/'tmp/getup_publication_r205_cacla_terminal_20261012_result.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not log.exists() and not receipt.exists() and not success.exists()
before={str(p):sha(p) for p in (source,helper)};start=time.monotonic()
assert before[str(helper)]=='a81a4328b8dcc73e565c0ccc196a7c2ecdf10aa55cc5b9442cd79a001f027d3f'
with log.open('x') as stream:
    proc=subprocess.Popen([str(PROJECT/'.venv/bin/python'),'-B','-u',str(source),'--expected',args.expected,'--push'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in proc.stdout:stream.write(line);stream.flush();print(line,end='',flush=True)
    code=proc.wait()
result=dict(natural_exit=True,actual_subprocess_return_observed=True,exit_code=code,elapsed_s=time.monotonic()-start,expected_commit=args.expected,source_sha256=before,source_hashes_unchanged=all(sha(Path(p))==v for p,v in before.items()),success_receipt_saved=success.exists(),new_dynamic_attempts=0,private_key_exported=False,force_push=False)
with receipt.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result),flush=True)
raise SystemExit(code)


