"""Independent post-publication verification with original helper and no push."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
ROOT=Path('/data/shijinsheng/open_duck');PROJECT=ROOT/'projects/Open_Duck_Playground'
parser=argparse.ArgumentParser();parser.add_argument('--expected',required=True);args=parser.parse_args()
helper=ROOT/'tmp/push_openduck_server_20261009.py'
log=ROOT/'tmp/getup_total_target_vmpo_r202_interruption_publication_readonly_20261011.log'
receipt=ROOT/'tmp/getup_total_target_vmpo_r202_interruption_publication_readonly_20261011.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not log.exists() and not receipt.exists()
before=sha(helper);assert before=='a81a4328b8dcc73e565c0ccc196a7c2ecdf10aa55cc5b9442cd79a001f027d3f'
start=time.monotonic()
proc=subprocess.run([str(PROJECT/'.venv/bin/python'),'-u',str(helper),'--expected',args.expected],capture_output=True,text=True)
with log.open('x') as stream:stream.write(proc.stdout);stream.write(proc.stderr)
rows=[json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{')]
verified=proc.returncode==0 and len(rows)==2 and rows[0]['remote_before']==rows[0]['local_head']==rows[1]['remote_after']==args.expected and rows[1]['repository_clean']
result=dict(natural_exit=True,exit_code=proc.returncode,elapsed_s=time.monotonic()-start,expected_commit=args.expected,helper_source_sha256=before,helper_source_hash_unchanged=before==sha(helper),remote_before_after_local_head_exact_and_clean=verified,helper_stdout=proc.stdout,helper_stderr=proc.stderr,new_dynamic_attempts=0,no_push=True)
with receipt.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
print(json.dumps(result),flush=True)
raise SystemExit(0 if verified else 1)

