"""Unique R204 launch request; request exit is not launcher/smoke completion."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/data/shijinsheng/open_duck');PROJECT=ROOT/'projects/Open_Duck_Playground'
EXIT=ROOT/'tmp/getup_total_target_naf_r204_launch_request_exit_20261011.json'
LOG=ROOT/'tmp/getup_total_target_naf_r204_launch_request_20261011.log'
assert not EXIT.exists() and not LOG.exists()
names=['train_getup_total_target_naf_r204.py','total_target_naf_kernel_r204.py','test_getup_total_target_naf_r204.py','launch_getup_total_target_naf_r204.py']
sources=[PROJECT/'diagnostics'/n for n in names]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:6])
started=time.monotonic()
child=subprocess.run([str(PROJECT/'.venv/bin/python'),'-u','-m','diagnostics.launch_getup_total_target_naf_r204','--launch'],cwd=PROJECT,env=env,capture_output=True,text=True)
with LOG.open('x') as f:f.write(child.stdout);f.write(child.stderr)
unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in hashes.items())
receipt=dict(natural_exit=True,exit_code=child.returncode,elapsed_seconds=time.monotonic()-started,source_hashes_unchanged=unchanged,source_hashes=hashes,stdout=child.stdout,stderr=child.stderr,request_exit_not_launcher_or_smoke_completion=True)
with EXIT.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(receipt),flush=True)
assert unchanged
raise SystemExit(child.returncode)
