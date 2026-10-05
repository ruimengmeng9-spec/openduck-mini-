"""Unique simulation-only R110 launch, after independent smoke succeeds."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT=Path('/data/shijinsheng/open_duck')


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    module='diagnostics.train_getup_distill_r110'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        if module.encode() in cmd:raise RuntimeError('R110 already running; do not duplicate')
    if not args.smoke:
        report=json.loads((ROOT/'outputs/getup_distill_r110_smoke_20261005/results.json').read_text())
        assert report['smoke'] and report['nominal']['nominal_success']
        assert report['nominal']['rows'][0]['original_full_path_bitwise_parity']
        assert len(report['zero_actor_parity'])==2
        assert all(r['original_full_path_bitwise_parity'] and r['valid'] for r in report['zero_actor_parity'])
    output=ROOT/'outputs'/('getup_distill_r110_smoke_20261005' if args.smoke else 'getup_distill_r110_left_20261005')
    log=output.with_suffix('.log');assert not output.exists() and not log.exists()
    cwd=ROOT/'projects/Open_Duck_Playground';env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    command=[str(cwd/'.venv/bin/python'),'-u','-m',module,'--output',str(output),'--workers','6']
    if args.smoke:command.append('--smoke')
    with log.open('xb') as stream:
        child=subprocess.Popen(command,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,command=command,log=str(log),simulation_only=True)))


if __name__=='__main__':main()
