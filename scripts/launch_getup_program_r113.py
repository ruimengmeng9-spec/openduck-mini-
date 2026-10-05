"""Fresh bounded simulation launch; no hardware and no process signals."""
import argparse
import json
import os
from pathlib import Path
import subprocess
ROOT=Path('/data/shijinsheng/open_duck')


def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    module='diagnostics.train_getup_program_r113'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:command=(proc/'cmdline').read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        if any(m.encode() in command for m in (module,'diagnostics.train_getup_aggregate_r112')):
            raise RuntimeError('Related experiment still active; wait for natural exit')
    if not args.smoke:
        report=json.loads((ROOT/'outputs/getup_program_r113_smoke_20261005/results.json').read_text())
        assert report['nominal']['nominal_success'] and report['nominal']['rows'][0]['original_full_path_bitwise_parity']
        assert len(report['interface_teacher_parity'])==4
        assert all(r['original_full_path_bitwise_parity'] and r['valid'] for r in report['interface_teacher_parity'])
    output=ROOT/'outputs'/('getup_program_r113_smoke_20261005' if args.smoke else 'getup_program_r113_left_20261005')
    log=output.with_suffix('.log');assert not output.exists() and not log.exists()
    cwd=ROOT/'projects/Open_Duck_Playground';env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    cmd=[str(cwd/'.venv/bin/python'),'-u','-m',module,'--output',str(output)]
    if args.smoke:cmd.append('--smoke')
    with log.open('xb') as stream:
        child=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,command=cmd,log=str(log),simulation_only=True)))


if __name__=='__main__':main()
