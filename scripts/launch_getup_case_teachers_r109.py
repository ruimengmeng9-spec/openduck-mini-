"""Launch unique offline simulation teacher search after completed smoke."""
import json
import os
from pathlib import Path
import subprocess
from diagnostics.probe_getup_anchor_r101 import ROOT


def main():
    smoke=ROOT/'outputs/getup_case_teachers_r109_smoke_20261005'
    assert json.loads((smoke/'results.json').read_text())['smoke']
    parity=json.loads((smoke/'parity.json').read_text())
    assert len(parity)==4 and all(r['original_controller_full_path_bitwise_parity'] for r in parity)
    module='diagnostics.search_getup_case_teachers_r109'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes().split(b'\0')
        except (PermissionError,FileNotFoundError,ProcessLookupError):continue
        if module.encode() in cmd:raise RuntimeError('R109 still active; do not duplicate')
    output=ROOT/'outputs/getup_case_teachers_r109_left_20261005'
    logfile=output.with_suffix('.log')
    assert not output.exists() and not logfile.exists()
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES='',JAX_PLATFORMS='cpu')
    cwd=ROOT/'projects/Open_Duck_Playground'
    with logfile.open('xb') as log:
        child=subprocess.Popen([str(cwd/'.venv/bin/python'),'-u','-m',module,'--output',str(output),
            '--workers','6','--generations','24','--population','14'],cwd=cwd,env=env,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(pid=child.pid,module=module,output=str(output),log=str(logfile),
        simulation_only=True,teacher_data_only=True)))


if __name__=='__main__':main()
