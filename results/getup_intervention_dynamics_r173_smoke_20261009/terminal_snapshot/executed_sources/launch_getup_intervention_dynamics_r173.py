"""No signals; launch only unique simulation experiment after closed smoke."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np
from diagnostics import train_getup_intervention_dynamics_r173 as run

def main():
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');args=p.parse_args()
    lines=subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
    assert not any(' -m diagnostics.' in l and any(t in l for t in ('train_getup_','probe_getup_','audit_getup_')) for l in lines),'A diagnostic is active; wait without signals'
    for folder in (run.ROOT/'outputs').iterdir():
        match=re.search(r'_r(\d+)',folder.name)
        assert not match or int(match[1])<=173,'A higher task exists; inspect it before starting'
    output=run.SMOKE if args.smoke else run.OUTPUT
    assert not output.exists() and not output.with_suffix('.log').exists()
    assert shutil.disk_usage(run.ROOT).free>8*1024**3
    if not args.smoke:
        smoke=json.loads((run.SMOKE/'results.json').read_text())
        assert smoke['smoke'] and smoke['terminal_result_saved']
        assert len(smoke['parity']['rows'])==len(smoke['nonzero']['rows'])==3
        assert smoke['parity']['nominal_success'] and not smoke['parity']['physical_failures']
        assert smoke['nonzero']['nominal_success']
        for row in smoke['parity']['rows']:assert all(row['complete_R157_parity'].values())
        for case in (None,769002,773004):
            with np.load(run.SMOKE/'nonzero_smoke'/f'case_{case}'/'trajectory.npz',allow_pickle=False) as z:
                np.testing.assert_array_equal(z['planned_before_integration_rad'],z['applied'])
                np.testing.assert_array_equal(z['intervention_requested_extra_rad'][0],np.zeros(3))
                if case is None:assert not np.any(z['intervention_requested_extra_rad'])
                else:assert np.any(z['intervention_requested_extra_rad'])
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
    stage='smoke' if args.smoke else 'formal'
    regression=run.ROOT/f'tmp/getup_intervention_dynamics_r173_regression_{stage}_20261009.log'
    assert not regression.exists()
    with regression.open('w') as f:
        subprocess.run([sys.executable,'-u','-m','diagnostics.test_getup_intervention_dynamics_r173'],stdout=f,stderr=subprocess.STDOUT,env=env,check=True)
    command=[sys.executable,'-u','-m','diagnostics.train_getup_intervention_dynamics_r173']
    if args.smoke:command.append('--smoke')
    with output.with_suffix('.log').open('w') as f:
        process=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    print(json.dumps(dict(pid=process.pid,command=command,log=str(output.with_suffix('.log')),simulation_only=True)),flush=True)

if __name__=='__main__':main()
