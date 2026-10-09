"""Unique offline contract run; never starts a dynamic episode or controller."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from diagnostics import probe_getup_phase_coordinate_r185b as old

ROOT = old.ROOT
OUTPUT = ROOT/'outputs/getup_centroidal_momentum_kernel_r187b_20261010'


def main():
    commands = subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True).splitlines()
    assert not any(' -m diagnostics.' in v and any(t in v for t in ('train_getup_','probe_getup_','audit_getup_')) for v in commands)
    assert not OUTPUT.exists() and shutil.disk_usage(ROOT).free > 10.25*1024**3
    OUTPUT.mkdir(); source=OUTPUT/'executed_sources';source.mkdir()
    paths=[Path(__file__).with_name(n) for n in ('centroidal_momentum_kernel_r187b.py','test_centroidal_momentum_kernel_r187b.py',Path(__file__).name)]
    for path in paths:shutil.copy2(path,source/path.name)
    before={str(p):old.prior.digest(p) for p in paths}
    input_path=old.local.prior.OUTPUT/'snapshot/case_None/trajectory.npz'
    before[str(input_path)]=old.prior.digest(input_path)
    before[str(old.local.prior.program.SCENE)]=old.prior.digest(old.local.prior.program.SCENE)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','JAX_PLATFORMS':'cpu'}
    with (OUTPUT/'regression.log').open('x') as stream:
        result=subprocess.run([sys.executable,'-m','diagnostics.test_centroidal_momentum_kernel_r187b'],env=env,stdout=stream,stderr=subprocess.STDOUT)
    after={p:old.prior.digest(Path(p)) for p in before}
    assert before==after
    import mujoco
    old.local.write_json(OUTPUT/'results.json',dict(terminal_result_saved=True,offline_kernel_validation_only=True,
        tests_passed=result.returncode==0,regression_exit_code=result.returncode,regression_tests=15,
        new_dynamic_attempts=0,new_controller_run=False,full_task_completed=False,hardware_readiness=False,
        expanded_development_run=False,independent_qualification_run=False,mujoco_version=mujoco.__version__,
        source_input_hashes_before=before,source_input_hashes_after=after,storage_budget_gb=.25,reserve_gb=10,
        prospective_fixed_dynamic_budget=62,prospective_formal_attempts=56,prospective_independent_full_smoke=6))
    print(json.dumps(dict(output=str(OUTPUT),exit_code=result.returncode,new_dynamic_attempts=0)),flush=True)
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
