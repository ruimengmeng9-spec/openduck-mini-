"""Read-only batch/scalar anchor consistency diagnostic."""
import json
from pathlib import Path
import numpy as np
from diagnostics.getup_reference_env_r100 import numpy_action

root=Path('/data/shijinsheng/open_duck/outputs')
run=root/'getup_nominal_anchor_r101_left_20261005'
with np.load(run/'nominal_anchor.npz') as data:
    obs=data['observations'].copy();stored=data['actions'].copy()
with np.load(root/'getup_reference_residual_r100_left_20261005/training/final.npz') as data:
    weights={k:data[k].copy() for k in data.files}
scalar=np.stack([numpy_action(weights,row) for row in obs])
with np.load(run/'anchored_1.0/case_None.npz') as data:
    actual=np.abs(data['normalized_residual']).max()
print(json.dumps(dict(batch_vs_scalar_max_abs=float(np.abs(stored-scalar).max()),
                      nominal_residual_max_abs=float(actual),
                      exact_zero_assertion_valid=bool(actual==0))),flush=True)
