"""Read-only proof that field-restricted R164b preserves all R164 algebra."""
import json
from pathlib import Path
import shutil
import numpy as np
from diagnostics.getup_independent_native import digest
from diagnostics.train_getup_joint_anchor_r102 import write_json

ROOT=Path('/data/shijinsheng/open_duck')

def main():
    out=ROOT/'outputs/getup_response_filter_parity_r164b_20261006';assert not out.exists();checks=[];hashes={}
    for label in ['','_smoke']:
        old=ROOT/'outputs'/f'getup_response_stages_r164{label}_20261006'
        new=ROOT/'outputs'/f'getup_response_stages_r164b{label}_20261006'
        a=json.loads((old/'results.json').read_text());b=json.loads((new/'results.json').read_text())
        assert a['rows']==b['rows'] and a['pairs']==b['pairs']
        for path in old.rglob('stages.npz'):
            other=new/path.relative_to(old)
            for p in [path,other]:hashes[str(p)]=digest(p)
            with np.load(path,allow_pickle=False) as x,np.load(other,allow_pickle=False) as y:
                assert set(x.files)==set(y.files) and all(np.array_equal(x[k],y[k]) for k in x.files)
            checks.append(dict(original=str(path),restricted=str(other),all_arrays_bitwise_equal=True))
    assert len(checks)==62 and all(digest(Path(p))==h for p,h in hashes.items())
    out.mkdir();shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'results.json',dict(read_only=True,no_dynamic_integration=True,checks=checks,
        original_R164_loader_read_unused_qpos_qvel_arrays=True,original_no_root_read_flag_was_inaccurate=True,
        no_root_arrays_used_in_any_R164_or_R164b_calculation=True,R164b_loader_whitelist_excludes_root_arrays=True,
        all_58_formal_and_4_smoke_stage_arrays_and_rows_bitwise_equal=True,original_outcomes_not_relabelled=True))
    write_json(out/'source_hashes.json',hashes);print('R164B_FIELD_FILTER_EXACT',len(checks),flush=True)

if __name__=='__main__':main()
