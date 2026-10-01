"""Correct inherited PPO-only metadata; preserve original contract and weights."""
import json
from pathlib import Path
from diagnostics.distill_getup_rescue_r23 import physical_contract
from diagnostics.getup_independent_native import digest


def main():
    root=Path('/data/shijinsheng/open_duck/training/getup_distill_r23')
    path=root/'controller_contract.json';old=json.loads(path.read_text())
    backup=root/'controller_contract_before_metadata_cleanup.json'
    if backup.exists():raise FileExistsError(backup)
    backup.write_bytes(path.read_bytes())
    oldhash=digest(path);modelhash=digest(root/'final.onnx')
    c=physical_contract(old)
    for key in ('stage','seed','policy_action_repeat','policy_period_s','residual_scale_rad',
        'action_decoder','training_method','independent_training_seed_list',
        'offline_validation_seed_list','default_controller_replaced','epochs','teacher_provenance'):
        c[key]=old[key]
    c.update(learning_rate=3e-4,optimizer='Adam, global gradient clip 1',batch_size=256,
        early_demonstration_weight=4.,early_window_s=1.2,
        source_contract_sha256=digest('/data/shijinsheng/open_duck/training/getup_alignment_r20_base/controller_contract.json'))
    path.write_text(json.dumps(c,indent=2),encoding='utf-8')
    if digest(root/'final.onnx')!=modelhash:raise RuntimeError('weights changed during metadata cleanup')
    record=dict(reason='copied physical contract also contained old PPO-only training fields',
        prior_contract_sha256=oldhash,corrected_contract_sha256=digest(path),
        final_onnx_sha256=modelhash,model_weights_changed=False,physical_parameters_changed=False)
    (root/'metadata_cleanup.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record))


if __name__=='__main__':main()
