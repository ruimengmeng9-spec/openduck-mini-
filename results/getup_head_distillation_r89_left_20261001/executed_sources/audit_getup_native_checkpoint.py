"""Verify saved recovery parameters against actual exported ONNX artifacts."""
import argparse
import json
from pathlib import Path

from flax import serialization
import numpy as np
import onnxruntime as ort

from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',type=Path,required=True)
    args=p.parse_args()
    contract=json.loads((args.experiment/'controller_contract.json').read_text())
    options=ort.SessionOptions()
    options.intra_op_num_threads=options.inter_op_num_threads=1
    obs=np.random.default_rng(105).normal(0,.5,(32,50)).astype(np.float32)
    rows=[]
    for kind in ('initial','final'):
        saved=args.experiment/f'{kind}.msgpack'
        exported=args.experiment/f'{kind}.onnx'
        params=serialization.msgpack_restore(saved.read_bytes())['actor']
        expected=obs
        for layer in ('hidden0','hidden1','mean'):
            expected=np.tanh(expected@params[layer]['kernel']+params[layer]['bias'])
        session=ort.InferenceSession(str(exported),sess_options=options,providers=['CPUExecutionProvider'])
        actual=session.run(None,{'obs':obs})[0]
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
        if not np.isfinite(actual).all() or np.max(np.abs(actual))>1.:
            raise RuntimeError('invalid exported action')
        rows.append(dict(kind=kind,max_absolute_error=float(np.max(np.abs(expected-actual))),
                         normalized_action_max_abs=float(np.max(np.abs(actual))),
                         parameters_sha256=digest(saved),onnx_sha256=digest(exported)))
    output=args.experiment/'checkpoint_audit.json'
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(dict(rows=rows,observation_size=50,action_size=14,
                      interchangeable_with_101_dim_walking_actor=False,simulation_only=True,
                      hardware_readiness=False,script_sha256=digest(__file__),
                      contract_sha256=digest(args.experiment/'controller_contract.json')),indent=2),encoding='utf-8')
    print('CHECKPOINT PARITY:',json.dumps(rows),flush=True)


if __name__=='__main__': main()
