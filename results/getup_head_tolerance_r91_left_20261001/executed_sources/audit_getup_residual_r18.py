"""Audit actual saved R18 parameters, ONNX parity and physical residual bound."""
import argparse
import json
from pathlib import Path

from flax import serialization
import numpy as np
import onnxruntime as ort

from diagnostics.train_getup_residual_r18 import residual_decoder
from diagnostics.getup_independent_native import digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True)
    args=p.parse_args();c=json.loads((args.experiment/'controller_contract.json').read_text())
    obs=np.random.default_rng(118).normal(0,.5,(64,50)).astype(np.float32)
    options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
    rows=[]
    for kind in ('initial','final'):
        saved=args.experiment/f'{kind}.msgpack';model=args.experiment/f'{kind}.onnx'
        params=serialization.msgpack_restore(saved.read_bytes())['actor']
        x=obs
        for name in ('hidden0','hidden1'):
            x=np.tanh(x@params[name]['kernel']+params[name]['bias'])
        z=x@params['mean']['kernel']+params['mean']['bias']
        expected=residual_decoder(z,c['home_rad'],c['lower_rad'],c['upper_rad'],c['residual_scale_rad'])
        actual=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider']).run(None,{'obs':obs})[0]
        np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6)
        target=(np.array(c['lower_rad'])+np.array(c['upper_rad']))/2+actual*(np.array(c['upper_rad'])-np.array(c['lower_rad']))/2
        max_residual=float(np.max(np.abs(target-c['home_rad'])))
        if not np.isfinite(actual).all() or max_residual>c['residual_scale_rad']+1e-6:
            raise RuntimeError('residual contract failed')
        rows.append(dict(kind=kind,max_absolute_error=float(np.max(np.abs(actual-expected))),
                         max_residual_rad=max_residual,onnx_sha256=digest(model),parameters_sha256=digest(saved)))
    out=args.experiment/'checkpoint_audit.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(dict(rows=rows,script_sha256=digest(__file__),simulation_only=True,
                                  hardware_readiness=False),indent=2),encoding='utf-8')
    print('RESIDUAL PARITY:',json.dumps(rows),flush=True)


if __name__=='__main__':main()
