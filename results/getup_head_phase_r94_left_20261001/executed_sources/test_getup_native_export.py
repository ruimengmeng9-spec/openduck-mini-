"""Numerical parity of the independent recovery ONNX export."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from diagnostics.train_getup_native_ppo_v2 import export_actor


class NativeExportTests(unittest.TestCase):
    def test_export_matches_numpy_actor_and_stays_bounded(self):
        import onnxruntime as ort
        rng=np.random.default_rng(91)
        params={name:dict(kernel=rng.normal(0,.03,(n,m)).astype(np.float32),
                          bias=rng.normal(0,.03,m).astype(np.float32))
                for name,n,m in [('hidden0',50,128),('hidden1',128,64),('mean',64,14)]}
        obs=rng.normal(size=(4,50)).astype(np.float32)
        expected=obs
        for name in ('hidden0','hidden1','mean'):
            expected=np.tanh(expected@params[name]['kernel']+params[name]['bias'])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'actor.onnx'
            export_actor(params,path)
            options=ort.SessionOptions()
            options.intra_op_num_threads=options.inter_op_num_threads=1
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
            actual=session.run(None,{'obs':obs})[0]
            np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-5)
            self.assertTrue(np.isfinite(actual).all())
            self.assertLessEqual(float(np.max(np.abs(actual))),1.)


if __name__=='__main__': unittest.main()
