import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np

from diagnostics.train_getup_residual_r18 import residual_decoder,export_actor


class ResidualTests(unittest.TestCase):
    def test_zero_latent_home(self):
        low=np.full(14,-1.);high=np.full(14,2.);home=np.linspace(-.6,.8,14)
        a=residual_decoder(np.zeros(14),home,low,high,.15)
        np.testing.assert_allclose((low+high)/2+a*(high-low)/2,home)

    def test_extreme_latents_stay_in_joint_and_residual_limits(self):
        low=np.full(14,-1.);high=np.full(14,1.);home=np.linspace(-.95,.95,14)
        for z in (np.full(14,1e6),np.full(14,-1e6)):
            a=residual_decoder(z,home,low,high,.15)
            self.assertTrue(np.all(np.abs(a)<=1))
            self.assertLessEqual(np.abs(a-home).max(),.15+1e-12)

    def test_actual_onnx_residual_decoder(self):
        import onnxruntime as ort
        rng=np.random.default_rng(118);sizes=(50,128,64,14)
        params={name:dict(kernel=rng.normal(0,.02,(sizes[i],sizes[i+1])).astype(np.float32),
                          bias=rng.normal(0,.02,sizes[i+1]).astype(np.float32))
                for i,name in enumerate(('hidden0','hidden1','mean'))}
        home=np.linspace(-.5,.5,14);low=np.full(14,-1.);high=np.full(14,1.)
        obs=rng.normal(size=(4,50)).astype(np.float32);x=obs
        for name in ('hidden0','hidden1'):
            x=np.tanh(x@params[name]['kernel']+params[name]['bias'])
        z=x@params['mean']['kernel']+params['mean']['bias']
        with TemporaryDirectory() as directory:
            path=Path(directory)/'policy.onnx';export_actor(params,path,home,low,high,.15)
            options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
            actual=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider']).run(None,{'obs':obs})[0]
        np.testing.assert_allclose(actual,residual_decoder(z,home,low,high,.15),atol=1e-6)


if __name__=='__main__':unittest.main()
