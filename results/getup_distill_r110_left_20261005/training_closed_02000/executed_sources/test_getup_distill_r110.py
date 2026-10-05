import unittest
import numpy as np
from diagnostics.train_getup_distill_r110 import causal_input, normalize, policy_action, load_data, RECOVERY


class DistillationTest(unittest.TestCase):
    def weights(self):
        rng=np.random.default_rng(210)
        w=dict(input_mean=np.zeros(110,dtype=np.float32),input_std=np.ones(110,dtype=np.float32),
            nominal_observations=rng.normal(size=(2279,55)).astype(np.float32))
        for name,shape in [('hidden0',(110,128)),('hidden1',(128,64)),('mean',(64,10))]:
            w[name+'_kernel']=rng.normal(0,.1,size=shape).astype(np.float32);w[name+'_bias']=rng.normal(0,.1,size=shape[1]).astype(np.float32)
        return w

    def test_causal_inputs_and_bad_dimensions(self):
        a=np.ones(55,dtype=np.float32);b=np.zeros(55,dtype=np.float32)
        np.testing.assert_array_equal(causal_input(a,b),np.concatenate((a,b)))
        for wrong in (np.ones(56),np.full(55,np.nan)):
            with self.assertRaises(ValueError):causal_input(wrong,b)

    def test_actual_scalar_nominal_exact_zero(self):
        w=self.weights();nom=w['nominal_observations']
        for k in range(RECOVERY):np.testing.assert_array_equal(policy_action(w,nom[k],nom[0],k),np.zeros(10))

    def test_home_and_action_bounds(self):
        w=self.weights();obs=np.ones(55,dtype=np.float32)*1e6
        action=policy_action(w,obs,obs,1);self.assertLessEqual(np.abs(action).max(),1.)
        np.testing.assert_array_equal(policy_action(w,obs,obs,529),np.zeros(10))
        with self.assertRaises(ValueError):policy_action(w,obs,obs,2279)

    def test_data_and_normalization_training_only(self):
        x,y,a,mean,std,nom,metadata=load_data()
        self.assertEqual(x.shape,(25*529,110));self.assertEqual(y.shape,(25*529,10))
        np.testing.assert_array_equal(mean,x.mean(0,dtype=np.float64).astype(np.float32))
        self.assertEqual(len(metadata),25);self.assertTrue(np.all(std>=1e-4))
        for block in range(25):
            chunk=x[block*529:(block+1)*529]
            np.testing.assert_array_equal(chunk[:,55:],np.repeat(chunk[:1,55:],529,axis=0))
        np.testing.assert_array_equal(a[:,55:],np.repeat(nom[:1],529,axis=0))
        self.assertTrue(np.isfinite(normalize(x,mean,std)).all())


if __name__=='__main__':unittest.main()
