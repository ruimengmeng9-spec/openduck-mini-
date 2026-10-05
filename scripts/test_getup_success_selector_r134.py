import unittest
import numpy as np
from diagnostics.train_getup_success_selector_r134 import fit,predict,local


class SelectorTests(unittest.TestCase):
    def model(self):
        x=np.arange(200,dtype=float).reshape(4,50)/100
        y=np.eye(4);return {**fit(x,y,1.),'global_gains':np.arange(24,dtype=float).reshape(4,6)/100}

    def test_sensor_only_scalar(self):
        model=self.model();gains,choice,logits=predict(model,np.zeros(50))
        self.assertEqual(gains.shape,(6,));self.assertEqual(logits.shape,(4,))
        np.testing.assert_array_equal(gains,model['global_gains'][choice])
        for sensors in (np.zeros(51),np.full(50,np.nan)):
            with self.assertRaises(ValueError):predict(model,sensors)

    def test_export_has_no_training_context_table(self):
        self.assertEqual(set(self.model()),{'mean','std','weight','bias','global_gains'})
        self.assertEqual(self.model()['weight'].shape,(50,4))

    def test_multiple_success_labels_preserved(self):
        x=np.zeros((5,50));y=np.tile([1.,1.,0.,0.],(5,1));model={**fit(x,y,10.),'global_gains':np.zeros((4,6))}
        _,_,logits=predict(model,np.zeros(50));np.testing.assert_array_equal(logits,[1.,1.,0.,0.])

    def test_nominal_feedback_zero_for_every_global_program(self):
        obs=np.linspace(-1,1,55,dtype=np.float32)
        for gains in self.model()['global_gains']:
            np.testing.assert_array_equal(local.local_feedback(obs,obs,[9,10,11],gains),np.zeros(3))


if __name__=='__main__':unittest.main()
