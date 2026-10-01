import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
from diagnostics.audit_getup_micro_contact_r92 import Observer,onset


class ObserverTest(unittest.TestCase):
    def test_exactly_one_original_integration(self):
        model=SimpleNamespace(geom=lambda name:SimpleNamespace(id=0),body=lambda name:SimpleNamespace(id=1))
        data=SimpleNamespace(contact=[],time=1.)
        original=Mock(return_value='original');o=Observer(original,model,.8)
        self.assertEqual(o(model,data),'original');original.assert_called_once_with(model,data)
        self.assertEqual(len(o.rows),1);self.assertEqual(len(o.rows[0]),len(Observer.columns))

    def test_diagnostic_persistence(self):
        self.assertIsNone(onset([0,1,0,1],.5,2,.002))
        self.assertAlmostEqual(onset([0,1,1],.5,2,.002),.004)


if __name__=='__main__':unittest.main()
