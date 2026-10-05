import unittest
import mujoco
import numpy as np
from diagnostics.audit_getup_sensor_geometry_r137 import SensorGeometry,SCENE


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.fk=SensorGeometry(mujoco.MjModel.from_xml_path(str(SCENE)))

    def test_sensor_only_causality_and_nominal_zero(self):
        x=np.zeros(55,dtype=np.float32);y=x.copy();y[:6]=123.;y[34:]=12.
        np.testing.assert_array_equal(self.fk.risk(x),self.fk.risk(y))
        np.testing.assert_array_equal(self.fk.risk(x)-self.fk.risk(x),np.zeros(3))
        self.assertRaises(ValueError,self.fk.risk,np.zeros(56))
        self.assertRaises(ValueError,self.fk.risk,np.full(55,np.nan))
        self.assertRaises(ValueError,self.fk.risk,x,.04)

    def test_joint_order_velocity_scaling_and_independent_state(self):
        x=np.zeros(50,dtype=np.float32);x[6:20]=np.arange(14)*.001;x[20:34]=.05
        np.testing.assert_allclose(self.fk.pose(x,.02)[self.fk.qadr]-self.fk.pose(x,0)[self.fk.qadr],.02,atol=1e-8)
        separate=mujoco.MjData(self.fk.model);before=separate.qpos.copy();self.fk.risk(x)
        np.testing.assert_array_equal(separate.qpos,before)
        np.testing.assert_array_equal(self.fk.risk(x),self.fk.risk(x))


if __name__=='__main__':unittest.main()
