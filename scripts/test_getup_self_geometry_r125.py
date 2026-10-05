import unittest
from unittest.mock import patch
import mujoco
import numpy as np
from diagnostics import audit_getup_self_geometry_r125 as audit


class GeometryRegression(unittest.TestCase):
    def test_forward_is_not_integration_and_reproducible(self):
        m=mujoco.MjModel.from_xml_path(str(audit.SCENE))
        a=mujoco.MjData(m);b=mujoco.MjData(m)
        q=m.qpos0.copy();v=np.zeros(m.nv);floor=m.geom('floor').id
        with patch.object(mujoco,'mj_step',side_effect=AssertionError('No integration allowed')):
            one=audit.geometry(m,a,floor,q,v);two=audit.geometry(m,b,floor,q,v)
        self.assertEqual(one,two);np.testing.assert_array_equal(a.qpos,q)
        self.assertEqual(a.time,0.)

    def test_self_geometry_rigid_root_transform(self):
        m=mujoco.MjModel.from_xml_path(str(audit.SCENE));d=mujoco.MjData(m)
        q=m.qpos0.copy();v=np.zeros(m.nv);floor=m.geom('floor').id
        one=audit.geometry(m,d,floor,q,v)
        q[:3]=[1.,2.,10.];q[3:7]=[np.cos(.2),np.sin(.2),0.,0.]
        two=audit.geometry(m,d,floor,q,v)
        self.assertEqual(one.keys(),two.keys())
        self.assertLess(max([abs(one[p]-two[p]) for p in one] or [0.]),1e-8)


if __name__=='__main__':unittest.main()
