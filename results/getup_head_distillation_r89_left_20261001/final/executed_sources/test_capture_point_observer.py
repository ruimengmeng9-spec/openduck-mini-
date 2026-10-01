import unittest
import mujoco
import numpy as np
from diagnostics.capture_point_observer import CapturePointObserver


class CaptureObserverTest(unittest.TestCase):
    def setUp(self):
        self.model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <body name="robot" pos="0 0 .2"><freejoint/>
        <geom type="sphere" size=".04" mass=".6"/>
        <site name="left_foot" pos="0 .05 -.2"/>
        <site name="right_foot" pos="0 -.05 -.2"/>
        </body></worldbody></mujoco>''')
        self.data=mujoco.MjData(self.model)
        self.observer=CapturePointObserver(self.model)

    def test_velocity_units_and_no_live_mutation(self):
        self.data.qvel[0]=-.08
        before=(self.data.qpos.copy(),self.data.qvel.copy(),self.data.sensordata.copy(),self.data.time)
        measured=self.observer.forward_offset(self.data)
        self.assertAlmostEqual(measured,-.08/np.sqrt(9.81/.2))
        np.testing.assert_array_equal(self.data.qpos,before[0])
        np.testing.assert_array_equal(self.data.qvel,before[1])
        np.testing.assert_array_equal(self.data.sensordata,before[2])
        self.assertEqual(self.data.time,before[3])

    def test_translation_and_heading_invariance(self):
        self.data.qvel[0]=-.08
        initial=self.observer.forward_offset(self.data)
        self.data.qpos[0:2]=[5,7]
        self.assertAlmostEqual(self.observer.forward_offset(self.data),initial)
        self.data.qpos[3:7]=[np.sqrt(.5),0,0,np.sqrt(.5)]
        self.data.qvel[:3]=[0,-.08,0]
        self.assertAlmostEqual(self.observer.forward_offset(self.data),initial)


if __name__=='__main__':
    unittest.main()
