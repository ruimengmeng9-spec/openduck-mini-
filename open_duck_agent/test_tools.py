from __future__ import annotations

import unittest

from open_duck_agent.tools import RobotTools, TOOL_DEFINITIONS


class FakeController:
    def __init__(self) -> None:
        self.calls = []

    def stand(self, duration_s=1.0, source="tool"):
        self.calls.append(("stand", duration_s, source))
        return {"ok": True, "action": "stand"}

    def walk(self, speed_mps, duration_s):
        self.calls.append(("walk", speed_mps, duration_s))
        return {"ok": True, "action": "walk"}

    def turn(self, yaw_rate_rad_s, duration_s):
        self.calls.append(("turn", yaw_rate_rad_s, duration_s))
        return {"ok": True, "action": "turn"}

    def stop(self, settle_s=0.5, source="tool"):
        self.calls.append(("stop", settle_s, source))
        return {"ok": True, "action": "stop"}

    def status(self):
        self.calls.append(("status",))
        return {"fallen": False}


class RobotToolsTest(unittest.TestCase):
    def setUp(self):
        self.controller = FakeController()
        self.tools = RobotTools(self.controller)

    def test_all_tool_names_are_unique(self):
        names = [tool["name"] for tool in TOOL_DEFINITIONS]
        self.assertEqual(len(names), len(set(names)))

    def test_dispatches_the_four_motion_actions(self):
        self.tools.call("stand", {"duration_s": 1.0})
        self.tools.call("walk", {"speed_mps": 0.1, "duration_s": 2.0})
        self.tools.call("turn", {"yaw_rate_rad_s": -0.2, "duration_s": 1.0})
        self.tools.call("stop", {})
        self.assertEqual([call[0] for call in self.controller.calls], [
            "stand", "walk", "turn", "stop"
        ])

    def test_rejects_unknown_tool(self):
        with self.assertRaises(ValueError):
            self.tools.call("set_joint_torque", {"joint": 1, "torque": 100})


if __name__ == "__main__":
    unittest.main()
