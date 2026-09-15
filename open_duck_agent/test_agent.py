from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from open_duck_agent.agent import execute_local_plan, run_agent_turn, validate_local_plan
from open_duck_agent.test_tools import FakeController
from open_duck_agent.tools import RobotTools


class FakeResponses:
    def __init__(self):
        self.requests = []
        self.reply_index = 0

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.reply_index == 0:
            output = [
                SimpleNamespace(
                    type="function_call",
                    name="walk",
                    arguments=json.dumps({"speed_mps": 0.1, "duration_s": 2.0}),
                    call_id="call_1",
                )
            ]
            result = SimpleNamespace(output=output, output_text="")
        else:
            result = SimpleNamespace(output=[], output_text="已前进并停止。")
        self.reply_index += 1
        return result


class AgentLoopTest(unittest.TestCase):
    def test_function_call_round_trip_and_enforced_stop(self):
        controller = FakeController()
        responses = FakeResponses()
        client = SimpleNamespace(responses=responses)
        answer = run_agent_turn(
            client=client,
            model="test-model",
            tools=RobotTools(controller),
            user_text="前进两秒",
        )

        self.assertEqual(answer, "已前进并停止。")
        self.assertEqual(controller.calls[0], ("walk", 0.1, 2.0))
        self.assertEqual(controller.calls[-1][0], "stop")
        second_input = responses.requests[1]["input"]
        tool_outputs = [
            item for item in second_input
            if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        self.assertEqual(tool_outputs[0]["call_id"], "call_1")
        self.assertFalse(responses.requests[0]["parallel_tool_calls"])

    def test_local_plan_is_clamped_and_stopped(self):
        actions = validate_local_plan(
            {
                "actions": [
                    {"action": "walk", "speed_mps": 9, "duration_s": 99},
                    {"action": "turn", "value": -9, "duration_s": 0.01},
                ]
            }
        )
        self.assertEqual(
            actions,
            [
                ("walk", {"speed_mps": 0.15, "duration_s": 5.0}),
                ("turn", {"yaw_rate_rad_s": -0.3, "duration_s": 0.2}),
                ("stop", {}),
            ],
        )

        controller = FakeController()
        report = execute_local_plan(RobotTools(controller), actions)
        self.assertTrue(report["ok"])
        self.assertEqual(controller.calls[-2][0], "stop")
        self.assertEqual(controller.calls[-1][0], "status")

    def test_local_plan_rejects_joint_control(self):
        with self.assertRaises(ValueError):
            validate_local_plan(
                {"actions": [{"action": "set_joint_torque", "value": 100}]}
            )


if __name__ == "__main__":
    unittest.main()
