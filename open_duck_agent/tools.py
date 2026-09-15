"""Tool schemas and dispatch layer shared by the LLM agent and tests."""

from __future__ import annotations

from typing import Any, Protocol


class RobotController(Protocol):
    def stand(self, duration_s: float = 1.0, source: str = "tool") -> dict[str, Any]: ...
    def walk(self, speed_mps: float, duration_s: float) -> dict[str, Any]: ...
    def turn(self, yaw_rate_rad_s: float, duration_s: float) -> dict[str, Any]: ...
    def stop(self, settle_s: float = 0.5, source: str = "tool") -> dict[str, Any]: ...
    def status(self) -> dict[str, Any]: ...


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "name": "stand",
        "description": "Stand and balance in place for a short duration.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "duration_s": {
                    "type": "number",
                    "description": "Duration in seconds, from 0.2 to 5.0.",
                    "minimum": 0.2,
                    "maximum": 5.0,
                }
            },
            "required": ["duration_s"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "walk",
        "description": "Walk at a bounded forward/backward speed for a short duration.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "speed_mps": {
                    "type": "number",
                    "description": "Forward speed in m/s; negative means backward.",
                    "minimum": -0.15,
                    "maximum": 0.15,
                },
                "duration_s": {
                    "type": "number",
                    "minimum": 0.2,
                    "maximum": 5.0,
                },
            },
            "required": ["speed_mps", "duration_s"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "turn",
        "description": "Turn in place. Positive and negative values turn opposite ways.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "yaw_rate_rad_s": {
                    "type": "number",
                    "description": "Requested yaw rate in rad/s.",
                    "minimum": -0.3,
                    "maximum": 0.3,
                },
                "duration_s": {
                    "type": "number",
                    "minimum": 0.2,
                    "maximum": 5.0,
                },
            },
            "required": ["yaw_rate_rad_s", "duration_s"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "stop",
        "description": "Set the motion command to zero and settle safely.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "get_status",
        "description": "Read the current simulation pose and safety status.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
]


class RobotTools:
    """Small allow-listed API exposed to the language model."""

    def __init__(self, controller: RobotController) -> None:
        self.controller = controller

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "stand":
            return self.controller.stand(duration_s=arguments["duration_s"])
        if name == "walk":
            return self.controller.walk(
                speed_mps=arguments["speed_mps"],
                duration_s=arguments["duration_s"],
            )
        if name == "turn":
            return self.controller.turn(
                yaw_rate_rad_s=arguments["yaw_rate_rad_s"],
                duration_s=arguments["duration_s"],
            )
        if name == "stop":
            return self.controller.stop()
        if name == "get_status":
            return self.controller.status()
        raise ValueError(f"Unknown or disallowed robot tool: {name}")
