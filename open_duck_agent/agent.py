"""High-level Open Duck Mini agent with OpenAI and local MiniCPM-o backends."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from open_duck_agent.tools import RobotTools, TOOL_DEFINITIONS


AGENT_INSTRUCTIONS = """
你是 Open Duck Mini 仿真机器人的高层动作规划器。
你只能使用提供的 stand、walk、turn、stop、get_status 工具，绝不能生成或执行关节级控制。
把用户的自然语言目标拆成少量、短时动作。每次动作不超过 5 秒；不确定时先选择较小速度和较短时间。
执行动作后检查工具返回值。若 fallen=true、ok=false 或状态异常，立即调用 stop，不再继续运动。
一次用户请求最多规划 8 个工具调用。完成任务后必须调用 stop，最后用简短中文报告实际执行的动作和最终状态。
""".strip()


def run_agent_turn(
    client: Any,
    model: str,
    tools: RobotTools,
    user_text: str,
    max_tool_calls: int = 8,
) -> str:
    """Run one user request, including any function-call round trips."""
    conversation: list[Any] = [{"role": "user", "content": user_text}]
    tool_call_count = 0

    for _ in range(max_tool_calls + 2):
        response = client.responses.create(
            model=model,
            instructions=AGENT_INSTRUCTIONS,
            tools=TOOL_DEFINITIONS,
            input=conversation,
            parallel_tool_calls=False,
        )
        conversation.extend(response.output)
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            # Enforce a zero motion command even if the model forgot the final stop.
            tools.controller.stop(settle_s=0.5, source="turn_complete")
            return response.output_text or "动作执行完成，机器人已停止。"

        for call in calls:
            tool_call_count += 1
            if tool_call_count > max_tool_calls:
                result: dict[str, Any] = {
                    "ok": False,
                    "error": "Tool-call limit reached; robot was stopped automatically.",
                    "stop_status": tools.controller.stop(
                        settle_s=0.5, source="tool_limit"
                    ),
                }
            else:
                try:
                    arguments = json.loads(call.arguments)
                    result = tools.call(call.name, arguments)
                except Exception as exc:
                    result = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    try:
                        result["stop_status"] = tools.controller.stop(
                            settle_s=0.5, source="tool_error"
                        )
                    except Exception as stop_exc:
                        result["stop_error"] = f"{type(stop_exc).__name__}: {stop_exc}"

            conversation.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result, ensure_ascii=False),
                }
            )

    tools.controller.stop(settle_s=0.5, source="round_limit")
    return "达到规划轮数上限，机器人已自动停止。"


def run_demo(tools: RobotTools) -> None:
    """Verify the full simulator/tool path without an API key."""
    sequence = [
        ("stand", {"duration_s": 1.0}),
        ("walk", {"speed_mps": 0.10, "duration_s": 2.0}),
        ("turn", {"yaw_rate_rad_s": 0.20, "duration_s": 2.0}),
        ("stop", {}),
    ]
    for name, arguments in sequence:
        result = tools.call(name, arguments)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result.get("fallen"):
            break


def _bounded_number(value: Any, *, low: float, high: float, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"{field} must be finite")
    return max(low, min(high, number))


def validate_local_plan(raw_plan: dict[str, Any], max_actions: int = 8) -> list[tuple[str, dict[str, Any]]]:
    """Convert untrusted model JSON into the small allow-listed robot API."""
    raw_actions = raw_plan.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("Plan must contain a non-empty actions array")

    validated: list[tuple[str, dict[str, Any]]] = []
    for item in raw_actions[:max_actions]:
        if not isinstance(item, dict):
            raise ValueError("Every action must be an object")
        name = item.get("action")
        if name not in {"stand", "walk", "turn", "stop"}:
            raise ValueError(f"Unknown or disallowed action: {name!r}")

        if name == "stop":
            validated.append(("stop", {}))
            break

        duration = _bounded_number(
            item.get("duration_s", 1.0), low=0.2, high=5.0, field="duration_s"
        )
        if name == "stand":
            arguments = {"duration_s": duration}
        elif name == "walk":
            speed = item.get("speed_mps", item.get("value"))
            arguments = {
                "speed_mps": _bounded_number(
                    speed, low=-0.15, high=0.15, field="speed_mps"
                ),
                "duration_s": duration,
            }
        else:
            yaw_rate = item.get("yaw_rate_rad_s", item.get("value"))
            arguments = {
                "yaw_rate_rad_s": _bounded_number(
                    yaw_rate, low=-0.3, high=0.3, field="yaw_rate_rad_s"
                ),
                "duration_s": duration,
            }
        validated.append((name, arguments))

    if not validated:
        raise ValueError("Plan contains no executable action")
    if validated[-1][0] != "stop":
        if len(validated) >= max_actions:
            validated[-1] = ("stop", {})
        else:
            validated.append(("stop", {}))
    return validated


def execute_local_plan(
    tools: RobotTools,
    actions: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Execute validated actions, enforcing an immediate stop on anomalies."""
    executions: list[dict[str, Any]] = []
    for name, arguments in actions:
        try:
            result = tools.call(name, arguments)
        except Exception as exc:
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        executions.append({"action": name, "arguments": arguments, "result": result})
        if not result.get("ok", True) or result.get("fallen"):
            stop_status = tools.controller.stop(settle_s=0.5, source="local_plan_error")
            executions.append(
                {"action": "stop", "arguments": {}, "result": stop_status}
            )
            break
    return {
        "ok": all(entry["result"].get("ok", True) for entry in executions),
        "executions": executions,
        "final_status": tools.controller.status(),
    }


def run_minicpmo_turn(
    tools: RobotTools,
    user_text: str,
    *,
    repo_root: Path,
    root: Path,
    planner_python: Path,
    model_path: Path,
    gpu: str,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    """Generate a plan in the MiniCPM-o venv, then execute it in the sim venv."""
    temp_root = root / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="minicpmo_plan_", dir=temp_root) as folder:
        plan_path = Path(folder) / "plan.json"
        environment = os.environ.copy()
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": gpu,
                "HF_HOME": str(root / "cache/huggingface"),
                "TORCH_HOME": str(root / "cache/torch"),
                "TMPDIR": str(temp_root),
                "TMP": str(temp_root),
                "TEMP": str(temp_root),
                "PYTHONPATH": str(repo_root),
            }
        )
        command = [
            str(planner_python),
            "-m",
            "open_duck_agent.minicpmo_planner",
            "--model-path",
            str(model_path),
            "--prompt",
            user_text,
            "--output",
            str(plan_path),
        ]
        print(f"MiniCPM-o is planning on physical GPU {gpu} ...", flush=True)
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        if completed.stdout:
            print(completed.stdout, end="", flush=True)
        if completed.returncode != 0:
            tools.controller.stop(settle_s=0.5, source="planner_failure")
            raise RuntimeError(
                "MiniCPM-o planner failed:\n" + (completed.stderr or completed.stdout)
            )
        payload = json.loads(plan_path.read_text(encoding="utf-8"))

    raw_plan = payload.get("plan")
    if not isinstance(raw_plan, dict):
        raise ValueError("Planner output is missing a plan object")
    actions = validate_local_plan(raw_plan)
    report = execute_local_plan(tools, actions)
    report["validated_plan"] = [
        {"action": name, **arguments} for name, arguments in actions
    ]
    report["raw_model_output"] = payload.get("raw_model_output", "")
    return report


def build_parser() -> argparse.ArgumentParser:
    root = Path(os.environ.get("OPEN_DUCK_ROOT", Path.home() / "open_duck"))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=root / "projects/Open_Duck_Playground",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=root / "projects/Open_Duck_Mini/BEST_WALK_ONNX_2.onnx",
    )
    parser.add_argument("--output-root", type=Path, default=root / "outputs")
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument(
        "--backend",
        choices=("minicpmo", "openai"),
        default=os.environ.get("OPEN_DUCK_AGENT_BACKEND", "minicpmo"),
    )
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument(
        "--minicpmo-python",
        type=Path,
        default=root / "minicpmo/.venv/bin/python",
    )
    parser.add_argument(
        "--minicpmo-model",
        type=Path,
        default=root / "models/MiniCPM-o-4_5-awq",
    )
    parser.add_argument(
        "--minicpmo-gpu",
        default=os.environ.get("MINICPMO_GPU", "7"),
        help="Physical CUDA GPU index visible to the isolated planner process.",
    )
    parser.add_argument("--prompt", help="Run one instruction and exit.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run stand/walk/turn/stop without calling a language model.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Keep the API/planning layer importable for unit tests on machines that do
    # not have MuJoCo.  The actual simulator is required only when starting it.
    from open_duck_agent.duck_sim import DuckSimulation

    controller = DuckSimulation(
        repo_root=args.repo_root,
        onnx_model=args.onnx_model,
        output_root=args.output_root,
        warmup_s=args.warmup_s,
    )
    tools = RobotTools(controller)
    print(f"Simulation log: {controller.log_path}", flush=True)

    if args.demo:
        run_demo(tools)
        return

    if args.backend == "minicpmo":
        def local_turn(text: str) -> str:
            try:
                report = run_minicpmo_turn(
                    tools,
                    text,
                    repo_root=args.repo_root,
                    root=Path(os.environ.get("OPEN_DUCK_ROOT", Path.home() / "open_duck")),
                    planner_python=args.minicpmo_python,
                    model_path=args.minicpmo_model,
                    gpu=args.minicpmo_gpu,
                )
            except Exception:
                tools.controller.stop(settle_s=0.5, source="local_backend_exception")
                raise
            return json.dumps(report, ensure_ascii=False, indent=2)

        if args.prompt:
            print(local_turn(args.prompt), flush=True)
            return
        print("输入自然语言动作；输入 quit 退出。", flush=True)
        while True:
            try:
                text = input("OpenDuck> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if text.lower() in {"quit", "exit", "退出"}:
                break
            if text:
                print(local_turn(text), flush=True)
        controller.stop(settle_s=0.5, source="program_exit")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is not set. Use --demo first, or export the key securely."
        )
    # Keep the OpenAI client separate from the validated simulation venv.  The
    # README installs it under /data, so it cannot force-upgrade JAX/MuJoCo
    # dependencies in the existing environment.
    vendor_dir = Path(
        os.environ.get(
            "OPENAI_VENDOR_DIR",
            str(Path(os.environ.get("OPEN_DUCK_ROOT", Path.home() / "open_duck"))
                / "python/openai-client"),
        )
    )
    if vendor_dir.is_dir():
        sys.path.insert(0, str(vendor_dir))
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            f"OpenAI client is not installed. Follow README.md; expected vendor dir: {vendor_dir}"
        ) from exc

    client = OpenAI()
    if args.prompt:
        print(run_agent_turn(client, args.model, tools, args.prompt), flush=True)
        return

    print("输入自然语言动作；输入 quit 退出。", flush=True)
    while True:
        try:
            text = input("OpenDuck> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text.lower() in {"quit", "exit", "退出"}:
            break
        if text:
            print(run_agent_turn(client, args.model, tools, text), flush=True)
    controller.stop(settle_s=0.5, source="program_exit")


if __name__ == "__main__":
    main()
