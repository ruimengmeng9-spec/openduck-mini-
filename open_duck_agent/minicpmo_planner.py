"""Run MiniCPM-o in its isolated Python environment and emit a safe plan file."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PLANNER_PROMPT = """
你是 Open Duck Mini 仿真机器人的高层动作规划器。
只输出一个 JSON 对象，不要 Markdown，不要解释，不要输出思考过程。

JSON 格式：
{{"actions":[
  {{"action":"stand","duration_s":1.0}},
  {{"action":"walk","speed_mps":0.1,"duration_s":2.0}},
  {{"action":"turn","yaw_rate_rad_s":0.2,"duration_s":1.0}},
  {{"action":"stop"}}
]}}

规则：
- 只允许 stand、walk、turn、stop。
- walk 的 speed_mps 范围是 -0.15 到 0.15，正数前进，负数后退。
- turn 的 yaw_rate_rad_s 范围是 -0.3 到 0.3，正数向左，负数向右。
- stand、walk、turn 的 duration_s 范围是 0.2 到 5.0 秒。
- 总动作数不超过 8，最后一个动作必须是 stop。
- 不得输出关节、力矩、代码、命令行或其他字段。

用户指令：{user_text}
""".strip()


def extract_json(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object from a model response."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.I)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"MiniCPM-o did not return a JSON object: {text[:500]!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {args.model_path}")

    import torch
    from transformers import AutoModel

    print(f"Loading MiniCPM-o from {args.model_path} ...", flush=True)
    model = AutoModel.from_pretrained(
        str(args.model_path),
        trust_remote_code=True,
        attn_implementation="sdpa",
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        init_vision=False,
        init_audio=False,
        init_tts=False,
    )
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [PLANNER_PROMPT.format(user_text=args.prompt)],
        }
    ]
    response = model.chat(
        msgs=messages,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        enable_thinking=False,
        use_tts_template=False,
        generate_audio=False,
    )
    if isinstance(response, (list, tuple)):
        response_text = str(response[0])
    else:
        response_text = str(response)

    plan = extract_json(response_text)
    payload = {"plan": plan, "raw_model_output": response_text}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"MiniCPM-o plan saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
