"""Summarize native MuJoCo stand/walk/turn/stop validation JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    summaries = []
    for path in args.results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["rows"]

        def phase_values(phase: str, field: str) -> list[float]:
            return [float(row[field]) for row in rows if row["phase"] == phase]

        walk_speed = phase_values("walk", "mean_forward_speed_mps")
        positive_heading = phase_values("turn_positive", "heading_change_deg")
        negative_heading = phase_values("turn_negative", "heading_change_deg")
        stop_speeds = [
            abs(float(row["mean_forward_speed_mps"]))
            for row in rows
            if row["phase"].startswith("stop") or row["phase"] == "final_stop"
        ]
        summaries.append(
            {
                "name": path.parent.name,
                "model": payload["model"],
                "completed_phases": payload["completed_phases"],
                "expected_phases": payload["expected_phases"],
                "falls": payload["falls"],
                "walk_speed_mean_mps": round(mean(walk_speed), 6),
                "turn_positive_mean_deg": round(mean(positive_heading), 3),
                "turn_negative_mean_deg": round(mean(negative_heading), 3),
                "stop_abs_speed_max_mps": round(max(stop_speeds), 6),
            }
        )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
