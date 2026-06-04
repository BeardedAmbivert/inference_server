"""Shared helpers for benchmark scripts."""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_texts(texts_per_request: int) -> list[str]:
    return [
        f"benchmark sentence {index}"
        for index in range(texts_per_request)
    ]


def base_metadata() -> dict[str, str]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }


def percentile(values: list[float], p: int) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    index = math.ceil((p / 100) * len(sorted_values)) - 1
    return sorted_values[index]


def seconds_to_ms(value: float | None) -> float | None:
    if value is None:
        return None
    return value * 1000


def format_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f} ms"


def write_json(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
