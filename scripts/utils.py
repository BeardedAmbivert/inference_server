"""Shared helpers for benchmark scripts."""

from __future__ import annotations

import json
import math
import platform
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def build_texts(texts_per_request: int) -> list[str]:
    return [
        f"benchmark sentence {index}"
        for index in range(texts_per_request)
    ]


def load_text_pool(path: Path) -> list[str]:
    """Load a JSONL text pool (one {"text": ...} object per line) into a list of strings."""
    texts: list[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return texts


def sample_texts(pool: list[str], n: int, seed: int) -> list[str]:
    """Deterministically draw n texts from pool (with replacement only if n > len(pool))."""
    rng = random.Random(seed)
    if n <= len(pool):
        return rng.sample(pool, n)
    return [rng.choice(pool) for _ in range(n)]


def length_stats(texts: list[str]) -> dict[str, Any]:
    """Char-length summary for a set of texts (stored in run metadata for context)."""
    lengths = sorted(len(text) for text in texts)
    count = len(lengths)
    if count == 0:
        return {"count": 0, "min_chars": 0, "median_chars": 0, "max_chars": 0, "mean_chars": 0}
    return {
        "count": count,
        "min_chars": lengths[0],
        "median_chars": lengths[count // 2],
        "max_chars": lengths[-1],
        "mean_chars": round(sum(lengths) / count, 1),
    }


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
