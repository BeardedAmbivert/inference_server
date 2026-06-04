"""Plot the dynamic-batching sweep from the benchmark matrix JSONs.

Reads benchmarks/{config}-batch{N}-c32.json for the three sweep configs and writes
benchmarks/batching-sweep.png (p50 latency and throughput vs batch size).

Run:
    uv run --extra plot python scripts/plot_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BENCHMARKS_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
OUTPUT = BENCHMARKS_DIR / "batching-sweep.png"
BATCH_SIZES = [1, 8, 16, 32]
SERIES = {
    "PyTorch / CPU": "pytorch-cpu",
    "PyTorch / MPS": "pytorch-mps",
    "ONNX / CPU": "onnx-cpu",
}


def load(prefix: str) -> tuple[list[float], list[float]]:
    p50, tput = [], []
    for n in BATCH_SIZES:
        data = json.loads((BENCHMARKS_DIR / f"{prefix}-batch{n}-c32.json").read_text())
        summary = data["summary"]
        p50.append(summary["p50_latency_ms"])
        tput.append(summary["throughput_rps"])
    return p50, tput


def main() -> None:
    fig, (ax_latency, ax_tput) = plt.subplots(1, 2, figsize=(11, 4.5))

    for label, prefix in SERIES.items():
        p50, tput = load(prefix)
        ax_latency.plot(BATCH_SIZES, p50, marker="o", label=label)
        ax_tput.plot(BATCH_SIZES, tput, marker="o", label=label)

    ax_latency.set_title("p50 latency vs batch size")
    ax_latency.set_xlabel("max batch size")
    ax_latency.set_ylabel("p50 latency (ms)")

    ax_tput.set_title("Throughput vs batch size")
    ax_tput.set_xlabel("max batch size")
    ax_tput.set_ylabel("throughput (req/s)")

    for ax in (ax_latency, ax_tput):
        ax.set_xticks(BATCH_SIZES)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Dynamic batching sweep — all-MiniLM-L6-v2, concurrency 32, 500 req, 1 text/req")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=120)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
