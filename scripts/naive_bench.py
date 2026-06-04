"""Direct SentenceTransformers baseline benchmark.

This benchmark intentionally avoids HTTP, FastAPI, and DynamicBatcher overhead.
It measures one direct model.encode([text]) call per input text.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from utils import base_metadata, build_texts, format_ms, percentile, seconds_to_ms, write_json


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEFAULT_REQUESTS = 40
DEFAULT_TEXTS_PER_REQUEST = 2
DEFAULT_WARMUP = 0


def encode_one(model: SentenceTransformer, text: str) -> float:
    start = time.perf_counter()
    model.encode([text])
    return time.perf_counter() - start


def run_naive(
    model: SentenceTransformer,
    requests: int,
    texts_per_request: int,
) -> tuple[list[float], float]:
    texts = build_texts(texts_per_request)
    latencies: list[float] = []

    start = time.perf_counter()
    for _ in range(requests):
        for text in texts:
            latencies.append(encode_one(model, text))
    wall_time = time.perf_counter() - start

    return latencies, wall_time


def summarize(
    latencies: list[float],
    wall_time_seconds: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    total_sequences = len(latencies)
    return {
        "metadata": metadata,
        "summary": {
            "total_requests": metadata["requests"],
            "total_sequences": total_sequences,
            "wall_time_seconds": wall_time_seconds,
            "throughput_sequences_per_sec": total_sequences / wall_time_seconds if wall_time_seconds > 0 else 0.0,
            "avg_latency_ms": seconds_to_ms(sum(latencies) / total_sequences) if total_sequences else None,
            "p50_latency_ms": seconds_to_ms(percentile(latencies, 50)),
            "p95_latency_ms": seconds_to_ms(percentile(latencies, 95)),
            "p99_latency_ms": seconds_to_ms(percentile(latencies, 99)),
        },
    }


def print_summary(report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    summary = report["summary"]

    print("\nNaive Benchmark")
    print(f"Label: {metadata['label']}")
    print(f"Model: {metadata['model_name']}")
    print(f"Device: {metadata['device']}")
    print(f"Requests: {metadata['requests']}")
    print(f"Texts/request: {metadata['texts_per_request']}")
    print(f"Warmup requests: {metadata['warmup']}")

    print("\nResults")
    print(f"Total sequences: {summary['total_sequences']}")
    print(f"Wall time: {summary['wall_time_seconds']:.3f}s")
    print(f"Sequence throughput: {summary['throughput_sequences_per_sec']:.2f} seq/s")
    print(f"Avg latency: {format_ms(summary['avg_latency_ms'])}")
    print(f"p50 latency: {format_ms(summary['p50_latency_ms'])}")
    print(f"p95 latency: {format_ms(summary['p95_latency_ms'])}")
    print(f"p99 latency: {format_ms(summary['p99_latency_ms'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark naive direct SentenceTransformers encoding.")
    parser.add_argument("--label", default="naive-direct", help="Human-readable run label stored in output metadata.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help=f"SentenceTransformer model. Default: {DEFAULT_MODEL_NAME}")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help=f"Torch device passed to SentenceTransformer. Default: {DEFAULT_DEVICE}")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="Request-equivalent loop count.")
    parser.add_argument("--texts-per-request", type=int, default=DEFAULT_TEXTS_PER_REQUEST, help="Texts per request-equivalent loop.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup request-equivalent loops excluded from results.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.requests < 1:
        raise SystemExit("--requests must be >= 1")
    if args.texts_per_request < 1:
        raise SystemExit("--texts-per-request must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")


def main() -> None:
    args = parse_args()
    validate_args(args)

    model = SentenceTransformer(args.model_name, device=args.device)

    if args.warmup:
        run_naive(model, args.warmup, args.texts_per_request)

    latencies, wall_time = run_naive(model, args.requests, args.texts_per_request)
    metadata = {
        **base_metadata(),
        "label": args.label,
        "model_name": args.model_name,
        "device": args.device,
        "requests": args.requests,
        "texts_per_request": args.texts_per_request,
        "warmup": args.warmup,
    }
    report = summarize(latencies, wall_time, metadata)
    print_summary(report)

    if args.output:
        write_json(report, args.output)
        print(f"\nWrote JSON results to {args.output}")


if __name__ == "__main__":
    main()
