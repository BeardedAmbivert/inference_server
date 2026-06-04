"""HTTP benchmark runner for the embedding inference server."""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from utils import base_metadata, build_texts, format_ms, percentile, seconds_to_ms, write_json


DEFAULT_URL = "http://localhost:8000/embed"
DEFAULT_REQUESTS = 40
DEFAULT_CONCURRENCY = 1
DEFAULT_TEXTS_PER_REQUEST = 2
DEFAULT_WARMUP = 0
DEFAULT_TIMEOUT_SECONDS = 30.0
BACKENDS = ("pytorch", "onnx")


@dataclass(frozen=True)
class RequestResult:
    latency_seconds: float
    status_code: int | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.status_code is not None and 200 <= self.status_code < 300


def build_payload(texts_per_request: int) -> dict[str, list[str]]:
    return {"texts": build_texts(texts_per_request)}


async def send_one(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, list[str]],
) -> RequestResult:
    start = time.perf_counter()
    try:
        response = await client.post(url, json=payload)
        latency = time.perf_counter() - start
        return RequestResult(
            latency_seconds=latency,
            status_code=response.status_code,
            error=None if 200 <= response.status_code < 300 else response.text[:200],
        )
    except Exception as exc:
        latency = time.perf_counter() - start
        return RequestResult(
            latency_seconds=latency,
            status_code=None,
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_requests(
    url: str,
    total_requests: int,
    concurrency: int,
    texts_per_request: int,
    timeout_seconds: float,
) -> tuple[list[RequestResult], float]:
    payload = build_payload(texts_per_request)
    timeout = httpx.Timeout(timeout_seconds)
    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        start = time.perf_counter()

        async def bounded_send() -> RequestResult:
            async with semaphore:
                return await send_one(client, url, payload)

        results = await asyncio.gather(*(bounded_send() for _ in range(total_requests)))
        wall_time = time.perf_counter() - start

    return list(results), wall_time


def summarize(
    results: list[RequestResult],
    wall_time_seconds: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    successful = [result.latency_seconds for result in results if result.ok]
    failed = [result for result in results if not result.ok]
    success_count = len(successful)
    successful_sequences = success_count * metadata["texts_per_request"]

    summary = {
        "metadata": metadata,
        "summary": {
            "total_requests": len(results),
            "successful_requests": success_count,
            "failed_requests": len(failed),
            "successful_sequences": successful_sequences,
            "wall_time_seconds": wall_time_seconds,
            "throughput_rps": success_count / wall_time_seconds if wall_time_seconds > 0 else 0.0,
            "throughput_sequences_per_sec": successful_sequences / wall_time_seconds if wall_time_seconds > 0 else 0.0,
            "avg_latency_ms": seconds_to_ms(sum(successful) / success_count) if success_count else None,
            "p50_latency_ms": seconds_to_ms(percentile(successful, 50)),
            "p95_latency_ms": seconds_to_ms(percentile(successful, 95)),
            "p99_latency_ms": seconds_to_ms(percentile(successful, 99)),
        },
        "failures": [
            {
                "status_code": result.status_code,
                "error": result.error,
                "latency_ms": seconds_to_ms(result.latency_seconds),
            }
            for result in failed[:10]
        ],
    }
    return summary


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    metadata = report["metadata"]

    print("\nBenchmark")
    print(f"Label: {metadata['label']}")
    print(f"Backend: {metadata['backend']}")
    print(f"Server batch size: {metadata['server_batch_size']}")
    print(f"URL: {metadata['url']}")
    print(f"Requests: {metadata['requests']}")
    print(f"Concurrency: {metadata['concurrency']}")
    print(f"Texts/request: {metadata['texts_per_request']}")
    print(f"Warmup requests: {metadata['warmup']}")

    print("\nResults")
    print(f"Successful: {summary['successful_requests']}")
    print(f"Failed: {summary['failed_requests']}")
    print(f"Successful sequences: {summary['successful_sequences']}")
    print(f"Wall time: {summary['wall_time_seconds']:.3f}s")
    print(f"Request throughput: {summary['throughput_rps']:.2f} req/s")
    print(f"Sequence throughput: {summary['throughput_sequences_per_sec']:.2f} seq/s")
    print(f"Avg latency: {format_ms(summary['avg_latency_ms'])}")
    print(f"p50 latency: {format_ms(summary['p50_latency_ms'])}")
    print(f"p95 latency: {format_ms(summary['p95_latency_ms'])}")
    print(f"p99 latency: {format_ms(summary['p99_latency_ms'])}")

    if report["failures"]:
        print("\nFirst failures")
        for failure in report["failures"]:
            print(f"- status={failure['status_code']} error={failure['error']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the embedding inference HTTP API.")
    parser.add_argument("--label", default="http-benchmark", help="Human-readable run label stored in output metadata.")
    parser.add_argument("--backend", choices=BACKENDS, default="pytorch", help="Backend label for metadata only.")
    parser.add_argument("--device", help="Device the server ran on (cpu/mps). Metadata only.")
    parser.add_argument("--server-batch-size", type=int, help="Server MAX_BATCH_SIZE used for this run. Metadata only.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Endpoint URL. Default: {DEFAULT_URL}")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="Total measured requests.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Concurrent in-flight requests.")
    parser.add_argument("--texts-per-request", type=int, default=DEFAULT_TEXTS_PER_REQUEST, help="Texts in each /embed request.")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP, help="Warmup requests excluded from results.")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="Per-request timeout in seconds.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.requests < 1:
        raise SystemExit("--requests must be >= 1")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.texts_per_request < 1:
        raise SystemExit("--texts-per-request must be >= 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if args.server_batch_size is not None and args.server_batch_size < 1:
        raise SystemExit("--server-batch-size must be >= 1")


async def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.warmup:
        await run_requests(
            url=args.url,
            total_requests=args.warmup,
            concurrency=args.concurrency,
            texts_per_request=args.texts_per_request,
            timeout_seconds=args.timeout,
        )

    results, wall_time = await run_requests(
        url=args.url,
        total_requests=args.requests,
        concurrency=args.concurrency,
        texts_per_request=args.texts_per_request,
        timeout_seconds=args.timeout,
    )

    metadata = {
        **base_metadata(),
        "label": args.label,
        "backend": args.backend,
        "device": args.device,
        "server_batch_size": args.server_batch_size,
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "texts_per_request": args.texts_per_request,
        "warmup": args.warmup,
        "timeout_seconds": args.timeout,
    }
    report = summarize(results, wall_time, metadata)
    print_summary(report)

    if args.output:
        write_json(report, args.output)
        print(f"\nWrote JSON results to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
