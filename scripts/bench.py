"""Benchmark script: compare sequential vs concurrent request throughput.

Usage:
    Start the server first: uv run uvicorn app.main:app
    Then run: uv run python scripts/bench.py

This script sends N requests both sequentially and concurrently.
Sequential = no batching benefit (one at a time).
Concurrent = requests arrive together, batcher can group them.
Compare the two to measure batching impact.
"""

import asyncio
import time
import httpx
import math

URL = "http://localhost:8000/predict"
PAYLOAD = {"texts": ["hello world", "benchmark test"]}
N_REQUESTS = 20


async def send_one(client: httpx.AsyncClient) -> float:
    """Send a single POST request, return the response time in seconds."""
    start_time = time.time()
    await client.post(URL, json=PAYLOAD)
    elapsed_time = time.time() - start_time
    return elapsed_time


async def run_sequential(n: int) -> dict:
    """Send n requests one after another."""
    async with httpx.AsyncClient() as client:
        latencies = []
        for _ in range(n):
            latencies.append(await send_one(client))

    total_time = sum(latencies)
    return {
        "total_time": total_time,
        "avg_latency": total_time / n,
        "rps": n / total_time,
        "latencies": latencies,
    }


async def run_concurrent(n: int) -> dict:
    """Send n requests all at once using asyncio.gather."""
    async with httpx.AsyncClient() as client:
        start = time.time()
        latencies = list(await asyncio.gather(*(send_one(client) for _ in range(n))))
        total_time = time.time() - start

    return {
        "total_time": total_time,
        "avg_latency": sum(latencies) / n,
        "rps": n / total_time,
        "latencies": latencies,
    }


def percentile(latencies: list[float], p: int) -> float:
    """Calculate the p-th percentile from a list of latencies."""
    return sorted(latencies)[math.ceil((p/100) * len(latencies)) - 1]


async def main():
    """Run both benchmarks and print comparison."""
    print(f"Benchmarking with {N_REQUESTS} requests...\n")

    print("Running sequential...")
    seq = await run_sequential(N_REQUESTS)

    print("Running concurrent...")
    con = await run_concurrent(N_REQUESTS)

    header = f"{'Mode':<12} {'Total (s)':>10} {'Avg (s)':>10} {'p50 (s)':>10} {'p95 (s)':>10} {'p99 (s)':>10} {'RPS':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for label, r in [("Sequential", seq), ("Concurrent", con)]:
        lats = r["latencies"]
        print(
            f"{label:<12} {r['total_time']:>10.3f} {r['avg_latency']:>10.3f} "
            f"{percentile(lats, 50):>10.3f} {percentile(lats, 95):>10.3f} "
            f"{percentile(lats, 99):>10.3f} {r['rps']:>8.1f}"
        )

    speedup = seq["total_time"] / con["total_time"]
    print(f"\nSpeedup: {speedup:.2f}x")



if __name__ == "__main__":
    asyncio.run(main())
