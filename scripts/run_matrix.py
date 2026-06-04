"""Driver for the benchmark run matrix defined in benchmarks/ANALYSIS.md.

Each server run needs a fresh uvicorn process because device, backend, and batch size
are read once at startup (app/config.py), so they cannot be changed in a running server.
For every config this script starts a server with the right env vars, waits for /health,
runs scripts/bench.py against it, then tears the server down. The two naive baseline runs
skip the server entirely and call scripts/naive_bench.py directly.

Usage:
    uv run python scripts/run_matrix.py             # run all 16
    uv run python scripts/run_matrix.py --dry-run   # print the plan, run nothing
    uv run python scripts/run_matrix.py --filter mps  # only runs whose name contains "mps"
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
ONNX_MODEL_DIR = REPO_ROOT / "models" / "minilm-onnx"

HOST = "127.0.0.1"
PORT = 8000
HEALTH_URL = f"http://{HOST}:{PORT}/health"
EMBED_URL = f"http://{HOST}:{PORT}/embed"

REQUESTS = 500
WARMUP = 50
TEXTS_PER_REQUEST = 1
HEALTH_TIMEOUT_S = 120.0


@dataclass(frozen=True)
class Run:
    name: str          # output file stem, e.g. "pytorch-cpu-batch32-c32"
    script: str        # "bench" (server) or "naive" (no server)
    backend: str       # "pytorch" | "onnx" for server runs; "" for naive
    device: str        # "cpu" | "mps" (metadata label; ONNX always runs on CPU)
    batch: int | None  # server MAX_BATCH_SIZE; None for naive
    concurrency: int   # bench concurrency; 1 for naive


def build_runs() -> list[Run]:
    runs = [
        Run("naive-cpu", "naive", "", "cpu", None, 1),
        Run("naive-mps", "naive", "", "mps", None, 1),
        Run("pytorch-cpu-batch1-c1", "bench", "pytorch", "cpu", 1, 1),
        Run("pytorch-mps-batch1-c1", "bench", "pytorch", "mps", 1, 1),
    ]
    for batch in (1, 8, 16, 32):
        runs.append(Run(f"pytorch-cpu-batch{batch}-c32", "bench", "pytorch", "cpu", batch, 32))
    for batch in (1, 8, 16, 32):
        runs.append(Run(f"pytorch-mps-batch{batch}-c32", "bench", "pytorch", "mps", batch, 32))
    for batch in (1, 8, 16, 32):
        runs.append(Run(f"onnx-cpu-batch{batch}-c32", "bench", "onnx", "cpu", batch, 32))
    return runs


def output_path(run: Run) -> Path:
    return BENCHMARKS_DIR / f"{run.name}.json"


def start_server(run: Run) -> subprocess.Popen:
    env = os.environ.copy()
    env["MAX_BATCH_SIZE"] = str(run.batch)
    if run.backend == "onnx":
        env["BACKEND"] = "onnx"  # ONNX path ignores DEVICE and runs on CPU
    else:
        env.pop("BACKEND", None)
        env["DEVICE"] = run.device
    return subprocess.Popen(
        ["uv", "run", "uvicorn", "app.main:app", "--host", HOST, "--port", str(PORT), "--no-access-log"],
        cwd=REPO_ROOT,
        env=env,
    )


def wait_for_health(proc: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup (code {proc.returncode})")
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=5) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"server did not become healthy within {timeout_s:.0f}s")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_bench(run: Run) -> None:
    subprocess.run(
        [
            "uv", "run", "python", "scripts/bench.py",
            "--label", run.name,
            "--backend", run.backend,
            "--device", run.device,
            "--server-batch-size", str(run.batch),
            "--concurrency", str(run.concurrency),
            "--requests", str(REQUESTS),
            "--warmup", str(WARMUP),
            "--texts-per-request", str(TEXTS_PER_REQUEST),
            "--url", EMBED_URL,
            "--output", str(output_path(run)),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def run_naive(run: Run) -> None:
    subprocess.run(
        [
            "uv", "run", "python", "scripts/naive_bench.py",
            "--label", run.name,
            "--device", run.device,
            "--requests", str(REQUESTS),
            "--warmup", str(WARMUP),
            "--texts-per-request", str(TEXTS_PER_REQUEST),
            "--output", str(output_path(run)),
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def execute(run: Run) -> None:
    if run.script == "naive":
        run_naive(run)
        return
    proc = start_server(run)
    try:
        wait_for_health(proc, HEALTH_TIMEOUT_S)
        run_bench(run)
    finally:
        stop_server(proc)
        time.sleep(1.0)  # let the port free before the next server binds


def print_plan(runs: list[Run]) -> None:
    print(f"Planned runs ({len(runs)}):")
    for i, run in enumerate(runs, 1):
        batch = "-" if run.batch is None else str(run.batch)
        print(
            f"  {i:2d}. {run.name:28s} script={run.script:5s} "
            f"backend={run.backend or '-':7s} device={run.device:3s} "
            f"batch={batch:>2s} c={run.concurrency:<2d} -> benchmarks/{run.name}.json"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark matrix.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs and exit.")
    parser.add_argument("--filter", help="Only run configs whose name contains this substring.")
    args = parser.parse_args()

    runs = build_runs()
    if args.filter:
        runs = [r for r in runs if args.filter in r.name]
        if not runs:
            raise SystemExit(f"no runs match filter {args.filter!r}")

    print_plan(runs)
    if args.dry_run:
        return

    if any(r.backend == "onnx" for r in runs) and not ONNX_MODEL_DIR.exists():
        raise SystemExit(
            f"ONNX runs selected but {ONNX_MODEL_DIR} is missing. "
            f"Run `uv run python scripts/export_onnx.py` first, or use --filter to skip onnx."
        )

    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    for i, run in enumerate(runs, 1):
        print(f"\n=== [{i}/{len(runs)}] {run.name} ===", flush=True)
        execute(run)

    print("\nMatrix complete.")


if __name__ == "__main__":
    main()
