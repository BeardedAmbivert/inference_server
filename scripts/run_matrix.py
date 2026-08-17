"""Driver for the benchmark run matrices.

Each server run needs a fresh uvicorn process because device, backend, batch size, and the
ONNX model file are read once at startup (app/config.py), so they cannot be changed in a
running server. For every config this script starts a server with the right env vars, waits
for /health, runs scripts/bench.py against it, then tears the server down. Naive baseline runs
skip the server and call scripts/naive_bench.py directly.

Groups:
  legacy      - the original 16 synthetic-input runs (1 short text/request, batch-size sweep).
  latency     - nfcorpus *queries* (short), texts/request 1/8/32 at concurrency 1.
  throughput  - nfcorpus *corpus* docs (long, mixed), encode batch 128/256/512 (concurrency = batch).
The latency/throughput groups run across 4 backends: pytorch-cpu, pytorch-mps, onnx-fp32, onnx-int8.

Usage:
    uv run python scripts/run_matrix.py                       # default: new = latency + throughput
    uv run python scripts/run_matrix.py --group all           # legacy + latency + throughput
    uv run python scripts/run_matrix.py --group throughput --filter onnx-int8
    uv run python scripts/run_matrix.py --dry-run             # print the plan, run nothing
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
DATA_DIR = BENCHMARKS_DIR / "data"
ONNX_MODEL_DIR = REPO_ROOT / "models" / "minilm-onnx"

HOST = "127.0.0.1"
PORT = 8000
HEALTH_URL = f"http://{HOST}:{PORT}/health"
EMBED_URL = f"http://{HOST}:{PORT}/embed"

HEALTH_TIMEOUT_S = 120.0

QUERIES_FILE = "benchmarks/data/nfcorpus-queries.jsonl"  # short -> latency
CORPUS_FILE = "benchmarks/data/nfcorpus-corpus.jsonl"    # long/mixed -> throughput

# (label, backend, device, onnx_file_name) for the latency/throughput sweeps.
WORKLOAD_BACKENDS = [
    ("pytorch-cpu", "pytorch", "cpu", None),
    ("pytorch-mps", "pytorch", "mps", None),
    ("onnx-fp32", "onnx", "cpu", "onnx/model_O3.onnx"),
    ("onnx-int8", "onnx", "cpu", "onnx/model_int8.onnx"),
]


@dataclass(frozen=True)
class Run:
    name: str                       # output file stem / bench label
    group: str                      # "legacy" | "latency" | "throughput"
    script: str                     # "bench" (server) or "naive" (no server)
    backend: str                    # "pytorch" | "onnx" for server runs; "" for naive
    device: str                     # "cpu" | "mps" (metadata label; ONNX always runs on CPU)
    batch: int | None               # server MAX_BATCH_SIZE; None for naive
    concurrency: int                # bench concurrency; 1 for naive
    max_wait_ms: int = 500          # server MAX_WAIT_MS (batch time-trigger)
    texts_per_request: int = 1
    text_source: str = "synthetic"  # "synthetic" | "file"
    text_file: str | None = None    # JSONL pool path (relative to repo root) when source=file
    onnx_file_name: str | None = None  # ONNX_FILE_NAME (fp32 vs int8) for onnx runs
    requests: int = 500
    warmup: int = 50


def build_legacy_runs() -> list[Run]:
    runs = [
        Run("naive-cpu", "legacy", "naive", "", "cpu", None, 1),
        Run("naive-mps", "legacy", "naive", "", "mps", None, 1),
        Run("pytorch-cpu-batch1-c1", "legacy", "bench", "pytorch", "cpu", 1, 1),
        Run("pytorch-mps-batch1-c1", "legacy", "bench", "pytorch", "mps", 1, 1),
    ]
    for batch in (1, 8, 16, 32):
        runs.append(Run(f"pytorch-cpu-batch{batch}-c32", "legacy", "bench", "pytorch", "cpu", batch, 32))
    for batch in (1, 8, 16, 32):
        runs.append(Run(f"pytorch-mps-batch{batch}-c32", "legacy", "bench", "pytorch", "mps", batch, 32))
    for batch in (1, 8, 16, 32):
        runs.append(Run(f"onnx-cpu-batch{batch}-c32", "legacy", "bench", "onnx", "cpu", batch, 32))
    return runs


def build_latency_runs() -> list[Run]:
    runs = []
    for label, backend, device, onnx_file in WORKLOAD_BACKENDS:
        for tpr in (1, 8, 32):
            runs.append(Run(
                name=f"lat-{label}-tpr{tpr}-c1",
                group="latency", script="bench", backend=backend, device=device,
                # batch=32 so a request's 8/32 texts encode in one pass; tiny max_wait_ms so
                # the batcher's time-trigger doesn't add latency waiting for c=1 requests.
                batch=32, concurrency=1, max_wait_ms=5, texts_per_request=tpr,
                text_source="file", text_file=QUERIES_FILE, onnx_file_name=onnx_file,
                requests=200, warmup=20,
            ))
    return runs


def build_throughput_runs() -> list[Run]:
    runs = []
    for label, backend, device, onnx_file in WORKLOAD_BACKENDS:
        for bs in (128, 256, 512):
            runs.append(Run(
                name=f"tput-{label}-bs{bs}",
                group="throughput", script="bench", backend=backend, device=device,
                batch=bs, concurrency=bs, texts_per_request=1,
                text_source="file", text_file=CORPUS_FILE, onnx_file_name=onnx_file,
                requests=4 * bs, warmup=bs,
            ))
    return runs


def build_runs(group: str) -> list[Run]:
    groups = {
        "legacy": build_legacy_runs,
        "latency": build_latency_runs,
        "throughput": build_throughput_runs,
    }
    if group == "all":
        selected = ["legacy", "latency", "throughput"]
    elif group == "new":
        selected = ["latency", "throughput"]
    else:
        selected = [group]
    runs: list[Run] = []
    for name in selected:
        runs.extend(groups[name]())
    return runs


def output_path(run: Run) -> Path:
    return BENCHMARKS_DIR / f"{run.name}.json"


def start_server(run: Run) -> subprocess.Popen:
    env = os.environ.copy()
    env["MAX_BATCH_SIZE"] = str(run.batch)
    env["MAX_WAIT_MS"] = str(run.max_wait_ms)
    if run.backend == "onnx":
        env["BACKEND"] = "onnx"  # ONNX path ignores DEVICE and runs on CPU
        if run.onnx_file_name:
            env["ONNX_FILE_NAME"] = run.onnx_file_name
    else:
        env.pop("BACKEND", None)
        env.pop("ONNX_FILE_NAME", None)
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
    cmd = [
        "uv", "run", "python", "scripts/bench.py",
        "--label", run.name,
        "--backend", run.backend,
        "--device", run.device,
        "--server-batch-size", str(run.batch),
        "--concurrency", str(run.concurrency),
        "--requests", str(run.requests),
        "--warmup", str(run.warmup),
        "--texts-per-request", str(run.texts_per_request),
        "--text-source", run.text_source,
        "--url", EMBED_URL,
        "--output", str(output_path(run)),
    ]
    if run.text_source == "file":
        cmd += ["--text-file", run.text_file]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_naive(run: Run) -> None:
    subprocess.run(
        [
            "uv", "run", "python", "scripts/naive_bench.py",
            "--label", run.name,
            "--device", run.device,
            "--requests", str(run.requests),
            "--warmup", str(run.warmup),
            "--texts-per-request", str(run.texts_per_request),
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
            f"  {i:2d}. {run.name:24s} grp={run.group:10s} backend={run.backend or '-':7s} "
            f"device={run.device:3s} batch={batch:>3s} c={run.concurrency:<3d} "
            f"tpr={run.texts_per_request:<3d} src={run.text_source:9s} -> benchmarks/{run.name}.json"
        )


def validate_data_files(runs: list[Run]) -> None:
    needed = {run.text_file for run in runs if run.text_source == "file"}
    missing = [f for f in needed if f and not (REPO_ROOT / f).exists()]
    if missing:
        raise SystemExit(
            f"missing dataset pools: {missing}. Run `uv run python scripts/prepare_dataset.py` first."
        )
    if any(r.backend == "onnx" for r in runs) and not ONNX_MODEL_DIR.exists():
        raise SystemExit(
            f"ONNX runs selected but {ONNX_MODEL_DIR} is missing. "
            f"Run `uv run python scripts/export_onnx.py` first, or use --filter to skip onnx."
        )
    int8_needed = any(r.onnx_file_name == "onnx/model_int8.onnx" for r in runs)
    if int8_needed and not (ONNX_MODEL_DIR / "onnx" / "model_int8.onnx").exists():
        raise SystemExit(
            "onnx-int8 runs selected but onnx/model_int8.onnx is missing. "
            "Run `uv run python scripts/quantize_onnx.py` first, or use --filter to skip int8."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark matrices.")
    parser.add_argument(
        "--group", choices=("legacy", "latency", "throughput", "new", "all"), default="new",
        help="Which run group(s). 'new' = latency + throughput (default); 'all' adds legacy.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned runs and exit.")
    parser.add_argument("--filter", help="Only run configs whose name contains this substring.")
    args = parser.parse_args()

    runs = build_runs(args.group)
    if args.filter:
        runs = [r for r in runs if args.filter in r.name]
        if not runs:
            raise SystemExit(f"no runs match filter {args.filter!r}")

    print_plan(runs)
    if args.dry_run:
        return

    validate_data_files(runs)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    for i, run in enumerate(runs, 1):
        print(f"\n=== [{i}/{len(runs)}] {run.name} ===", flush=True)
        execute(run)

    print("\nMatrix complete.")


if __name__ == "__main__":
    main()
