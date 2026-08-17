# Benchmarks

This directory documents reproducible HTTP benchmark runs for the embedding inference server. Do not publish performance claims without recording the hardware, backend, server configuration, and exact command used.

Two matrices, same machine. Why the numbers look this way: [`ANALYSIS.md`](ANALYSIS.md).

- **Machine:** macOS 26.5, Apple Silicon (arm64), Python 3.12.11
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 256-token max sequence)
- Single run per config (not averaged) - treat mid-sweep points as indicative, not precise.

## nfcorpus results (2026-06-12)

[BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) sampled by `scripts/prepare_dataset.py` (pools are gitignored under `benchmarks/data/`).

| Workload | Inputs | What varies | Concurrency | JSON prefix |
| --- | --- | --- | --- | --- |
| Latency | short queries (median 17 chars) | texts/request = 1 / 8 / 32 | 1 | `lat-*-tpr*-c1.json` |
| Throughput | long docs (median ~1.6k chars) | encode batch = 128 / 256 / 512 | = batch | `tput-*-bs*.json` |

Four backends: PyTorch CPU, PyTorch MPS, ONNX fp32 (`onnx/model_O3.onnx`), ONNX dynamic INT8 (`onnx/model_int8.onnx`).

**Latency** - p50 ms (seq/s):

| Backend | tpr = 1 | tpr = 8 | tpr = 32 |
| --- | ---: | ---: | ---: |
| pytorch-cpu | 14.4 (69) | 19.1 (426) | 28.5 (1111) |
| pytorch-mps | 17.5 (40) | 20.1 (377) | 25.4 (1201) |
| onnx-fp32 | **12.4 (80)** | 19.0 (415) | 45.2 (690) |
| onnx-int8 | 22.7 (44) | 34.7 (227) | 83.7 (364) |

**Throughput** - seq/s (higher is better). INT8 rows all hit the 30 s client timeout (0 successful):

| Backend | bs = 128 | bs = 256 | bs = 512 |
| --- | ---: | ---: | ---: |
| pytorch-cpu | 39.4 | 44.0 | 58.6 |
| pytorch-mps | 40.2 | 48.0 | **68.0** |
| onnx-fp32 | 25.4 | 26.9 | 32.1 |
| onnx-int8 | timeout | timeout | timeout |

What this study shows:

1. **Two regimes.** Short queries are overhead-bound (packing texts/request dominates). Long docs + large batches are compute-bound (batch size and device start to matter; MPS wins only here, +16% at bs 512).
2. **Dynamic INT8 lost.** 4× smaller on disk, slower on short queries, and 15.4× slower than fp32 on a 128-doc batch (39.4 s vs 2.55 s). The earlier "INT8 will make ONNX win" hypothesis is falsified for dynamic quant on this hardware.

## Earlier synthetic matrix (2026-06-04)

Identical 3-word inputs, one text per request, 500 measured / 50 warmup. Isolates the batcher. JSON: `naive-*.json`, `pytorch-*-batch*-c*.json`, `onnx-cpu-batch*-c32.json`.

Latencies in ms; throughput in requests/sec (sequences/sec for the no-server baselines).

![Dynamic batching sweep](batching-sweep.png)

### Baselines - pure model, no server (serial)

| Run | Device | p50 | p95 | p99 | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| `naive-cpu` | CPU | 5.5 | 6.1 | 6.5 | 179.7 |
| `naive-mps` | MPS | 5.9 | 6.0 | 6.6 | 170.1 |

The floor: `model.encode` one text at a time, nothing else running.

### Serving overhead - one request through the full stack (batch 1, concurrency 1)

| Run | Device | p50 | p95 | p99 | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| `pytorch-cpu-batch1-c1` | CPU | 6.9 | 7.2 | 7.4 | 143.3 |
| `pytorch-mps-batch1-c1` | MPS | 7.3 | 8.8 | 9.2 | 132.0 |

HTTP + FastAPI + the batcher add only **~1.4 ms/request** over the raw model floor.

### Batching sweep - concurrency 32 (the headline)

PyTorch / CPU:

| Batch | p50 | p95 | p99 | Throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 199.5 | 202.2 | 202.8 | 159.6 |
| 8 | 117.6 | 309.5 | 608.8 | 177.1 |
| 16 | 109.2 | 231.4 | 364.6 | 215.5 |
| 32 | 103.7 | 124.2 | 594.0 | 231.1 |

PyTorch / MPS:

| Batch | p50 | p95 | p99 | Throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 220.9 | 225.2 | 228.2 | 144.2 |
| 8 | 104.1 | 323.7 | 471.1 | 179.9 |
| 16 | 98.4 | 197.9 | 328.4 | 230.0 |
| 32 | 105.9 | 128.3 | 642.5 | 222.9 |

ONNX / CPU:

| Batch | p50 | p95 | p99 | Throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 106.1 | 108.6 | 109.1 | 297.9 |
| 8 | 121.6 | 321.3 | 593.1 | 172.6 |
| 16 | 102.7 | 175.1 | 232.2 | 227.1 |
| 32 | 108.5 | 128.6 | 588.4 | 224.1 |

### What the data shows

1. **Batching is the real win.** At a fixed concurrency of 32, moving from batch 1 → 32 roughly **halves p50 latency** (PyTorch CPU 199→104 ms, MPS 221→106 ms) and lifts throughput **~45–55%**. This is the fair comparison - same concurrency, same device, only the batcher changes - unlike comparing against the serial naive baseline, which differs on three axes at once.
2. **Serving overhead is negligible** (~1.4 ms) - the stack is not the bottleneck.
3. **MPS is not faster here.** For this tiny model and single-sentence inputs the workload is overhead-bound, so the Apple GPU matches or trails CPU at every batch size.
4. **ONNX-fp32 ties PyTorch-CPU here** (104 vs 108 ms p50 at batch 32) because the workload is overhead-bound. The nfcorpus study above is the one that actually tests INT8 and longer inputs: dynamic INT8 did **not** win.
5. **Caveats / noise.** Each config is a single run. The `batch=8` rows and `onnx-cpu-batch1` look noisy (odd p95–p99 tails, and an anomalously high 298 rps for ONNX batch 1 - likely the executor thread pool running several single-item ONNX inferences in parallel). The high p99 (~580–640 ms) on several `batch=32` runs is the classic batching tail: the last requests in a wave plus the cold first batch. Repeat runs before quoting a precise figure.

### Reproduce

```bash
uv sync --extra bench

# one-time artifacts
uv run python scripts/export_onnx.py
uv run python scripts/quantize_onnx.py          # onnx/model_int8.onnx (from base model.onnx)
uv run python scripts/prepare_dataset.py        # benchmarks/data/*.jsonl (gitignored)

# run matrices (starts/stops a server per config)
uv run python scripts/run_matrix.py                       # latency + throughput (24 runs)
uv run python scripts/run_matrix.py --group latency
uv run python scripts/run_matrix.py --group throughput --filter onnx-int8
uv run python scripts/run_matrix.py --group legacy        # original 16 synthetic runs
uv run python scripts/run_matrix.py --dry-run
```

## Start the Server

Default PyTorch/SentenceTransformers backend:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Optional ONNX backend (O3 fp32 by default), after exporting the model:

```bash
uv run python scripts/export_onnx.py
BACKEND=onnx uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Dynamic INT8 (quantizes `onnx/model.onnx`, not `onnx/model_O3.onnx`):

```bash
uv run python scripts/quantize_onnx.py
ONNX_FILE_NAME=onnx/model_int8.onnx BACKEND=onnx uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run Benchmarks

Naive direct baseline:

```bash
uv run python scripts/naive_bench.py \
  --label naive-direct \
  --requests 500 \
  --texts-per-request 1 \
  --warmup 50 \
  --output benchmarks/naive-direct.json
```

Sequential baseline:

```bash
uv run python scripts/bench.py --requests 40 --concurrency 1
```

Concurrent benchmark:

```bash
uv run python scripts/bench.py --requests 40 --concurrency 8
```

Save JSON results (synthetic texts):

```bash
uv run python scripts/bench.py \
  --label pytorch-batch32-c32 \
  --backend pytorch \
  --server-batch-size 32 \
  --requests 500 \
  --concurrency 32 \
  --texts-per-request 1 \
  --warmup 50 \
  --output benchmarks/pytorch-batch32-c32.json
```

File-sourced texts (nfcorpus queries):

```bash
uv run python scripts/bench.py \
  --label lat-pytorch-cpu-tpr1-c1 \
  --backend pytorch \
  --text-source file \
  --text-file benchmarks/data/nfcorpus-queries.jsonl \
  --requests 200 \
  --concurrency 1 \
  --texts-per-request 1 \
  --warmup 20 \
  --output benchmarks/lat-pytorch-cpu-tpr1-c1.json
```

## Compare Batching Modes

Run the same benchmark commands against each server configuration.

Batching disabled:

```bash
MAX_BATCH_SIZE=1 uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Default batching:

```bash
MAX_BATCH_SIZE=32 uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Use the same `--requests`, `--concurrency`, `--texts-per-request`, and `--warmup` values for both runs.

`--backend` and `--server-batch-size` document the server configuration in the JSON output. They do not change a running server. Restart the server with the matching environment variables before running the benchmark.

## Report Format

`scripts/bench.py` prints:

- successful and failed request counts
- wall time
- throughput in requests per second
- throughput in embedded sequences per second
- average latency
- p50, p95, and p99 latency

When `--output` is provided, the JSON file includes run metadata and the same summary metrics.
