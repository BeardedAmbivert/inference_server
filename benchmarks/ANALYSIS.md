# Benchmark Analysis & Run Plan

Why the current latency numbers look the way they do, why direct comparisons
between them mislead, and the full set of runs needed to make honest claims.

> Status: analysis + planned runs. The runs in the matrix below have **not** been
> executed yet. The only results captured so far are `naive-direct.json`,
> `pytorch-batch32-c32.json`, and `onnx-batch32-c32.json`.

---

## TL;DR

- The three existing benchmarks measure **different things**, so "which is fastest"
  is the wrong question to ask of them as-is.
- `naive_bench` has the lowest latency (6.3 ms p50) only because it runs **one call
  at a time with no queue and no network** — but it also has the **lowest throughput**
  (145 seq/s). That's not "winning"; it's the single-user floor.
- The server has higher per-request latency (115 ms p50) **and** higher throughput
  (210 req/s). Trading latency for throughput under load is the entire point of batching.
- ONNX wasn't faster mainly because the comparison was **not apples-to-apples**:
  PyTorch ran on the **MPS GPU** while ONNX ran on **CPU**, the workload is
  **overhead-bound** (so the compute engine barely matters), and the ONNX model is
  **FP32, not quantized**, on tiny inputs.

---

## What each existing benchmark actually measures

| Benchmark | Stack exercised | Concurrency | Device | p50 | Throughput |
| --- | --- | ---: | --- | ---: | ---: |
| `naive-direct.json` | model only (`model.encode`) | 1 (serial) | MPS (GPU) | 6.3 ms | 145 seq/s |
| `pytorch-batch32-c32.json` | full HTTP + FastAPI + batcher | 32 | MPS (GPU) | 115 ms | 210 req/s |
| `onnx-batch32-c32.json` | full HTTP + FastAPI + batcher | 32 | **CPU** (ORT default) | 132 ms | 195 req/s |

Key code facts behind this:
- `scripts/naive_bench.py` calls `model.encode([text])` one text at a time in a plain
  loop — no server involved. Defaults to `mps` (`DEFAULT_DEVICE`).
- `scripts/bench.py` fires HTTP requests at the running server with `--concurrency`
  in flight; it measures **end-to-end per-request wall time** (send → network → queue →
  batch → infer → response). Its `--server-batch-size` flag is **metadata only** — the
  real batch size/device/backend are set on the **server** via env vars
  (`MAX_BATCH_SIZE`, `DEVICE`, `BACKEND`) at startup (`app/config.py`).
- `app/model.py`: the PyTorch path passes `device` (→ MPS here); the **ONNX path does
  not pass a device**, so ONNX Runtime falls back to the **CPU** provider on Mac ARM.

---

## Why the numbers look the way they do

### 1. Naive looks "fast" but is not winning
`naive_bench` times a single tiny sentence with **nobody in line**. Because it's strictly
serial, throughput is just `1 / latency` ≈ 145 seq/s — the *lowest* of the three. It's the
best case for one user and the worst case for utilization.

### 2. Server: latency up, throughput up (the batching trade)
With 32 requests in flight, each request waits ~one batch cycle, so per-request latency
rises to ~115 ms. But ~32 are served per cycle, so throughput rises to ~210 req/s. Raising
per-request latency while raising aggregate throughput **is** what dynamic batching is for.

### 3. Why ONNX wasn't faster
1. **Device confound (biggest).** PyTorch ran on the **MPS GPU**; ONNX ran on **CPU**.
   ONNX-on-CPU losing to PyTorch-on-GPU is expected and says nothing about ONNX itself.
2. **Overhead-bound, not compute-bound (Amdahl).** Wall time 2.38 s ÷ ~16 waves ≈ ~150 ms
   per batch cycle, but the batched GPU inference of 32 short sentences is only tens of ms.
   The rest is HTTP, JSON serialization of 32×384 floats, `.tolist()`, asyncio scheduling,
   and the GIL serializing result-splitting. ONNX only speeds the *compute* slice, which
   isn't the bottleneck — so it barely moves the total. (Same reason batching only bought
   ~1.45×: overhead dominates, not model math.)
3. **No quantization + tiny inputs.** ONNX's CPU wins usually come from **INT8
   quantization** and from **larger models / longer texts / bigger batches**. This export
   is O3 graph optimization but still FP32, and every input is one ultra-short sentence —
   the worst case for showcasing ONNX.

---

## The missing baseline: "naive, but on the server"

`naive_bench` (no server) is a *pure model floor*. It is **not** the right thing to compare
the server against, because it differs from the server on three axes simultaneously:
concurrency (1 vs 32), serving stack (none vs full), and — for ONNX — device.

To isolate one variable at a time we need **three** distinct baselines, not one:

| Baseline | How to produce it | Isolates |
| --- | --- | --- |
| **Pure model floor** | `naive_bench` (no server) | raw `model.encode` time, no serving stack |
| **Serving overhead** | server, `MAX_BATCH_SIZE=1`, **concurrency 1** | cost of HTTP + FastAPI + batcher per request (vs the floor) |
| **No-batching-under-load** | server, `MAX_BATCH_SIZE=1`, **concurrency 32** | what batching actually buys (vs `batch=32` at same concurrency) |

So: yes — we want a "naive on the server" run. Concretely it's **`MAX_BATCH_SIZE=1`**.
- `batch=1, c=1` vs the pure floor → tells you the per-request serving overhead.
- `batch=1, c=32` vs `batch=32, c=32` → tells you the true batching win (same stack, same
  concurrency, same device — only the batcher changes).

---

## Runs to do (matrix)

### Fair-comparison rules
- **Backend comparison must hold device fixed.** Compare PyTorch-CPU vs ONNX-CPU (ONNX
  has no GPU provider configured today). Comparing PyTorch-MPS vs ONNX-CPU is invalid.
- **Concurrency ≥ max batch size**, or batches can never fill. The batch sweep below uses
  `concurrency=32`, so it's valid for batch sizes up to 32. For batch 64, raise concurrency.
- Keep everything else constant across runs: `--requests 500 --warmup 50
  --texts-per-request 1`, same machine, server on port 8001.

### How to set each knob
```bash
# Server (restart per config). Examples:
DEVICE=cpu  MAX_BATCH_SIZE=32 uv run uvicorn app.main:app --port 8001   # PyTorch on CPU
DEVICE=mps  MAX_BATCH_SIZE=32 uv run uvicorn app.main:app --port 8001   # PyTorch on MPS
BACKEND=onnx MAX_BATCH_SIZE=32 uv run uvicorn app.main:app --port 8001  # ONNX (CPU, ignores DEVICE)

# Client (one per run; --backend/--server-batch-size are labels for the output file):
uv run python scripts/bench.py --backend pytorch --server-batch-size 32 \
  --concurrency 32 --requests 500 --warmup 50 --texts-per-request 1 \
  --url http://localhost:8001/embed --output benchmarks/pytorch-cpu-batch32-c32.json
```

### Naming convention
`{backend}-{device}-batch{N}-c{C}.json` — e.g. `pytorch-cpu-batch16-c32.json`.
(The current `pytorch-batch32-c32.json` was actually MPS, and `onnx-batch32-c32.json`
was CPU; regenerate/rename them under this convention. `naive-direct.json` was MPS.)

### Core matrix (16 runs)

| # | Group | Backend | Device | Batch | Concurrency | Output file | Isolates |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| 1 | Floor | — (naive) | CPU | n/a | 1 | `naive-cpu.json` | raw model time, CPU |
| 2 | Floor | — (naive) | MPS | n/a | 1 | `naive-mps.json` | raw model time, GPU |
| 3 | Overhead | pytorch | CPU | 1 | 1 | `pytorch-cpu-batch1-c1.json` | serving overhead vs floor (CPU) |
| 4 | Overhead | pytorch | MPS | 1 | 1 | `pytorch-mps-batch1-c1.json` | serving overhead vs floor (GPU) |
| 5 | Batch sweep | pytorch | CPU | 1 | 32 | `pytorch-cpu-batch1-c32.json` | no-batching-under-load (CPU) |
| 6 | Batch sweep | pytorch | CPU | 8 | 32 | `pytorch-cpu-batch8-c32.json` | batching curve (CPU) |
| 7 | Batch sweep | pytorch | CPU | 16 | 32 | `pytorch-cpu-batch16-c32.json` | batching curve (CPU) |
| 8 | Batch sweep | pytorch | CPU | 32 | 32 | `pytorch-cpu-batch32-c32.json` | batching curve (CPU) |
| 9 | Batch sweep | pytorch | MPS | 1 | 32 | `pytorch-mps-batch1-c32.json` | no-batching-under-load (GPU) |
| 10 | Batch sweep | pytorch | MPS | 8 | 32 | `pytorch-mps-batch8-c32.json` | batching curve (GPU) |
| 11 | Batch sweep | pytorch | MPS | 16 | 32 | `pytorch-mps-batch16-c32.json` | batching curve (GPU) |
| 12 | Batch sweep | pytorch | MPS | 32 | 32 | `pytorch-mps-batch32-c32.json` | batching curve (GPU) |
| 13 | Backend | onnx | CPU | 1 | 32 | `onnx-cpu-batch1-c32.json` | ONNX vs PyTorch, no batching (CPU) |
| 14 | Backend | onnx | CPU | 8 | 32 | `onnx-cpu-batch8-c32.json` | ONNX vs PyTorch (CPU) |
| 15 | Backend | onnx | CPU | 16 | 32 | `onnx-cpu-batch16-c32.json` | ONNX vs PyTorch (CPU) |
| 16 | Backend | onnx | CPU | 32 | 32 | `onnx-cpu-batch32-c32.json` | ONNX vs PyTorch (CPU) |

### What each group answers
- **Floor (1–2):** raw model speed and the model-only CPU-vs-GPU gap, no serving stack.
- **Overhead (3–4):** subtract the floor to get the per-request cost of HTTP + FastAPI +
  the batcher at a single request.
- **Batch sweep (5–12):** the headline **latency vs throughput** curve. Compare `batch=1`
  to `batch=8/16/32` at fixed concurrency to show what batching buys, separately for CPU
  and GPU.
- **Backend (13–16 vs 5–8):** ONNX vs PyTorch **on the same device (CPU)** — the only fair
  way to judge the backend.

### Optional / extended runs
- **Concurrency saturation:** fix `pytorch-mps-batch32`, sweep `--concurrency 1, 8, 16,
  32, 64, 128`. Shows where throughput plateaus and where p99 tail latency explodes.
  (For c ≥ 64 you also see batches stay full — useful with batch 64.)
- **Bigger batch:** `batch=64, c=64` (concurrency must rise to keep batches full).
- **Realistic payloads:** longer and varied text lengths, and `--texts-per-request > 1`.
  Larger inputs make compute a bigger share of total time — the regime where ONNX and
  batching matter more. (`build_texts` in `scripts/utils.py` currently emits tiny
  identical sentences.)
- **Quantized ONNX:** export an INT8 model and rerun group 13–16; this is where ONNX-CPU
  typically pulls ahead of PyTorch-CPU.
- **ONNX on GPU:** would require configuring an ORT GPU/CoreML provider in
  `app/model.py` (code change); only then is a PyTorch-MPS vs ONNX-GPU comparison valid.

---

## Caveats to keep in mind when reporting
- Warmup matters: first calls include lazy init / graph compile. Keep `--warmup 50`.
- MPS has kernel-launch overhead; for tiny inputs CPU can be surprisingly competitive.
- `run_in_executor(None, ...)` uses a thread pool; the GIL serializes Python-side work,
  though torch/ORT release it during native compute — another reason results are
  overhead-sensitive.
- All single-machine, localhost runs: no real network latency is included.
