# Benchmarks - methodology, results & analysis

How the embedding server behaves under two realistic workloads on real data, why the
numbers look the way they do, and the earlier synthetic-input study kept for comparison.

- **Machine:** macOS 26.5, Apple Silicon (arm64), Python 3.12.11
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, 256-token max sequence)
- **Data:** [BeIR/nfcorpus](https://huggingface.co/datasets/BeIR/nfcorpus) - short medical **queries**
  (median 17 chars) and long **corpus** abstracts (median ~1.6k chars, capped at 8192). Sampled with
  `scripts/prepare_dataset.py`.
- Single run per config (not averaged) - treat mid-sweep points as indicative, not precise.

> Why real data: SentenceTransformers warns that backend performance differs strongly between short
> and long texts and recommends benchmarking your own distribution. The original study (below) used
> identical 3-word synthetic inputs, which is purely **overhead-bound** - it can't show where the
> compute backend or batch size actually matters. nfcorpus gives both regimes.

---

## Two workloads

| Workload | Inputs | What varies | Concurrency | Measures |
| --- | --- | --- | --- | --- |
| **Latency** | short queries | texts/request = 1 / 8 / 32 | 1 | per-request p50/p95 |
| **Throughput** | long docs | encode batch = 128 / 256 / 512 | = batch size | sequences/sec |

Both run through the real `POST /embed` path. The encode batch size equals the server's
`MAX_BATCH_SIZE` (the worker encodes the whole aggregated batch in one pass - see
[Methodology](#methodology--fair-comparison)), so the throughput sweep is just `MAX_BATCH_SIZE`
driven at `concurrency = batch`. Latency runs use a tiny `MAX_WAIT_MS` so the batcher's time-trigger
doesn't add idle wait at concurrency 1.

---

## Latency workload - short queries, concurrency 1

p50 latency (ms), with sequence throughput (seq/s) in brackets:

| Backend | tpr = 1 | tpr = 8 | tpr = 32 |
| --- | ---: | ---: | ---: |
| pytorch-cpu | 14.4 ms (69) | 19.1 ms (426) | 28.5 ms (1111) |
| pytorch-mps | 17.5 ms (40) | 20.1 ms (377) | 25.4 ms (1201) |
| onnx-fp32 | **12.4 ms (80)** | 19.0 ms (415) | 45.2 ms (690) |
| onnx-int8 | 22.7 ms (44) | 34.7 ms (227) | 83.7 ms (364) |

1. **Packing texts per request is the dominant latency-amortization lever.** Going from 1 to 32 texts
   per request lifts throughput ~16× (pytorch-cpu 69 → 1111 seq/s) while p50 only doubles - fixed
   per-request overhead (HTTP, JSON, scheduling) is spread over more sequences.
2. **Different winners at different sizes.** ONNX-fp32 is fastest for a single short query (12.4 ms);
   PyTorch (CPU/MPS) wins at 32 texts/request (~25–28 ms, ~1100–1200 seq/s).
3. **INT8 is the slowest backend at every size** - see the INT8 finding below.

---

## Throughput workload - long docs, concurrency = batch

Sequence throughput (seq/s); higher is better. p50 here is multi-second by design (requests queue in
deep waves at `concurrency = batch` - that is the latency *cost* of maximizing throughput):

| Backend | bs = 128 | bs = 256 | bs = 512 |
| --- | ---: | ---: | ---: |
| pytorch-cpu | 39.4 | 44.0 | 58.6 |
| pytorch-mps | 40.2 | 48.0 | **68.0** |
| onnx-fp32 | 25.4 | 26.9 | 32.1 |
| onnx-int8 | timeout | timeout | timeout |

1. **Throughput rises with batch size** once inputs are compute-bound (pytorch-cpu 39 → 59,
   mps 40 → 68 from bs128 → bs512). This is the opposite end of the curve from the latency workload.
2. **MPS finally wins - but only here.** On long docs at bs512 the Apple GPU leads CPU by ~16%
   (68 vs 59 seq/s). In the overhead-bound synthetic study (below) MPS never beat CPU. Same hardware,
   opposite conclusion depending on the input distribution - exactly the reason to test real data.
3. **ONNX-fp32 trails PyTorch on long docs** (32 vs 59–68 seq/s at bs512) in this single-session,
   single-thread serving setup.

### The INT8 finding (hypothesis falsified)

`ANALYSIS.md` previously hypothesized that ONNX would win once quantized to INT8. **Measured, it does
the opposite.** Dynamic-INT8 (`scripts/quantize_onnx.py`, 4× smaller on disk) is:

- **Slightly slower on short inputs** (latency table: 22.7 vs 12.4 ms at tpr=1) - quantize/dequantize
  overhead with no compute to amortize.
- **Catastrophically slower on the large-batch long-doc workload** - every request exceeded the 30 s
  client timeout (0 successful at bs128/256/512). Offline, a single 128-doc batch takes **39.4 s vs
  2.55 s for fp32 - 15.4× slower** (3.2 vs 50 docs/s).

This is a known ONNX Runtime dynamic-quantization failure mode: activations are quantized per
inference, and the INT8 MatMul kernels are not optimized for these batch×sequence shapes on Apple
ARM (no AVX-512 VNNI). **Takeaway:** dynamic INT8 is the wrong tool for MiniLM on this hardware;
weight size shrinks 4× but latency and throughput both regress. Static/calibrated quantization, or a
target with VNNI, would be the next experiment - not dynamic quant.

---

## Quality - does INT8 still retrieve? (2026-08-19)

The speed study falsifies "INT8 is faster." It does not say whether the quantized graph still
embeds. `scripts/eval_quality.py` runs the production encode path (`app.model.load_model` →
`model.encode`) on the BeIR/nfcorpus **test** split: 323 queries, 3,633 corpus docs, 12,334 qrels.
PyTorch CPU is the reference. Both ONNX graphs are compared on:

1. **Cosine drift** of matching rows (short queries vs long corpus docs separately, because
   SentenceTransformers warns that backend error depends on length).
2. **Rank agreement** vs PyTorch (does the nearest neighbor stay the same?).
3. **Retrieval:** nDCG / recall / MRR at 10 and 100 with cosine ranking on L2-normalized vectors.
   DCG uses the TREC/BEIR gain `2^{rel} - 1`. BeIR's nfcorpus-qrels are binary (`score=1`).

Machine: macOS 26.6.1, Apple Silicon (arm64). JSON: `benchmarks/quality-nfcorpus.json`.

**Cosine drift vs pytorch:**

| Backend | queries | corpus | overall | p05 | min | mean angle | top-1 overlap | top-10 overlap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| onnx-fp32 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.08° | 0.997 | 1.000 |
| onnx-int8 | 0.961 | 0.949 | 0.950 | 0.927 | 0.855 | 18.1° | 0.693 | 0.757 |

**Retrieval:**

| Backend | nDCG@10 | nDCG@100 | recall@10 | recall@100 | MRR@10 | Δ nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pytorch | 0.317 | 0.301 | 0.155 | 0.311 | 0.508 | - |
| onnx-fp32 | 0.317 | 0.300 | 0.155 | 0.312 | 0.506 | -0.001 |
| onnx-int8 | 0.308 | 0.293 | 0.152 | 0.299 | 0.508 | -0.009 |

1. **ONNX O3 fp32 is numerically a stand-in for PyTorch.** Mean cosine 0.999999, mean angle 0.08°,
   nDCG@10 0.3166 vs 0.3173. Export/optimization is not a hidden quality regression.
2. **Dynamic INT8 is a different embedding space that still ranks.** Mean cosine 0.950 (queries
   0.961, long docs 0.949 - the length split the ST warning predicts). Mean angular error 18°.
   Top-1 neighbor vs PyTorch matches on only 69% of queries. nDCG@10 drops 0.9 points, recall@10
   0.155 → 0.152, MRR@10 is flat (0.508). nfcorpus has ~38 relevant docs/query, so a changed
   nearest neighbor is often still relevant - cosine drift overstates the IR damage.
3. **The two INT8 results are one claim.** Smaller on disk, slower on this hardware, geometrically
   drifted, retrieval almost unchanged. A speed-only bench would have stopped at "INT8 lost." The
   quality bench says *why you might still export it* (disk) and *why you should not* (no speed
   win, and you do not get PyTorch-identical vectors).

`--qa` is the regression gate: onnx-fp32 mean cosine ≥ 0.995 and nDCG@10 drop ≤ 0.005; INT8 mean
cosine ≥ 0.94 and nDCG@10 drop ≤ 0.015. Metric functions are unit-tested without downloading the
model (`tests/test_quality_metrics.py`). The full encode is a local job, not CI - it needs the
ONNX artifacts, BeIR, and ~4 minutes on this machine.

---

## Earlier study - synthetic-input batching sweep (kept for comparison)

The original 16-run matrix used identical 3-word synthetic inputs (1 text/request), sweeping
`MAX_BATCH_SIZE` at concurrency 32. It established the core batching result and is retained as a
baseline (`benchmarks/{naive,pytorch,onnx}-*.json`, regenerate with `run_matrix.py --group legacy`).

![Dynamic batching sweep](batching-sweep.png)

PyTorch / CPU, concurrency 32 (p50 ms / throughput):

| Batch | p50 | Throughput |
| ---: | ---: | ---: |
| 1 | 199.5 ms | 159.6 req/s |
| 8 | 117.6 ms | 177.1 req/s |
| 16 | 109.2 ms | 215.5 req/s |
| 32 | 103.7 ms | 231.1 req/s |

- **Batching is the win:** batch 1 → 32 roughly halves p50 and lifts throughput ~45% - same
  concurrency, same device, only the batcher changes.
- **Serving overhead ≈ 1.4 ms/request** over the raw `model.encode` floor (naive baseline ~5.5 ms).
- **MPS and ONNX-fp32 tie PyTorch-CPU here** because the workload is overhead-bound - the contrast
  with the real-data throughput workload above is the whole point.

---

## Methodology & fair-comparison

- **Backend comparison holds device fixed.** PyTorch-CPU vs ONNX-CPU is fair; PyTorch-MPS vs ONNX-CPU
  is not (ONNX has no GPU provider configured). MPS rows are labelled separately.
- **Encode batch = `MAX_BATCH_SIZE`.** The worker flattens the aggregated batch and calls
  `model.encode(..., batch_size=max_batch_size)` (`app/batching.py` → `app/model.py:predict`),
  replacing `model.encode`'s hidden default of 32. This is what lets the throughput workload form real
  128/256/512 batches through the API rather than being silently chunked into 32s.
- **Concurrency ≥ batch size**, or batches can never fill. Throughput runs use `concurrency = batch`.
- **Per-run knobs are server env vars** (`MAX_BATCH_SIZE`, `MAX_WAIT_MS`, `DEVICE`, `BACKEND`,
  `ONNX_FILE_NAME`) read once at startup; `scripts/run_matrix.py` restarts a server per config.
- **Caveats:** single run per config; warmup excluded; localhost only (no real network); occasional
  client `ReadError` (≤1 / 1024) under 256–512 concurrent sockets is connection noise, not a server
  error; the INT8 throughput rows are all-timeout by the 30 s client deadline.

---

## Reproduce

```bash
uv sync --extra bench

# one-time: export the ONNX model + INT8 variant, and sample the dataset
uv run python scripts/export_onnx.py
uv run python scripts/quantize_onnx.py
uv run python scripts/prepare_dataset.py        # writes benchmarks/data/*.jsonl (gitignored)

# run the workloads (starts/stops a server per config)
uv run python scripts/run_matrix.py                       # latency + throughput (24 runs)
uv run python scripts/run_matrix.py --group latency       # just one group
uv run python scripts/run_matrix.py --group throughput --filter onnx-int8
uv run python scripts/run_matrix.py --group legacy        # the original synthetic matrix
uv run python scripts/run_matrix.py --dry-run             # print the plan, run nothing

# quality (no server): cosine drift + nfcorpus nDCG/recall, fp32 vs INT8
uv run python scripts/eval_quality.py
uv run python scripts/eval_quality.py --qa                # exit 1 if a quality gate fails
```

Each run writes `benchmarks/<name>.json` with full metadata (backend, device, batch, concurrency,
texts/request, input length stats) and the p50/p95/p99 + throughput summary.
