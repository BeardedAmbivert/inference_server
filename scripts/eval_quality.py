"""Embedding quality eval: cosine drift and nfcorpus retrieval, fp32 vs INT8.

Speed benches in this repo already show dynamic INT8 is slower on this hardware.
This script answers the missing question: does the quantized graph still retrieve?

Two measurements, same encode path as production (`model.encode` via app.model.load_model):

1. Cosine drift of each backend vs PyTorch CPU (short queries and long corpus docs).
2. BeIR/nfcorpus test retrieval: nDCG / recall / MRR at 10 and 100, plus top-k rank
   agreement vs PyTorch (does INT8 preserve nearest neighbors, not just vector values?).

Usage:
    uv sync --extra bench
    uv run python scripts/export_onnx.py          # once
    uv run python scripts/quantize_onnx.py        # once, for INT8
    uv run python scripts/eval_quality.py
    uv run python scripts/eval_quality.py --qa    # also fail if drift/nDCG gates trip

Writes benchmarks/quality-nfcorpus.json by default.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from quality_metrics import (  # noqa: E402
    cosine_drift_report,
    cosine_matrix,
    mean_rank_agreement,
    mean_retrieval_metrics,
    ranked_doc_ids,
)
from utils import base_metadata, write_json  # noqa: E402

from app.config import settings  # noqa: E402
from app.model import load_model  # noqa: E402

MAX_CHARS = settings.max_chars_per_text
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "quality-nfcorpus.json"
DEFAULT_KS = (10, 100)
DEFAULT_AGREEMENT_KS = (1, 10)

# QA gates from the committed nfcorpus run (benchmarks/quality-nfcorpus.json).
# onnx-fp32 must match pytorch geometrically. INT8 is allowed vector-space drift
# (measured mean cosine ~0.95) as long as nDCG@10 stays within 1.5 points.
DEFAULT_MIN_COSINE = {
    "onnx-fp32": 0.995,
    "onnx-int8": 0.94,
}
DEFAULT_MAX_NDCG10_DROP = {
    "onnx-fp32": 0.005,
    "onnx-int8": 0.015,
}


def _row_text(title: str | None, text: str | None) -> str:
    raw = " ".join(part for part in (title, text) if part)
    collapsed = " ".join(raw.split())
    return collapsed[:MAX_CHARS]


def load_nfcorpus(qrels_split: str) -> tuple[list[str], list[str], list[str], list[str], dict[str, dict[str, int]]]:
    """Return (query_ids, query_texts, doc_ids, doc_texts, qrels) for the BeIR test split."""
    from datasets import load_dataset

    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
    queries_ds = load_dataset("BeIR/nfcorpus", "queries")["queries"]
    qrels_ds = load_dataset("BeIR/nfcorpus-qrels", split=qrels_split)

    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        query_id = str(row["query-id"])
        doc_id = str(row["corpus-id"])
        score = int(row["score"])
        if score <= 0:
            continue
        qrels.setdefault(query_id, {})[doc_id] = score

    query_map = {
        str(row["_id"]): _row_text(row.get("title"), row.get("text")) for row in queries_ds
    }
    doc_map = {
        str(row["_id"]): _row_text(row.get("title"), row.get("text")) for row in corpus_ds
    }

    query_ids = [qid for qid in qrels if qid in query_map and query_map[qid]]
    query_texts = [query_map[qid] for qid in query_ids]
    doc_ids = [did for did, text in doc_map.items() if text]
    doc_texts = [doc_map[did] for did in doc_ids]
    return query_ids, query_texts, doc_ids, doc_texts, qrels


def onnx_file_path(onnx_dir: Path, file_name: str) -> Path:
    return onnx_dir / file_name


def load_backend(name: str, onnx_dir: Path) -> Any | None:
    """Load a backend or return None if its ONNX artifact is missing."""
    if name == "pytorch":
        print(f"loading {name} ({settings.model_name}, cpu)")
        return load_model(settings.model_name, device="cpu", backend=None)
    if name == "onnx-fp32":
        file_name = "onnx/model_O3.onnx"
    elif name == "onnx-int8":
        file_name = "onnx/model_int8.onnx"
    else:
        raise ValueError(f"unknown backend {name}")
    path = onnx_file_path(onnx_dir, file_name)
    if not path.exists():
        print(f"skip {name}: {path} not found (export/quantize first)")
        return None
    print(f"loading {name} ({path})")
    return load_model(str(onnx_dir), device="cpu", backend="onnx", onnx_file_name=file_name)


def encode_texts(model: Any, texts: list[str], batch_size: int, label: str) -> np.ndarray:
    print(f"  encode {label}: {len(texts)} texts, batch_size={batch_size}")
    start = time.perf_counter()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    elapsed = time.perf_counter() - start
    arr = np.asarray(embeddings, dtype=np.float32)
    print(f"  done {label} in {elapsed:.1f}s ({len(texts) / elapsed:.1f} seq/s)")
    return arr


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    print()
    print(title)
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "-|-".join("-" * widths[i] for i in range(len(headers)))
    print(line)
    print(rule)
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def evaluate_backend(
    query_emb: np.ndarray,
    doc_emb: np.ndarray,
    query_ids: list[str],
    doc_ids: list[str],
    qrels: dict[str, dict[str, int]],
    ks: tuple[int, ...],
) -> dict[str, Any]:
    scores = cosine_matrix(query_emb, doc_emb)
    rankings_list = ranked_doc_ids(scores, doc_ids)
    rankings = {qid: ranking for qid, ranking in zip(query_ids, rankings_list)}
    retrieval = mean_retrieval_metrics(rankings, qrels, ks=ks)
    return {"retrieval": retrieval, "rankings": rankings}


def qa_failures(
    report: dict[str, Any],
    min_cosine: dict[str, float],
    max_ndcg10_drop: dict[str, float],
) -> list[str]:
    """Return human-readable gate failures. Empty list means pass."""
    failures: list[str] = []
    backends = report["backends"]
    if "pytorch" not in backends:
        failures.append("QA gate requires pytorch as the reference backend")
        return failures
    ref_ndcg = backends["pytorch"]["retrieval"]["ndcg@10"]
    for name, backend in backends.items():
        if name == "pytorch":
            continue
        overall = backend["drift"]["overall"]
        mean_cos = overall["cosine_mean"]
        floor = min_cosine.get(name)
        if floor is not None and mean_cos < floor:
            failures.append(
                f"{name} mean cosine vs pytorch is {mean_cos:.4f} (floor {floor:.4f})"
            )
        ndcg = backend["retrieval"]["ndcg@10"]
        drop = ref_ndcg - ndcg
        cap = max_ndcg10_drop.get(name)
        if cap is not None and drop > cap:
            failures.append(
                f"{name} nDCG@10 drop vs pytorch is {drop:.4f} "
                f"({ref_ndcg:.4f} -> {ndcg:.4f}, cap {cap:.4f})"
            )
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fp32 vs INT8 embedding quality on nfcorpus")
    parser.add_argument(
        "--backends",
        default="pytorch,onnx-fp32,onnx-int8",
        help="Comma-separated backends to load.",
    )
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=REPO_ROOT / settings.onnx_model_path,
        help="Local SentenceTransformers ONNX export directory.",
    )
    parser.add_argument("--qrels-split", default="test", help="BeIR qrels split (default: test).")
    parser.add_argument("--batch-size", type=int, default=32, help="Encode batch for fp32 backends.")
    parser.add_argument(
        "--int8-batch-size",
        type=int,
        default=8,
        help="Encode batch for onnx-int8 (smaller: avoids the large-batch INT8 slowdown).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--qa",
        action="store_true",
        help="Exit 1 if cosine/nDCG gates fail (use in a quality-check job).",
    )
    parser.add_argument(
        "--min-cosine-fp32",
        type=float,
        default=DEFAULT_MIN_COSINE["onnx-fp32"],
        help="QA: minimum mean cosine of onnx-fp32 vs pytorch.",
    )
    parser.add_argument(
        "--min-cosine-int8",
        type=float,
        default=DEFAULT_MIN_COSINE["onnx-int8"],
        help="QA: minimum mean cosine of onnx-int8 vs pytorch.",
    )
    parser.add_argument(
        "--max-ndcg10-drop-fp32",
        type=float,
        default=DEFAULT_MAX_NDCG10_DROP["onnx-fp32"],
    )
    parser.add_argument(
        "--max-ndcg10-drop-int8",
        type=float,
        default=DEFAULT_MAX_NDCG10_DROP["onnx-int8"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wanted = [name.strip() for name in args.backends.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in {"pytorch", "onnx-fp32", "onnx-int8"}]
    if unknown:
        raise SystemExit(f"unknown backends: {unknown}")

    print("loading BeIR/nfcorpus ...")
    query_ids, query_texts, doc_ids, doc_texts, qrels = load_nfcorpus(args.qrels_split)
    print(
        f"nfcorpus {args.qrels_split}: {len(query_ids)} queries, {len(doc_ids)} docs, "
        f"{sum(len(v) for v in qrels.values())} qrels"
    )

    onnx_dir = args.onnx_dir.resolve()
    models: dict[str, Any] = {}
    for name in wanted:
        model = load_backend(name, onnx_dir)
        if model is not None:
            models[name] = model
    if "pytorch" not in models:
        raise SystemExit("pytorch backend is required as the quality reference")
    if len(models) < 2:
        raise SystemExit("need at least one ONNX backend to compare against pytorch")

    embeddings: dict[str, dict[str, np.ndarray]] = {}
    encode_meta: dict[str, dict[str, Any]] = {}
    for name, model in models.items():
        batch_size = args.int8_batch_size if name == "onnx-int8" else args.batch_size
        print(f"\n== {name} ==")
        t0 = time.perf_counter()
        query_emb = encode_texts(model, query_texts, batch_size, "queries")
        doc_emb = encode_texts(model, doc_texts, batch_size, "corpus")
        encode_meta[name] = {
            "batch_size": batch_size,
            "encode_s": round(time.perf_counter() - t0, 2),
        }
        embeddings[name] = {"queries": query_emb, "corpus": doc_emb}

    ks = DEFAULT_KS
    per_backend: dict[str, Any] = {}
    ranking_store: dict[str, dict[str, list[str]]] = {}
    ref_q = embeddings["pytorch"]["queries"]
    ref_d = embeddings["pytorch"]["corpus"]

    for name, emb in embeddings.items():
        result = evaluate_backend(
            emb["queries"], emb["corpus"], query_ids, doc_ids, qrels, ks
        )
        ranking_store[name] = result["rankings"]
        drift: dict[str, Any] = {}
        if name != "pytorch":
            drift["queries"] = cosine_drift_report(ref_q, emb["queries"])
            drift["corpus"] = cosine_drift_report(ref_d, emb["corpus"])
            stacked_ref = np.vstack([ref_q, ref_d])
            stacked_cand = np.vstack([emb["queries"], emb["corpus"]])
            drift["overall"] = cosine_drift_report(stacked_ref, stacked_cand)
            drift["rank_agreement"] = mean_rank_agreement(
                ranking_store["pytorch"], ranking_store[name], ks=DEFAULT_AGREEMENT_KS
            )
        per_backend[name] = {
            "encode": encode_meta[name],
            "retrieval": result["retrieval"],
            "drift": drift,
        }

    report: dict[str, Any] = {
        **base_metadata(),
        "model": settings.model_name,
        "device": "cpu",
        "dataset": "BeIR/nfcorpus",
        "qrels_split": args.qrels_split,
        "n_queries": len(query_ids),
        "n_docs": len(doc_ids),
        "n_qrels": sum(len(v) for v in qrels.values()),
        "ks": list(ks),
        "reference": "pytorch",
        "backends": per_backend,
    }

    # Cosine drift table
    drift_rows: list[list[str]] = []
    for name, backend in per_backend.items():
        if name == "pytorch":
            continue
        overall = backend["drift"]["overall"]
        queries = backend["drift"]["queries"]
        corpus = backend["drift"]["corpus"]
        agree = backend["drift"]["rank_agreement"]
        drift_rows.append(
            [
                name,
                fmt(queries["cosine_mean"]),
                fmt(corpus["cosine_mean"]),
                fmt(overall["cosine_mean"]),
                fmt(overall["cosine_p05"]),
                fmt(overall["cosine_min"]),
                fmt(overall["angle_deg_mean"], 2),
                fmt(agree["top1_overlap"]),
                fmt(agree["top10_overlap"]),
            ]
        )
    print_table(
        "Cosine drift vs pytorch (higher cosine / overlap is better)",
        [
            "backend",
            "cos queries",
            "cos corpus",
            "cos overall",
            "cos p05",
            "cos min",
            "angle° mean",
            "top1 overlap",
            "top10 overlap",
        ],
        drift_rows,
    )

    retrieval_rows: list[list[str]] = []
    ref_ndcg = per_backend["pytorch"]["retrieval"]["ndcg@10"]
    for name, backend in per_backend.items():
        r = backend["retrieval"]
        delta = r["ndcg@10"] - ref_ndcg
        retrieval_rows.append(
            [
                name,
                fmt(r["ndcg@10"]),
                fmt(r["ndcg@100"]),
                fmt(r["recall@10"]),
                fmt(r["recall@100"]),
                fmt(r["mrr@10"]),
                f"{delta:+.4f}",
            ]
        )
    print_table(
        "nfcorpus test retrieval (cosine ranking, L2-normalized)",
        ["backend", "nDCG@10", "nDCG@100", "recall@10", "recall@100", "MRR@10", "Δ nDCG@10"],
        retrieval_rows,
    )

    write_json(report, args.output)
    print(f"\nWrote {args.output}")

    min_cosine = {
        "onnx-fp32": args.min_cosine_fp32,
        "onnx-int8": args.min_cosine_int8,
    }
    max_drop = {
        "onnx-fp32": args.max_ndcg10_drop_fp32,
        "onnx-int8": args.max_ndcg10_drop_int8,
    }
    failures = qa_failures(report, min_cosine, max_drop)
    if failures:
        print("\nQA gates:")
        for item in failures:
            print(f"  FAIL {item}")
        if args.qa:
            return 1
    else:
        print("\nQA gates: pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
