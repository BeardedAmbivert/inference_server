"""Download and sample BeIR/nfcorpus into local text pools for benchmarking.

nfcorpus is a small biomedical IR dataset whose two subsets give us the length contrast
SentenceTransformers warns about: short **queries** (latency workload) and long mixed-length
**corpus** documents (throughput workload).

Usage:
    uv sync --extra bench
    uv run python scripts/prepare_dataset.py            # default n=1000, seed=42

Writes (gitignored):
    benchmarks/data/nfcorpus-queries.jsonl   short query text   -> latency workload
    benchmarks/data/nfcorpus-corpus.jsonl    title + abstract   -> throughput workload
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from utils import length_stats, sample_texts

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "benchmarks" / "data"
MAX_CHARS = 8192  # mirror app MAX_CHARS_PER_TEXT so HTTP runs don't hit 422 on long docs


def build_pool(rows, to_text) -> list[str]:
    """Map dataset rows to cleaned, de-duplicated, length-capped strings."""
    seen: set[str] = set()
    pool: list[str] = []
    for row in rows:
        text = " ".join(to_text(row).split())  # collapse newlines/whitespace
        if not text:
            continue
        text = text[:MAX_CHARS]
        if text in seen:
            continue
        seen.add(text)
        pool.append(text)
    return pool


def write_jsonl(texts: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample BeIR/nfcorpus into benchmark text pools.")
    parser.add_argument("--n", type=int, default=1000, help="Max texts to keep per pool.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (reproducible).")
    args = parser.parse_args()

    # queries -> short query text (latency workload)
    queries = load_dataset("BeIR/nfcorpus", "queries")["queries"]
    query_pool = build_pool(queries, lambda r: r["text"] or r["title"])

    # corpus -> title + abstract (throughput workload, mixed/long)
    corpus = load_dataset("BeIR/nfcorpus", "corpus")["corpus"]
    corpus_pool = build_pool(corpus, lambda r: f"{r['title']}. {r['text']}")

    query_sample = sample_texts(query_pool, min(args.n, len(query_pool)), args.seed)
    corpus_sample = sample_texts(corpus_pool, min(args.n, len(corpus_pool)), args.seed)

    write_jsonl(query_sample, DATA_DIR / "nfcorpus-queries.jsonl")
    write_jsonl(corpus_sample, DATA_DIR / "nfcorpus-corpus.jsonl")

    print(f"queries ({len(query_pool)} available) -> {length_stats(query_sample)}")
    print(f"corpus  ({len(corpus_pool)} available) -> {length_stats(corpus_sample)}")
    print(f"Wrote {DATA_DIR}/nfcorpus-queries.jsonl and nfcorpus-corpus.jsonl")


if __name__ == "__main__":
    main()
