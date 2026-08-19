"""Embedding quality metrics: cosine drift, rank agreement, and IR retrieval.

Used by scripts/eval_quality.py and unit-tested in tests/test_quality_metrics.py.
No model or dataset dependency so the math can run in CI.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

Qrels = Mapping[str, Mapping[str, int | float]]


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalize. Zero rows stay zero (divided by eps)."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, eps)


def pairwise_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Cosine similarity of matching rows. left[i] vs right[i]."""
    a = l2_normalize(left)
    b = l2_normalize(right)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return np.sum(a * b, axis=1)


def cosine_matrix(queries: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """Dense query-document cosine matrix, shape (n_queries, n_docs)."""
    return l2_normalize(queries) @ l2_normalize(documents).T


def angular_error_deg(cosine: np.ndarray) -> np.ndarray:
    """Angle in degrees between two unit vectors given their cosine."""
    return np.degrees(np.arccos(np.clip(np.asarray(cosine, dtype=np.float64), -1.0, 1.0)))


def summarize_values(values: np.ndarray | Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p05": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": float(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p05": float(np.quantile(arr, 0.05)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def cosine_drift_report(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Per-row cosine and angular error of candidate embeddings vs a reference."""
    cosine = pairwise_cosine(reference, candidate)
    angle = angular_error_deg(cosine)
    out = {f"cosine_{k}": v for k, v in summarize_values(cosine).items()}
    out.update({f"angle_deg_{k}": v for k, v in summarize_values(angle).items()})
    return out


def ranked_doc_ids(scores: np.ndarray, doc_ids: Sequence[str]) -> list[list[str]]:
    """For each query row, document ids sorted by descending score."""
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-d, got {scores.shape}")
    if scores.shape[1] != len(doc_ids):
        raise ValueError(f"doc count {len(doc_ids)} != scores columns {scores.shape[1]}")
    order = np.argsort(-scores, axis=1, kind="stable")
    ids = list(doc_ids)
    return [[ids[j] for j in row] for row in order]


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """TREC/BEIR DCG: sum_i (2^{rel_i} - 1) / log2(i + 1), 1-based ranks."""
    if k <= 0:
        raise ValueError("k must be positive")
    total = 0.0
    for rank, rel in enumerate(list(gains)[:k], start=1):
        total += (2.0 ** float(rel) - 1.0) / math.log2(rank + 1)
    return total


def ndcg_at_k(ranked_ids: Sequence[str], qrel: Mapping[str, int | float], k: int) -> float:
    """nDCG@k for one query. Queries with no relevant docs score 0."""
    gains = [float(qrel.get(doc_id, 0)) for doc_id in ranked_ids[:k]]
    dcg = dcg_at_k(gains, k)
    ideal = sorted((float(rel) for rel in qrel.values() if rel > 0), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def recall_at_k(ranked_ids: Sequence[str], qrel: Mapping[str, int | float], k: int) -> float:
    """Fraction of relevant documents retrieved in the top k."""
    relevant = {doc_id for doc_id, rel in qrel.items() if rel > 0}
    if not relevant:
        return 0.0
    hit = sum(1 for doc_id in ranked_ids[:k] if doc_id in relevant)
    return hit / len(relevant)


def mrr_at_k(ranked_ids: Sequence[str], qrel: Mapping[str, int | float], k: int) -> float:
    """Reciprocal rank of the first relevant document, 0 if none in the top k."""
    relevant = {doc_id for doc_id, rel in qrel.items() if rel > 0}
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_retrieval_metrics(
    rankings: Mapping[str, Sequence[str]],
    qrels: Qrels,
    ks: Sequence[int] = (10, 100),
) -> dict[str, float]:
    """Macro-average nDCG/recall/MRR over queries that have at least one qrel."""
    scores: dict[str, list[float]] = {}
    for k in ks:
        scores[f"ndcg@{k}"] = []
        scores[f"recall@{k}"] = []
        scores[f"mrr@{k}"] = []

    n_scored = 0
    for query_id, qrel in qrels.items():
        if not any(rel > 0 for rel in qrel.values()):
            continue
        ranked = rankings.get(query_id)
        if ranked is None:
            continue
        n_scored += 1
        for k in ks:
            scores[f"ndcg@{k}"].append(ndcg_at_k(ranked, qrel, k))
            scores[f"recall@{k}"].append(recall_at_k(ranked, qrel, k))
            scores[f"mrr@{k}"].append(mrr_at_k(ranked, qrel, k))

    out: dict[str, float] = {"n_queries": float(n_scored)}
    for name, values in scores.items():
        out[name] = float(sum(values) / len(values)) if values else 0.0
    return out


def topk_overlap(left: Sequence[str], right: Sequence[str], k: int) -> float:
    """Jaccard-style overlap of two top-k lists: |A ∩ B| / k."""
    if k <= 0:
        raise ValueError("k must be positive")
    return len(set(left[:k]) & set(right[:k])) / k


def mean_rank_agreement(
    reference_rankings: Mapping[str, Sequence[str]],
    candidate_rankings: Mapping[str, Sequence[str]],
    ks: Sequence[int] = (1, 10),
) -> dict[str, float]:
    """How often the candidate ranking preserves the reference top-k."""
    shared = [qid for qid in reference_rankings if qid in candidate_rankings]
    out: dict[str, float] = {"n_queries": float(len(shared))}
    for k in ks:
        if not shared:
            out[f"top{k}_overlap"] = 0.0
            continue
        overlaps = [
            topk_overlap(reference_rankings[qid], candidate_rankings[qid], k) for qid in shared
        ]
        out[f"top{k}_overlap"] = float(sum(overlaps) / len(overlaps))
    return out
