"""Unit tests for embedding quality metrics. No model download, no BeIR."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from eval_quality import DEFAULT_MAX_NDCG10_DROP, DEFAULT_MIN_COSINE, qa_failures
from quality_metrics import (
    angular_error_deg,
    cosine_drift_report,
    cosine_matrix,
    dcg_at_k,
    l2_normalize,
    mean_rank_agreement,
    mean_retrieval_metrics,
    mrr_at_k,
    ndcg_at_k,
    pairwise_cosine,
    ranked_doc_ids,
    recall_at_k,
    summarize_values,
    topk_overlap,
)


def test_l2_normalize_unit_rows():
    raw = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)
    normed = l2_normalize(raw)
    assert normed == pytest.approx(np.array([[0.6, 0.8], [0.0, 1.0]]))


def test_pairwise_cosine_identical_and_orthogonal():
    a = np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]])
    b = np.array([[2.0, 0.0], [0.0, 3.0], [-1.0, -1.0]])
    cos = pairwise_cosine(a, b)
    assert cos[0] == pytest.approx(1.0)
    assert cos[1] == pytest.approx(0.0)
    assert cos[2] == pytest.approx(-1.0)


def test_pairwise_cosine_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        pairwise_cosine(np.ones((2, 3)), np.ones((3, 3)))


def test_cosine_matrix_self_is_identity_on_orthonormal_rows():
    queries = np.eye(3)
    scores = cosine_matrix(queries, queries)
    assert scores == pytest.approx(np.eye(3))


def test_angular_error_right_angle():
    assert angular_error_deg(np.array([0.0]))[0] == pytest.approx(90.0)
    assert angular_error_deg(np.array([1.0]))[0] == pytest.approx(0.0)


def test_summarize_values_percentiles():
    stats = summarize_values([1.0, 2.0, 3.0, 4.0])
    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0
    assert stats["p50"] == pytest.approx(2.5)


def test_cosine_drift_report_near_identity():
    reference = np.array([[1.0, 0.0], [0.0, 2.0]])
    candidate = np.array([[1.0, 0.0], [0.01, 2.0]])
    report = cosine_drift_report(reference, candidate)
    assert report["cosine_mean"] > 0.999
    assert report["angle_deg_mean"] < 1.0
    assert report["cosine_min"] <= report["cosine_p50"] <= report["cosine_max"]


def test_dcg_matches_trec_formula():
    # gains [1, 0, 1] -> (2^1-1)/log2(2) + 0 + (2^1-1)/log2(4) = 1 + 0.5
    assert dcg_at_k([1, 0, 1], k=3) == pytest.approx(1.5)


def test_ndcg_perfect_ranking_is_one():
    ranked = ["d1", "d2", "d3"]
    qrel = {"d1": 1, "d2": 1}
    assert ndcg_at_k(ranked, qrel, k=3) == pytest.approx(1.0)


def test_ndcg_known_imperfect_ranking():
    # 3 relevant docs, ranking: rel, irrel, rel, irrel, rel
    ranked = ["d1", "x", "d2", "y", "d3"]
    qrel = {"d1": 1, "d2": 1, "d3": 1}
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4) + 1.0 / math.log2(6)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
    assert ndcg_at_k(ranked, qrel, k=5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(ranked, {}, k=5) == 0.0


def test_recall_and_mrr():
    ranked = ["x", "d1", "y"]
    qrel = {"d1": 1, "d2": 1}
    assert recall_at_k(ranked, qrel, k=1) == pytest.approx(0.0)
    assert recall_at_k(ranked, qrel, k=2) == pytest.approx(0.5)
    assert mrr_at_k(ranked, qrel, k=1) == pytest.approx(0.0)
    assert mrr_at_k(ranked, qrel, k=3) == pytest.approx(0.5)


def test_ranked_doc_ids_stable_on_ties():
    scores = np.array([[0.1, 0.5, 0.5]])
    ranked = ranked_doc_ids(scores, ["a", "b", "c"])
    assert ranked[0][0] in {"b", "c"}
    assert set(ranked[0]) == {"a", "b", "c"}


def test_mean_retrieval_metrics_macro_average():
    rankings = {
        "q1": ["d1", "d2", "d3"],
        "q2": ["d3", "d1", "d2"],
    }
    qrels = {
        "q1": {"d1": 1},
        "q2": {"d2": 1},
        "empty": {},
    }
    metrics = mean_retrieval_metrics(rankings, qrels, ks=(1, 2))
    assert metrics["n_queries"] == 2
    # q1: first hit at rank 1; q2: first hit at rank 3 (miss @1 and @2)
    assert metrics["mrr@1"] == pytest.approx(0.5)
    assert metrics["recall@1"] == pytest.approx(0.5)
    assert metrics["ndcg@1"] == pytest.approx(0.5)


def test_qa_failures_pass_and_fail():
    report = {
        "backends": {
            "pytorch": {"retrieval": {"ndcg@10": 0.3200}},
            "onnx-fp32": {
                "retrieval": {"ndcg@10": 0.3198},
                "drift": {"overall": {"cosine_mean": 0.9995}},
            },
            "onnx-int8": {
                "retrieval": {"ndcg@10": 0.3000},
                "drift": {"overall": {"cosine_mean": 0.90}},
            },
        }
    }
    min_cosine = {"onnx-fp32": 0.995, "onnx-int8": 0.95}
    max_drop = {"onnx-fp32": 0.005, "onnx-int8": 0.01}
    failures = qa_failures(report, min_cosine, max_drop)
    assert any("onnx-int8 mean cosine" in item for item in failures)
    assert any("onnx-int8 nDCG@10 drop" in item for item in failures)
    assert not any(item.startswith("onnx-fp32") for item in failures)

    passing = {
        "backends": {
            "pytorch": {"retrieval": {"ndcg@10": 0.32}},
            "onnx-int8": {
                "retrieval": {"ndcg@10": 0.319},
                "drift": {"overall": {"cosine_mean": 0.99}},
            },
        }
    }
    assert qa_failures(passing, min_cosine, max_drop) == []


def test_committed_quality_json_passes_default_qa_gates():
    """The published nfcorpus run must stay inside the documented --qa floors."""
    path = Path(__file__).resolve().parent.parent / "benchmarks" / "quality-nfcorpus.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert qa_failures(report, DEFAULT_MIN_COSINE, DEFAULT_MAX_NDCG10_DROP) == []


def test_topk_overlap_and_rank_agreement():
    assert topk_overlap(["a", "b", "c"], ["a", "c", "d"], k=2) == pytest.approx(0.5)
    agreement = mean_rank_agreement(
        {"q1": ["a", "b"], "q2": ["c", "d"]},
        {"q1": ["a", "x"], "q2": ["c", "d"]},
        ks=(1, 2),
    )
    assert agreement["n_queries"] == 2
    assert agreement["top1_overlap"] == pytest.approx(1.0)
    assert agreement["top2_overlap"] == pytest.approx(0.75)
