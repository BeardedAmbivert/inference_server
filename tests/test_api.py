"""Endpoint tests using FastAPI's TestClient with the model stubbed out.

The `client` fixture patches `app.main.load_model` so the lifespan loads the FakeModel
instead of downloading MiniLM, while still exercising the real batcher and endpoint code.
"""

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.batching import QueueFullError, RequestTimeoutError
from app.config import settings
from app.main import app


@pytest.fixture
def client(monkeypatch, make_model):
    """A TestClient whose lifespan loads a FakeModel instead of downloading MiniLM."""
    monkeypatch.setattr(main_module, "load_model", lambda *args, **kwargs: make_model())
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["worker_alive"] is True
    assert body["queue_depth"] == 0
    assert body["inflight"] == 0
    assert "model" in body
    assert "device" in body


def test_health_not_ready_returns_503(client, monkeypatch):
    """/health returns 503 when the batch worker is not alive."""
    monkeypatch.setattr(app.state.batcher, "is_running", lambda: False)
    resp = client.get("/health")

    assert resp.status_code == 503
    assert resp.json()["status"] == "not ready"


def test_request_id_generated(client):
    """Every response carries an X-Request-ID header."""
    resp = client.get("/health")
    assert resp.headers.get("X-Request-ID")


def test_request_id_echoed(client):
    """An inbound X-Request-ID is echoed back on the response."""
    resp = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert resp.headers.get("X-Request-ID") == "abc123"


def test_embed(client, make_model):
    resp = client.post("/embed", json={"texts": ["hello", "world"]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["num_texts"] == 2
    assert body["dim"] == 4
    assert len(body["embeddings"]) == 2
    assert body["embeddings"] == make_model().encode(["hello", "world"]).tolist()


def test_embed_rejects_empty_texts(client):
    """An empty texts list fails validation (422) before reaching the model."""
    assert client.post("/embed", json={"texts": []}).status_code == 422


def test_embed_rejects_too_many_texts(client, monkeypatch):
    """More than max_texts_per_request texts is rejected with 422."""
    monkeypatch.setattr(settings, "max_texts_per_request", 2)
    assert client.post("/embed", json={"texts": ["a", "b", "c"]}).status_code == 422


def test_embed_rejects_long_text(client, monkeypatch):
    """A text longer than max_chars_per_text is rejected with 422."""
    monkeypatch.setattr(settings, "max_chars_per_text", 5)
    assert client.post("/embed", json={"texts": ["this is too long"]}).status_code == 422


def test_embed_queue_full_returns_503(client, monkeypatch):
    """A QueueFullError from the batcher is mapped to 503 by the endpoint."""

    async def reject(texts):
        raise QueueFullError("full")

    monkeypatch.setattr(app.state.batcher, "submit", reject)
    assert client.post("/embed", json={"texts": ["x"]}).status_code == 503


def test_embed_timeout_returns_504(client, monkeypatch):
    """A RequestTimeoutError from the batcher is mapped to 504 by the endpoint."""

    async def time_out(texts):
        raise RequestTimeoutError("timeout")

    monkeypatch.setattr(app.state.batcher, "submit", time_out)
    assert client.post("/embed", json={"texts": ["x"]}).status_code == 504
