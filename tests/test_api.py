"""Endpoint tests using FastAPI's TestClient with the model stubbed out.

Patching `app.main.load_model` means the lifespan loads the FakeModel instead of
downloading MiniLM, while still exercising the real batcher and endpoint code.
"""

from fastapi.testclient import TestClient

import app.main as main_module
from app.main import app


def test_health(monkeypatch, make_model):
    monkeypatch.setattr(main_module, "load_model", lambda *args, **kwargs: make_model())
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model" in body
    assert "device" in body


def test_embed(monkeypatch, make_model):
    fake = make_model(dim=4)
    monkeypatch.setattr(main_module, "load_model", lambda *args, **kwargs: fake)
    with TestClient(app) as client:
        resp = client.post("/embed", json={"texts": ["hello", "world"]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["num_texts"] == 2
    assert body["dim"] == 4
    assert len(body["embeddings"]) == 2
    assert body["embeddings"] == fake.encode(["hello", "world"]).tolist()
