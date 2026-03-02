# inference_server

Embedding inference server built with FastAPI + PyTorch serving `all-MiniLM-L6-v2` via HTTP. Supports dynamic batching for concurrent requests.

## Run locally

```bash
uv sync
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Docker

```bash
docker build -t inference-server .
docker run -p 7860:7860 inference-server
```

Image uses CPU-only PyTorch with the model baked in (no runtime download).

## Endpoints

- `GET /health` — health check
- `POST /predict` — generate embeddings
  ```bash
  curl -X POST http://localhost:7860/predict \
    -H "Content-Type: application/json" \
    -d '{"texts": ["hello world"]}'
  ```