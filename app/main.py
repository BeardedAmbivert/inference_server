from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from .schemas import EmbedRequest, EmbedResponse
from .model import load_model
from .config import settings
from .batching import DynamicBatcher, QueueFullError, RequestTimeoutError

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager — runs on startup and shutdown.

    The 'yield' separates startup from shutdown.
    After yield, add any cleanup logic if needed (e.g. logging shutdown).

    Docs: https://fastapi.tiangolo.com/advanced/events/#lifespan
    """
    # Startup
    model_path = settings.onnx_model_path if settings.backend == "onnx" else settings.model_name
    model = load_model(model_path, settings.device, settings.backend)
    app.state.model = model
    app.state.batcher = DynamicBatcher(
        model,
        settings.max_batch_size,
        settings.max_wait_ms,
        max_queue_size=settings.max_queue_size,
        request_timeout_s=settings.request_timeout_s,
    )
    app.state.batcher.start()
    print(f"model loaded on {settings.device}")
    yield
    await app.state.batcher.stop()


app = FastAPI(
    title="Embedding Inference Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    """Health check endpoint.

    This lets you verify the server is running and which model is loaded.
    """
    return {
        "status": "ok",
        "model" : settings.model_name,
        "device": settings.device
    }


@app.post("/embed", response_model=EmbedResponse)
async def embed_endpoint(request: Request, body: EmbedRequest):
    """Generate embeddings for input texts.

    Input validation failures return 422 (handled by Pydantic). Under load the batcher may
    reject with 503 (queue full) or 504 (timeout); unexpected inference errors return a
    sanitized 500.
    """
    try:
        result = await app.state.batcher.submit(body.texts)
    except QueueFullError:
        raise HTTPException(status_code=503, detail="server overloaded, retry later")
    except RequestTimeoutError:
        raise HTTPException(status_code=504, detail="inference timed out")
    except Exception:
        raise HTTPException(status_code=500, detail="inference failed")
    return {
        "embeddings": result,
        "dim": app.state.model.get_sentence_embedding_dimension(),
        "num_texts": len(body.texts)
    }
