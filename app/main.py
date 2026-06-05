import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .schemas import EmbedRequest, EmbedResponse
from .model import load_model
from .config import settings
from .batching import DynamicBatcher, QueueFullError, RequestTimeoutError
from .logging_config import configure_logging, request_id_var

logger = logging.getLogger("inference_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager — runs on startup and shutdown.

    The 'yield' separates startup from shutdown.
    Docs: https://fastapi.tiangolo.com/advanced/events/#lifespan
    """
    # Startup
    configure_logging(settings.log_level)
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
    logger.info("model loaded", extra={"device": settings.device, "model": settings.model_name})
    yield
    await app.state.batcher.stop()


app = FastAPI(
    title="Embedding Inference Server",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag each request with a correlation ID, time it, and emit a structured access log.

    The ID comes from an inbound `X-Request-ID` header or is generated; it is stored in a
    ContextVar so every log during the request carries it, and echoed back on the response.
    """
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        # Logged while the contextvar is still set, so the access log carries the request_id.
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        request_id_var.reset(token)


@app.get("/health")
async def health():
    """Readiness probe.

    Returns 200 when the model is loaded and the batch worker is alive, otherwise 503,
    plus live queue/in-flight numbers for quick operational visibility.
    """
    model_loaded = getattr(app.state, "model", None) is not None
    batcher = getattr(app.state, "batcher", None)
    worker_alive = batcher is not None and batcher.is_running()
    ready = model_loaded and worker_alive
    body = {
        "status": "ready" if ready else "not ready",
        "model": settings.model_name,
        "device": settings.device,
        "model_loaded": model_loaded,
        "worker_alive": worker_alive,
        "queue_depth": batcher.queue_depth() if batcher is not None else None,
        "inflight": batcher.inflight() if batcher is not None else None,
        "max_queue_size": settings.max_queue_size,
    }
    return JSONResponse(status_code=200 if ready else 503, content=body)


@app.post("/embed", response_model=EmbedResponse)
async def embed_endpoint(body: EmbedRequest):
    """Generate embeddings for input texts.

    Input validation failures return 422 (handled by Pydantic). Under load the batcher may
    reject with 503 (queue full) or 504 (timeout); unexpected inference errors return a
    sanitized 500 (the traceback is logged server-side, not leaked to the client).
    """
    try:
        result = await app.state.batcher.submit(body.texts)
    except QueueFullError:
        raise HTTPException(status_code=503, detail="server overloaded, retry later")
    except RequestTimeoutError:
        raise HTTPException(status_code=504, detail="inference timed out")
    except Exception:
        logger.exception("inference failed")
        raise HTTPException(status_code=500, detail="inference failed")
    return {
        "embeddings": result,
        "dim": app.state.model.get_sentence_embedding_dimension(),
        "num_texts": len(body.texts),
    }
