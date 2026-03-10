from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .schemas import EmbedRequest, EmbedResponse
from .model import load_model
from .config import settings
from .batching import DynamicBatcher

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager — runs on startup and shutdown.

    The 'yield' separates startup from shutdown.
    After yield, add any cleanup logic if needed (e.g. logging shutdown).

    Docs: https://fastapi.tiangolo.com/advanced/events/#lifespan
    """
    # Startup
    model = load_model(settings.model_name, settings.device)
    app.state.model = model
    app.state.batcher = DynamicBatcher(model, settings.max_batch_size, settings.max_wait_ms)
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
    """Generate embeddings for input texts."""
    result = await app.state.batcher.submit(body.texts)
    return {
        "embeddings": result,
        "dim": app.state.model.get_sentence_embedding_dimension(),
        "num_texts": len(body.texts)
    }
