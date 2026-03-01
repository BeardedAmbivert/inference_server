from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from .schemas import PredictRequest, PredictResponse
from .model import load_model, predict
from .config import settings


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
    print(f"model loaded on {settings.device}")
    # raise NotImplementedError  # Replace with your startup logic
    yield
    # Shutdown (optional cleanup here)
    # model.clear()


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


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: Request, body: PredictRequest):
    """Generate embeddings for input texts."""
    model = request.app.state.model
    result = predict(model, body.texts)
    return {
        "embeddings": result,
        "dim": model.get_sentence_embedding_dimension(),
        "num_texts": len(body.texts)
    }