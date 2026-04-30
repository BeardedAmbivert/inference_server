from sentence_transformers import SentenceTransformer


def load_model(model_name: str, device: str, backend: str | None = None) -> SentenceTransformer:
    """Load a SentenceTransformer model onto the specified device.

    Args:
        model_name: HuggingFace model ID (e.g. "sentence-transformers/all-MiniLM-L6-v2")
        device: torch device string ("cpu", "mps", "cuda")

    Returns:
        Loaded SentenceTransformer model ready for inference
    """
    if backend == "onnx":
        model = SentenceTransformer(
            model_name,
            backend="onnx",
            model_kwargs={"file_name": "onnx/model_O3.onnx"}
        )
    else:
        model = SentenceTransformer(model_name, device=device)
    return model


def predict(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a list of text strings.

    Args:
        model: Loaded SentenceTransformer model
        texts: List of strings to embed

    Returns:
        List of embedding vectors, each a list of floats
    """
    embeddings = model.encode(texts).tolist()
    return embeddings
