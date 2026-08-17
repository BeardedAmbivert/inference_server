from sentence_transformers import SentenceTransformer


def load_model(
    model_name: str,
    device: str,
    backend: str | None = None,
    onnx_file_name: str = "onnx/model_O3.onnx",
) -> SentenceTransformer:
    """Load a SentenceTransformer model onto the specified device.

    Args:
        model_name: HuggingFace model ID (e.g. "sentence-transformers/all-MiniLM-L6-v2")
        device: torch device string ("cpu", "mps", "cuda")
        backend: "onnx" to load the ONNX Runtime backend, otherwise the default torch backend
        onnx_file_name: which ONNX graph to load under the model dir (e.g. the O3-optimized
            fp32 export or "onnx/model_int8.onnx" for the quantized model)

    Returns:
        Loaded SentenceTransformer model ready for inference
    """
    if backend == "onnx":
        model = SentenceTransformer(
            model_name,
            backend="onnx",
            model_kwargs={"file_name": onnx_file_name}
        )
    else:
        model = SentenceTransformer(model_name, device=device)
    return model


def predict(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Generate embeddings for a list of text strings.

    Args:
        model: Loaded SentenceTransformer model
        texts: List of strings to embed
        batch_size: Texts encoded per forward pass (model.encode default is 32)

    Returns:
        List of embedding vectors, each a list of floats
    """
    embeddings = model.encode(texts, batch_size=batch_size).tolist()
    return embeddings
