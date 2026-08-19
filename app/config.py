from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    """Server settings. Each field is overridable by the matching env var (uppercase)."""

    onnx_model_path: str = "models/minilm-onnx"
    # ONNX backend file under onnx_model_path. Set ONNX_FILE_NAME=onnx/model_int8.onnx
    # to serve the INT8-quantized model (see scripts/quantize_onnx.py).
    onnx_file_name: str = "onnx/model_O3.onnx"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    backend: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    max_wait_ms: int = 500
    log_level: str = "INFO"

    # Request limits & backpressure
    max_texts_per_request: int = 256  # latency benches send up to 32; generous cap
    max_chars_per_text: int = 8192
    max_queue_size: int = 1000  # bounds memory; queue peaks ~32-64 under the c32 benchmark
    request_timeout_s: float = 30.0


settings = Settings()
