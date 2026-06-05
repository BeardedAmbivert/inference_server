from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Hint: pydantic-settings reads from env vars automatically.
    Prefix with model_config = SettingsConfigDict(env_prefix="INFERENCE_") if you want
    namespaced env vars like INFERENCE_MODEL_NAME.
    """

    onnx_model_path: str = "models/minilm-onnx"
    model_name: str | None = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    backend: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    max_wait_ms: int = 500

    # Request limits & backpressure
    max_texts_per_request: int = 256  # benchmark sends 1 text/request; generous cap
    max_chars_per_text: int = 8192
    max_queue_size: int = 1000  # bounds memory; queue peaks ~32-64 under the c32 benchmark
    request_timeout_s: float = 30.0


settings = Settings()
