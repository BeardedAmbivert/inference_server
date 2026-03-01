from pydantic_settings import BaseSettings
import torch

class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Hint: pydantic-settings reads from env vars automatically.
    Prefix with model_config = SettingsConfigDict(env_prefix="INFERENCE_") if you want
    namespaced env vars like INFERENCE_MODEL_NAME.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    host: str = "0.0.0.0"
    port: int = 8000
    max_batch_size: int = 32
    max_wait_ms: int = 1


settings = Settings()
