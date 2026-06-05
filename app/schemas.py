from pydantic import BaseModel, Field, field_validator

from .config import settings


class EmbedRequest(BaseModel):
    """Request body for the /embed endpoint.

    Limits are configurable in app/config.py; the validator reads them at call time so
    env overrides take effect. FastAPI surfaces validation failures as 422 responses.
    - texts must be a non-empty list (at most settings.max_texts_per_request items)
    - each text must be non-empty and at most settings.max_chars_per_text characters
    """

    texts: list[str] = Field(min_length=1)

    @field_validator("texts")
    @classmethod
    def _check_limits(cls, texts: list[str]) -> list[str]:
        if len(texts) > settings.max_texts_per_request:
            raise ValueError(
                f"too many texts: {len(texts)} > max {settings.max_texts_per_request}"
            )
        for text in texts:
            if len(text) < 1:
                raise ValueError("each text must be non-empty")
            if len(text) > settings.max_chars_per_text:
                raise ValueError(
                    f"text too long: {len(text)} chars > max {settings.max_chars_per_text}"
                )
        return texts


class EmbedResponse(BaseModel):
    """Response body for the /embed endpoint."""

    embeddings: list[list[float]]
    dim: int | None
    num_texts: int
