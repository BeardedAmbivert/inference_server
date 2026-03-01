from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Request body for the /predict endpoint.

    Consider adding validation:
    - Non-empty list (min_length=1)
    - Individual strings should be non-empty
    """

    texts: list[str]


class PredictResponse(BaseModel):
    """Response body for the /predict endpoint."""

    embeddings: list[list[float]]
    dim: int | None
    num_texts: int
