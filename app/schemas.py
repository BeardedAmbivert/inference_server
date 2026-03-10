from pydantic import BaseModel


class EmbedRequest(BaseModel):
    """Request body for the /embed endpoint.

    Consider adding validation:
    - Non-empty list (min_length=1)
    - Individual strings should be non-empty
    """

    texts: list[str]


class EmbedResponse(BaseModel):
    """Response body for the /embed endpoint."""

    embeddings: list[list[float]]
    dim: int | None
    num_texts: int
