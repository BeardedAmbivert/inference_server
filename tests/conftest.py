"""Shared test fixtures.

`FakeModel` is a deterministic stand-in for SentenceTransformer with no weights and
no download. It plugs straight into the real `predict()` (it exposes `.encode()` /
`.get_sentence_embedding_dimension()`), so batcher and endpoint tests run fast on CPU.
Each text maps to a unique row, which lets tests assert exact request->response mapping
regardless of how requests get batched together.
"""

import numpy as np
import pytest


class FakeModel:
    def __init__(self, dim: int = 4, error: Exception | None = None, gate=None):
        self._dim = dim
        self._error = error  # if set, encode() raises it
        self._gate = gate     # if set, encode() blocks until the event is set

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._gate is not None:
            self._gate.wait()
        if self._error is not None:
            raise self._error
        return np.array(
            [[float(sum(map(ord, text)) + j) for j in range(self._dim)] for text in texts],
            dtype=float,
        )

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim


@pytest.fixture
def make_model():
    """Factory so each test can configure dim / error / gate."""
    def _make(dim: int = 4, error: Exception | None = None, gate=None) -> FakeModel:
        return FakeModel(dim=dim, error=error, gate=gate)
    return _make
