import asyncio
from sentence_transformers import SentenceTransformer

from .model import predict


class QueueFullError(Exception):
    """Raised when the request queue is at capacity (maps to HTTP 503)."""


class RequestTimeoutError(Exception):
    """Raised when a request is not served within request_timeout_s (maps to HTTP 504)."""


class DynamicBatcher:
    """Collects concurrent requests into batches for efficient inference.
    Flow:
    1. Endpoint calls submit(texts) → creates a Future, puts (texts, future) on queue
    2. Background _worker loop collects items from queue
    3. When max_batch_size reached OR max_wait_ms elapsed → flush:
       - Flatten all texts into one list
       - Run predict(model, all_texts) via run_in_executor (non-blocking)
       - Split results back by request, set each future's result
    4. submit() awaits its future and returns the result to the endpoint
    """

    def __init__(
        self,
        model: SentenceTransformer,
        max_batch_size: int,
        max_wait_ms: int,
        max_queue_size: int = 0,
        request_timeout_s: float | None = None,
    ):
        """Initialize the batcher.

        max_queue_size=0 leaves the queue unbounded; request_timeout_s=None disables the
        per-request deadline.
        """
        self._model = model
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._request_timeout_s = request_timeout_s
        self._queue = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task = None

    def start(self) -> None:
        """Launch the background worker as an asyncio task."""
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the background worker and drain remaining requests."""

        try:
            self._worker_task.cancel()
            await self._worker_task
        except asyncio.CancelledError:
            pass
        try:
            while not self._queue.empty():
                texts, future = self._queue.get_nowait()
                if not future.done():
                    future.set_exception(asyncio.CancelledError)
        except asyncio.QueueEmpty:
            pass

    async def submit(self, texts: list[str]) -> list[list[float]]:
        """Submit a request for batched inference. Called by the /embed endpoint.

        Rejects immediately with QueueFullError when the queue is at capacity (backpressure),
        and raises RequestTimeoutError if the result isn't ready within request_timeout_s.
        """
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        try:
            self._queue.put_nowait((texts, future))
        except asyncio.QueueFull:
            raise QueueFullError("request queue is full")
        try:
            return await asyncio.wait_for(future, timeout=self._request_timeout_s)
        except asyncio.TimeoutError:
            raise RequestTimeoutError("request timed out before inference completed")

    async def _worker(self) -> None:
        """Background loop that collects and processes batches."""
        while True:
            batch = []
            first_item = await self._queue.get()
            batch.append(first_item)
            while len(batch) < self._max_batch_size:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=self._max_wait_ms / 1000)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            # flatten texts
            all_texts, sizes = [], [0]
            for texts, _ in batch:
                all_texts.extend(texts)
                sizes.append(sizes[-1] + len(texts))

            # run inference
            loop = asyncio.get_event_loop()
            try:
                all_embeddings = await loop.run_in_executor(None, predict, self._model, all_texts)

                # split results back and resolve futures (skip ones already timed out/cancelled)
                for idx, (_, future) in enumerate(batch):
                    if not future.done():
                        future.set_result(all_embeddings[sizes[idx]:sizes[idx+1]])

            except Exception as e:
                # to handle raise in predict()
                for _, future in batch:
                    if not future.done():
                        future.set_exception(e)
