import asyncio
import time
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from .model import predict


@dataclass(frozen=True)
class EmbedBatchResult:
    """Embeddings plus per-request batcher timings.

    ttft_ms: enqueue -> encode start (queue wait + batch collection).
    ttfr_ms: enqueue -> embeddings ready (ttft + encode).
    """

    embeddings: list[list[float]]
    ttft_ms: float
    ttfr_ms: float


@dataclass
class _Queued:
    texts: list[str]
    future: asyncio.Future[EmbedBatchResult]
    enqueued_at: float


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
        self._queue: asyncio.Queue[_Queued] = asyncio.Queue(maxsize=max_queue_size)
        self._worker_task: asyncio.Task[None] | None = None
        self._inflight = 0

    def start(self) -> None:
        """Launch the background worker as an asyncio task."""
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the background worker and drain remaining requests."""
        if self._worker_task is None:
            return
        try:
            self._worker_task.cancel()
            await self._worker_task
        except asyncio.CancelledError:
            pass
        try:
            while not self._queue.empty():
                item = self._queue.get_nowait()
                if not item.future.done():
                    item.future.set_exception(asyncio.CancelledError)
        except asyncio.QueueEmpty:
            pass

    def is_running(self) -> bool:
        """True when the background worker task is alive (used by the readiness check)."""
        return self._worker_task is not None and not self._worker_task.done()

    def queue_depth(self) -> int:
        """Number of requests currently waiting in the queue."""
        return self._queue.qsize()

    def inflight(self) -> int:
        """Number of accepted requests still awaiting their result."""
        return self._inflight

    async def submit(self, texts: list[str]) -> EmbedBatchResult:
        """Submit a request for batched inference. Called by the /embed endpoint.

        Rejects immediately with QueueFullError when the queue is at capacity (backpressure),
        and raises RequestTimeoutError if the result isn't ready within request_timeout_s.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        try:
            self._queue.put_nowait(_Queued(texts, future, time.perf_counter()))
        except asyncio.QueueFull:
            raise QueueFullError("request queue is full")
        self._inflight += 1
        try:
            return await asyncio.wait_for(future, timeout=self._request_timeout_s)
        except asyncio.TimeoutError:
            raise RequestTimeoutError("request timed out before inference completed")
        finally:
            self._inflight -= 1

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
            for item in batch:
                all_texts.extend(item.texts)
                sizes.append(sizes[-1] + len(item.texts))

            # run inference: encode the whole aggregated batch in one pass (batch_size =
            # max_batch_size), instead of model.encode's hidden default of 32.
            loop = asyncio.get_running_loop()
            encode_started = time.perf_counter()
            try:
                all_embeddings = await loop.run_in_executor(
                    None, predict, self._model, all_texts, self._max_batch_size
                )
                encode_finished = time.perf_counter()

                # split results back and resolve futures (skip ones already timed out/cancelled)
                for idx, item in enumerate(batch):
                    if not item.future.done():
                        item.future.set_result(
                            EmbedBatchResult(
                                embeddings=all_embeddings[sizes[idx] : sizes[idx + 1]],
                                ttft_ms=round((encode_started - item.enqueued_at) * 1000, 2),
                                ttfr_ms=round((encode_finished - item.enqueued_at) * 1000, 2),
                            )
                        )

            except Exception as e:
                # to handle raise in predict()
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(e)
