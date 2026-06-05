"""Unit tests for DynamicBatcher (app/batching.py)."""

import asyncio
import contextlib
import threading

import pytest

from app.batching import DynamicBatcher, QueueFullError, RequestTimeoutError


async def test_size_trigger_flush(make_model):
    """A full batch flushes on the size trigger without waiting out max_wait_ms."""
    model = make_model()
    batcher = DynamicBatcher(model, max_batch_size=2, max_wait_ms=10_000)
    batcher.start()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(batcher.submit(["a"]), batcher.submit(["b"])), timeout=5
        )
    finally:
        await batcher.stop()

    assert results[0] == make_model().encode(["a"]).tolist()
    assert results[1] == make_model().encode(["b"]).tolist()


async def test_time_trigger_flush(make_model):
    """A partial batch flushes once max_wait_ms elapses."""
    batcher = DynamicBatcher(make_model(), max_batch_size=10, max_wait_ms=50)
    batcher.start()
    try:
        result = await asyncio.wait_for(batcher.submit(["solo"]), timeout=5)
    finally:
        await batcher.stop()

    assert result == make_model().encode(["solo"]).tolist()


async def test_request_response_mapping(make_model):
    """Mixed-size requests batched together each get back exactly their own rows."""
    batcher = DynamicBatcher(make_model(), max_batch_size=8, max_wait_ms=200)
    batcher.start()
    requests = [["a"], ["b", "c"], ["d", "e", "f"]]
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*(batcher.submit(req) for req in requests)), timeout=5
        )
    finally:
        await batcher.stop()

    model = make_model()
    for req, res in zip(requests, results):
        assert res == model.encode(req).tolist()


async def test_exception_propagation(make_model):
    """An inference error surfaces to the awaiting submit()."""
    batcher = DynamicBatcher(
        make_model(error=RuntimeError("model failed")), max_batch_size=1, max_wait_ms=50
    )
    batcher.start()
    try:
        with pytest.raises(RuntimeError, match="model failed"):
            await asyncio.wait_for(batcher.submit(["x"]), timeout=5)
    finally:
        await batcher.stop()


async def test_start_stop_lifecycle(make_model):
    """start() launches the worker; stop() ends it cleanly."""
    batcher = DynamicBatcher(make_model(), max_batch_size=4, max_wait_ms=50)
    batcher.start()
    assert batcher._worker_task is not None
    await batcher.stop()
    assert batcher._worker_task.done()


async def test_shutdown_drains_queued_requests(make_model):
    """Requests still queued at shutdown are failed with CancelledError."""
    gate = threading.Event()
    # batch_size=1 so the worker takes request A and immediately enters encode(), which blocks.
    batcher = DynamicBatcher(make_model(gate=gate), max_batch_size=1, max_wait_ms=50)
    batcher.start()
    try:
        a = asyncio.create_task(batcher.submit(["A"]))
        await asyncio.sleep(0.1)  # let the worker pull A and block inside the executor
        b = asyncio.create_task(batcher.submit(["B"]))
        await asyncio.sleep(0.05)  # B is now sitting in the queue

        await batcher.stop()

        with pytest.raises(asyncio.CancelledError):
            await b
    finally:
        gate.set()  # unblock the executor thread so it can exit
        a.cancel()
        with pytest.raises(asyncio.CancelledError):
            await a


async def test_queue_full_rejects(make_model):
    """A submit beyond the queue's capacity is rejected immediately with QueueFullError."""
    gate = threading.Event()
    # batch_size=1: the worker pulls request A and blocks inside encode(); max_queue_size=1
    # leaves room for exactly one queued request (B) before the next submit is rejected.
    batcher = DynamicBatcher(
        make_model(gate=gate), max_batch_size=1, max_wait_ms=50, max_queue_size=1
    )
    batcher.start()
    a = asyncio.create_task(batcher.submit(["A"]))
    await asyncio.sleep(0.1)  # worker pulls A and blocks in the executor; queue is now empty
    b = asyncio.create_task(batcher.submit(["B"]))
    await asyncio.sleep(0.05)  # B is now sitting in the queue (at capacity)
    try:
        with pytest.raises(QueueFullError):
            await batcher.submit(["C"])  # queue at capacity -> rejected synchronously
    finally:
        gate.set()  # unblock the worker thread
        await batcher.stop()
        for task in (a, b):
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def test_request_timeout(make_model):
    """A request not served within request_timeout_s raises RequestTimeoutError, and the
    worker survives (the cancelled future is skipped) to serve later requests."""
    gate = threading.Event()
    batcher = DynamicBatcher(
        make_model(gate=gate), max_batch_size=1, max_wait_ms=50, request_timeout_s=0.1
    )
    batcher.start()
    try:
        with pytest.raises(RequestTimeoutError):
            await batcher.submit(["slow"])

        assert not batcher._worker_task.done()  # guard prevented a worker crash
        gate.set()  # let the stuck inference finish; the worker skips the cancelled future

        result = await asyncio.wait_for(batcher.submit(["after"]), timeout=5)
        assert result == make_model().encode(["after"]).tolist()
    finally:
        gate.set()
        await batcher.stop()
