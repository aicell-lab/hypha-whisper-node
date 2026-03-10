"""
tests/test_multi_client_sse.py — Unit tests proving multi-client SSE fan-out correctness.

These tests run without Hypha, without a GPU, and without a microphone.
They verify the three bugs fixed by the broadcaster refactor:

  1. Both subscriber queues receive every transcript item (no item consumed by first reader).
  2. A new client connecting does NOT call engine.init_session() (no session reset).
  3. A client disconnecting does NOT call engine.finish_session() (no session teardown for others).
  4. Items produced after a client disconnects are NOT delivered to that client.
"""

import asyncio
import pytest
import rpc.hypha_client as _hc
from tests.conftest import MockStreamingEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_module(engine):
    """Inject mock engine and reset all fan-out state."""
    _hc._engine = engine
    _hc._text_queue = engine.text_queue
    _hc._subscribers = set()
    if _hc._broadcast_task is not None and not _hc._broadcast_task.done():
        _hc._broadcast_task.cancel()
    _hc._broadcast_task = None


async def _drain_broadcast(seconds: float = 0.3):
    """Run the event loop briefly so _broadcast_loop can process queued items."""
    await asyncio.sleep(seconds)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

async def test_two_clients_both_receive_every_segment():
    """
    Core fan-out test: three transcript segments are delivered to BOTH
    subscriber queues, not consumed by the first reader.
    """
    engine = MockStreamingEngine()
    _reset_module(engine)

    q1: asyncio.Queue = asyncio.Queue()
    q2: asyncio.Queue = asyncio.Queue()
    _hc._subscribers.add(q1)
    _hc._subscribers.add(q2)

    # Seed three items directly (simulates engine.process_audio() committing text)
    for msg in ["alpha", "beta", "gamma"]:
        engine.text_queue.put(msg)

    task = asyncio.create_task(_hc._broadcast_loop())
    await _drain_broadcast()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    items1 = []
    while not q1.empty():
        items1.append(q1.get_nowait())
    items2 = []
    while not q2.empty():
        items2.append(q2.get_nowait())

    assert items1 == ["alpha", "beta", "gamma"], f"Client 1 got: {items1}"
    assert items2 == ["alpha", "beta", "gamma"], f"Client 2 got: {items2}"


async def test_disconnected_client_stops_receiving():
    """
    Items produced after a client unsubscribes are NOT delivered to it,
    but continue to reach remaining subscribers.
    """
    engine = MockStreamingEngine()
    _reset_module(engine)

    q1: asyncio.Queue = asyncio.Queue()
    q2: asyncio.Queue = asyncio.Queue()
    _hc._subscribers.add(q1)
    _hc._subscribers.add(q2)

    engine.text_queue.put("msg1")

    task = asyncio.create_task(_hc._broadcast_loop())
    await _drain_broadcast(0.2)

    # Client 2 disconnects
    _hc._subscribers.discard(q2)

    engine.text_queue.put("msg2")
    await _drain_broadcast(0.2)

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    items1 = []
    while not q1.empty():
        items1.append(q1.get_nowait())
    items2 = []
    while not q2.empty():
        items2.append(q2.get_nowait())

    assert items1 == ["msg1", "msg2"], f"Client 1 (stayed) got: {items1}"
    assert items2 == ["msg1"], f"Client 2 (disconnected) got: {items2}"


async def test_new_client_connect_does_not_call_init_session():
    """
    Connecting to /transcript_feed must NOT call engine.init_session().
    Session lifecycle is managed by main.py at startup, not per-client.
    """
    engine = MockStreamingEngine()
    _reset_module(engine)

    init_calls = []
    original_init = engine.init_session

    def patched_init(offset=None):
        init_calls.append("init_session")
        return original_init(offset)

    engine.init_session = patched_init

    # Simulate what transcript_feed() does on connect: add a subscriber queue
    # and call _ensure_broadcast_loop().
    client_q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _hc._subscribers.add(client_q)
    _hc._ensure_broadcast_loop()

    # Give the event loop a chance to run — init_session must NOT appear.
    await asyncio.sleep(0.1)

    _hc._broadcast_task.cancel()
    try:
        await _hc._broadcast_task
    except asyncio.CancelledError:
        pass

    assert init_calls == [], (
        f"init_session was called {len(init_calls)} time(s) on client connect — "
        "this resets the engine mid-transcription for other clients"
    )


async def test_client_disconnect_does_not_call_finish_session():
    """
    Disconnecting from /transcript_feed must NOT call engine.finish_session().
    Calling it would flush and terminate the transcription for all other clients.
    """
    engine = MockStreamingEngine()
    _reset_module(engine)

    finish_calls = []
    original_finish = engine.finish_session

    def patched_finish():
        finish_calls.append("finish_session")
        return original_finish()

    engine.finish_session = patched_finish

    # Simulate connect
    client_q: asyncio.Queue = asyncio.Queue(maxsize=256)
    _hc._subscribers.add(client_q)
    _hc._ensure_broadcast_loop()

    await asyncio.sleep(0.05)

    # Simulate disconnect (what transcript_feed() finally: block does)
    _hc._subscribers.discard(client_q)

    await asyncio.sleep(0.05)

    _hc._broadcast_task.cancel()
    try:
        await _hc._broadcast_task
    except asyncio.CancelledError:
        pass

    assert finish_calls == [], (
        f"finish_session was called {len(finish_calls)} time(s) on client disconnect — "
        "this would terminate transcription for remaining clients"
    )


async def test_ten_clients_all_receive_segment():
    """Stress: 10 simultaneous subscribers all receive the same item."""
    engine = MockStreamingEngine()
    _reset_module(engine)

    queues = [asyncio.Queue() for _ in range(10)]
    for q in queues:
        _hc._subscribers.add(q)

    engine.text_queue.put("broadcast-to-all")

    task = asyncio.create_task(_hc._broadcast_loop())
    await _drain_broadcast(0.3)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    for i, q in enumerate(queues):
        assert not q.empty(), f"Client {i} queue is empty — did not receive item"
        assert q.get_nowait() == "broadcast-to-all", f"Client {i} got wrong item"
