"""
tests/test_hypha.py — Tests for rpc/hypha_client.py ASGI endpoints.

Unit tests use the FastAPI TestClient (no network, no hardware).
Integration tests (marked @integration) hit the live Hypha server and
require HYPHA_WORKSPACE_TOKEN + HYPHA_WORKSPACE env vars.
"""

import os
import asyncio
import queue

import pytest
from starlette.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _inject_mocks(mock_mic, mock_whisper):
    """Inject test doubles into the hypha_client module globals."""
    import rpc.hypha_client as hc
    hc._mic = mock_mic
    hc._whisper = mock_whisper
    hc._start_time = 0.0


# ---------------------------------------------------------------------------
# Unit tests — no network / no hardware
# ---------------------------------------------------------------------------

def test_health_endpoint(mock_mic, mock_whisper):
    """GET /health returns status=ok JSON."""
    _inject_mocks(mock_mic, mock_whisper)
    from rpc.hypha_client import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert data["model"] == "mock"


def test_transcript_feed_returns_sse_content_type(mock_mic, mock_whisper):
    """GET /transcript_feed responds with text/event-stream content type."""
    _inject_mocks(mock_mic, mock_whisper)
    from rpc.hypha_client import app

    client = TestClient(app)
    with client.stream("GET", "/transcript_feed") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]


def test_transcript_feed_emits_data_line(mock_mic, mock_whisper):
    """GET /transcript_feed streams at least one 'data:' SSE line."""
    _inject_mocks(mock_mic, mock_whisper)
    from rpc.hypha_client import app

    client = TestClient(app)
    received = []
    with client.stream("GET", "/transcript_feed") as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                received.append(line)
                break  # got first chunk — disconnect

    assert len(received) == 1
    assert "Hello from mock Whisper" in received[0]


def test_transcript_feed_empty_queue_sends_keepalive():
    """When the mic queue is empty the feed emits a keep-alive SSE comment."""
    import rpc.hypha_client as hc
    from rpc.hypha_client import app
    from tests.conftest import MockMicCapture, MockWhisperEngine

    # Empty queue — no audio chunks
    hc._mic = MockMicCapture([])
    hc._whisper = MockWhisperEngine(["ignored"])
    hc._start_time = 0.0

    client = TestClient(app)
    received = []
    with client.stream("GET", "/transcript_feed") as resp:
        for line in resp.iter_lines():
            if line.startswith(":"):  # SSE comment = keep-alive
                received.append(line)
                break

    assert len(received) == 1
    assert "keep-alive" in received[0]


def test_queue_drained_after_disconnect(tone_pcm):
    """After client disconnects, stale chunks must be removed from the queue."""
    import rpc.hypha_client as hc
    from rpc.hypha_client import app
    from tests.conftest import MockMicCapture, MockWhisperEngine

    # Pre-load 3 chunks
    mic = MockMicCapture([tone_pcm, tone_pcm, tone_pcm])
    hc._mic = mic
    hc._whisper = MockWhisperEngine(["chunk"])
    hc._start_time = 0.0

    client = TestClient(app)
    with client.stream("GET", "/transcript_feed") as resp:
        for line in resp.iter_lines():
            if line.startswith("data:"):
                break  # consume one chunk then disconnect

    # Queue should be empty after disconnect (drained in finally block)
    assert hc._mic.queue.empty()


# ---------------------------------------------------------------------------
# Integration tests — require live Hypha + token
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_hypha_connection():
    """Can connect to hypha.aicell.io with the workspace token."""
    token = os.environ.get("HYPHA_WORKSPACE_TOKEN", "")
    workspace = os.environ.get("HYPHA_WORKSPACE", "")
    if not token:
        pytest.skip("HYPHA_WORKSPACE_TOKEN not set")

    from hypha_rpc import connect_to_server
    cfg = {"server_url": "https://hypha.aicell.io/", "token": token}
    if workspace:
        cfg["workspace"] = workspace

    server = await connect_to_server(cfg)
    assert server.config.workspace
    await server.disconnect()


@pytest.mark.integration
async def test_asgi_service_registers_and_responds(mock_mic, mock_whisper):
    """HyphaClient registers the ASGI service and /health responds via HTTPS."""
    token = os.environ.get("HYPHA_WORKSPACE_TOKEN", "")
    workspace = os.environ.get("HYPHA_WORKSPACE", "")
    if not token:
        pytest.skip("HYPHA_WORKSPACE_TOKEN not set")

    import httpx
    from rpc.hypha_client import HyphaClient

    client = HyphaClient(
        server_url="https://hypha.aicell.io/",
        workspace=workspace,
        token=token,
        mic_capture=mock_mic,
        whisper_engine=mock_whisper,
    )

    # Register the service (don't call run() — that blocks forever)
    await client._connect_with_backoff()
    await client._register()

    ws = client._server.config.workspace
    url = f"https://hypha.aicell.io/{ws}/apps/hypha-whisper/health"

    async with httpx.AsyncClient() as http:
        resp = await http.get(url, timeout=10)

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"

    await client._server.disconnect()
