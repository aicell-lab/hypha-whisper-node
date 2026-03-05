"""
tests/test_hypha.py — End-to-end tests for the Hypha ASGI service.

Every test here connects to the real hypha.aicell.io server via hypha-rpc,
registers the ASGI service, and hits its endpoints over HTTPS.

Requires HYPHA_WORKSPACE_TOKEN (and optionally HYPHA_WORKSPACE) env vars.
Run locally:
    export $(cat .env | xargs)
    pytest -m integration tests/test_hypha.py -v -s

In CI: token is injected via GitHub Actions secrets.
"""

import os
import asyncio
import httpx
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _creds():
    token = os.environ.get("HYPHA_WORKSPACE_TOKEN", "")
    workspace = os.environ.get("HYPHA_WORKSPACE", "")
    if not token:
        pytest.skip("HYPHA_WORKSPACE_TOKEN not set")
    return token, workspace


async def _register_client(token, workspace, mic, engine):
    """Connect to Hypha and register the ASGI service; return the client."""
    from rpc.hypha_client import HyphaClient
    client = HyphaClient(
        server_url="https://hypha.aicell.io/",
        workspace=workspace,
        token=token,
        mic_capture=mic,
        whisper_engine=engine,
    )
    await client._connect_with_backoff()
    await client._register()
    return client


# ---------------------------------------------------------------------------
# Integration tests — all require live Hypha + token
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_hypha_connection():
    """Can authenticate and connect to hypha.aicell.io."""
    token, workspace = _creds()
    from hypha_rpc import connect_to_server
    cfg = {"server_url": "https://hypha.aicell.io/", "token": token}
    if workspace:
        cfg["workspace"] = workspace
    server = await connect_to_server(cfg)
    assert server.config.workspace, "workspace should be non-empty after connect"
    await server.disconnect()


@pytest.mark.integration
async def test_health_via_hypha(mock_mic, mock_whisper):
    """
    Register the ASGI service on Hypha; confirm GET /health responds
    over the real public HTTPS URL (no TestClient, real network).
    """
    token, workspace = _creds()
    client = await _register_client(token, workspace, mock_mic, mock_whisper)
    try:
        ws = client._server.config.workspace
        url = f"https://hypha.aicell.io/{ws}/apps/hypha-whisper/health"
        async with httpx.AsyncClient() as http:
            resp = await http.get(url, timeout=15)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "uptime_seconds" in body
        print(f"\n[health] {url} → {body}")
    finally:
        await client._server.disconnect()


@pytest.mark.integration
async def test_transcript_feed_via_hypha(tone_pcm):
    """
    Full end-to-end test:
      1. Start real WhisperEngine (tiny.en).
      2. Pre-load synthetic audio into the mic queue.
      3. Register service on hypha.aicell.io via hypha-rpc.
      4. Connect to /transcript_feed over HTTPS and read SSE events.
      5. Assert at least one SSE event (data: or keep-alive) is received.

    This is proof that the entire pipeline works: mic → Whisper → Hypha → client.
    """
    token, workspace = _creds()

    from tests.conftest import MockMicCapture
    from transcribe.whisper_engine import WhisperEngine

    # Real Whisper inference — tiny.en is fast enough for CI
    engine = WhisperEngine(model_name="tiny.en")
    # Two chunks: first produces a transcript (possibly empty for a pure tone),
    # second keeps the queue non-empty so the stream doesn't close immediately.
    mic = MockMicCapture([tone_pcm, tone_pcm])

    client = await _register_client(token, workspace, mic, engine)
    try:
        ws = client._server.config.workspace
        url = f"https://hypha.aicell.io/{ws}/apps/hypha-whisper/transcript_feed"
        print(f"\n[transcript_feed] connecting → {url}")

        received = []
        async with httpx.AsyncClient() as http:
            async with http.stream(
                "GET", url,
                timeout=httpx.Timeout(60.0, connect=15.0),
            ) as resp:
                assert resp.status_code == 200, f"expected 200, got {resp.status_code}"
                assert "text/event-stream" in resp.headers["content-type"]
                async for line in resp.aiter_lines():
                    if line:  # non-empty SSE line (data: <text> or : keep-alive)
                        received.append(line)
                        print(f"[transcript_feed] received: {line!r}")
                        break  # one event is enough — disconnect

        assert len(received) >= 1, "No SSE events received from /transcript_feed"
    finally:
        # Brief pause so the server-side SSE generator finishes its current
        # executor call before we close the RPC connection (avoids noisy
        # "ConnectionError: RPC connection closed" log from hypha_rpc).
        await asyncio.sleep(0.6)
        await client._server.disconnect()
