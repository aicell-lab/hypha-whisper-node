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
import uuid
import httpx
import pytest
import rpc.hypha_client as _hc_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _creds():
    token = os.environ.get("HYPHA_WORKSPACE_TOKEN", "")
    workspace = os.environ.get("HYPHA_WORKSPACE", "")
    if not token:
        pytest.skip("HYPHA_WORKSPACE_TOKEN not set")
    return token, workspace


async def _register_client(token, workspace, engine):
    """
    Connect to Hypha and register the ASGI service with a unique service ID
    so back-to-back test registrations never collide on the server.
    Returns (client, service_id).
    """
    from rpc.hypha_client import HyphaClient

    # Use a fresh unique ID for each test registration to avoid Hypha routing
    # stale connections when tests run back-to-back with the same service ID.
    svc_id = f"hypha-whisper-test-{uuid.uuid4().hex[:8]}"
    _hc_module.SERVICE_ID = svc_id

    client = HyphaClient(
        server_url="https://hypha.aicell.io/",
        workspace=workspace,
        token=token,
        streaming_engine=engine,
    )
    await client._connect_with_backoff()
    await client._register()
    return client, svc_id


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
async def test_health_via_hypha(mock_engine):
    """
    Register the ASGI service on Hypha; confirm GET /health responds
    over the real public HTTPS URL (no TestClient, real network).
    """
    token, workspace = _creds()
    client, svc_id = await _register_client(token, workspace, mock_engine)
    try:
        ws = client._server.config.workspace
        url = f"https://hypha.aicell.io/{ws}/apps/{svc_id}/health"
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
async def test_transcript_feed_via_hypha(mock_engine):
    """
    Full end-to-end SSE test:
      1. Register ASGI service on hypha.aicell.io.
      2. Connect to /transcript_feed over HTTPS.
      3. MockStreamingEngine.init_session() pre-loads a transcript into text_queue.
      4. Assert at least one SSE data event is received.
    """
    token, workspace = _creds()
    client, svc_id = await _register_client(token, workspace, mock_engine)
    try:
        ws = client._server.config.workspace
        url = f"https://hypha.aicell.io/{ws}/apps/{svc_id}/transcript_feed"
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
        assert any(line.startswith("data:") for line in received), \
            "Expected a data: SSE event"
    finally:
        await asyncio.sleep(0.6)
        await client._server.disconnect()
