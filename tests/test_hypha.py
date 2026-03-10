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
      2. Pre-seed the engine (mirrors main.py calling init_session() at startup).
      3. Connect to /transcript_feed over HTTPS.
      4. Assert at least one SSE data event is received.
    """
    token, workspace = _creds()
    client, svc_id = await _register_client(token, workspace, mock_engine)
    # Seed the queue before the SSE client connects (mirrors main.py init_session call).
    mock_engine.init_session()
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


@pytest.mark.integration
async def test_two_clients_both_receive_via_hypha(mock_engine):
    """
    End-to-end multi-client fan-out test over real Hypha:
      1. Register ASGI on hypha.aicell.io.
      2. Connect two SSE clients simultaneously.
      3. After both are subscribed, emit one transcript item.
      4. Assert BOTH clients receive the exact same data: event.

    This test would FAIL with the old single-queue design because the first
    client's get() would consume the item leaving nothing for the second.
    """
    token, workspace = _creds()
    client, svc_id = await _register_client(token, workspace, mock_engine)
    ws = client._server.config.workspace
    url = f"https://hypha.aicell.io/{ws}/apps/{svc_id}/transcript_feed"
    print(f"\n[multi-client] SSE URL: {url}")

    received_a = []
    received_b = []

    async def _collect_one(label, bucket):
        async with httpx.AsyncClient() as http:
            async with http.stream(
                "GET", url,
                timeout=httpx.Timeout(45.0, connect=15.0),
            ) as resp:
                assert resp.status_code == 200, f"[{label}] expected 200"
                print(f"[{label}] connected")
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        bucket.append(line)
                        print(f"[{label}] received: {line!r}")
                        break  # got one event — disconnect

    async def _feeder():
        """Wait for both SSE clients to connect and subscribe, then emit."""
        await asyncio.sleep(5.0)
        mock_engine.text_queue.put("Hello from both clients")
        print("[feeder] emitted: 'Hello from both clients'")

    try:
        await asyncio.gather(
            _collect_one("client-A", received_a),
            _collect_one("client-B", received_b),
            _feeder(),
        )
    finally:
        await asyncio.sleep(0.5)
        await client._server.disconnect()

    import json
    assert len(received_a) >= 1, "Client A received no data: events"
    assert len(received_b) >= 1, "Client B received no data: events"

    text_a = json.loads(received_a[0].removeprefix("data: "))["text"]
    text_b = json.loads(received_b[0].removeprefix("data: "))["text"]
    assert text_a == "Hello from both clients", f"Client A got: {text_a!r}"
    assert text_b == "Hello from both clients", f"Client B got: {text_b!r}"
    print(f"\n[multi-client] Both clients received the same segment ✓")


@pytest.mark.integration
async def test_remaining_client_unaffected_by_peer_disconnect_via_hypha(mock_engine):
    """
    End-to-end disconnect-safety test over real Hypha:
      1. Connect two SSE clients.
      2. Emit first item — both receive it.
      3. Client B disconnects (breaks after first event).
      4. Emit second item — Client A still receives it; Client B does not.

    This test would FAIL with the old design because finish_session() was
    called on Client B's disconnect, terminating transcription for Client A.
    """
    token, workspace = _creds()
    client, svc_id = await _register_client(token, workspace, mock_engine)
    ws = client._server.config.workspace
    url = f"https://hypha.aicell.io/{ws}/apps/{svc_id}/transcript_feed"
    print(f"\n[disconnect-safety] SSE URL: {url}")

    received_a = []
    received_b = []

    async def _client_a():
        """Stay connected; collect up to 2 events."""
        async with httpx.AsyncClient() as http:
            async with http.stream(
                "GET", url,
                timeout=httpx.Timeout(60.0, connect=15.0),
            ) as resp:
                assert resp.status_code == 200
                print("[A] connected")
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        received_a.append(line)
                        print(f"[A] received: {line!r}")
                        if len(received_a) >= 2:
                            break

    async def _client_b():
        """Connect, receive first event, then disconnect."""
        async with httpx.AsyncClient() as http:
            async with http.stream(
                "GET", url,
                timeout=httpx.Timeout(45.0, connect=15.0),
            ) as resp:
                assert resp.status_code == 200
                print("[B] connected")
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        received_b.append(line)
                        print(f"[B] received and disconnecting: {line!r}")
                        break  # simulate tab close

    async def _feeder():
        await asyncio.sleep(5.0)  # wait for both to connect
        mock_engine.text_queue.put("first-message")
        print("[feeder] emitted first-message")
        await asyncio.sleep(3.0)  # wait for B to break and disconnect
        mock_engine.text_queue.put("second-message")
        print("[feeder] emitted second-message (B should be gone)")

    try:
        await asyncio.gather(
            _client_a(),
            _client_b(),
            _feeder(),
        )
    finally:
        await asyncio.sleep(0.5)
        await client._server.disconnect()

    import json

    texts_a = [json.loads(l.removeprefix("data: "))["text"] for l in received_a]
    texts_b = [json.loads(l.removeprefix("data: "))["text"] for l in received_b]

    assert "first-message" in texts_a, f"Client A missed first-message: {texts_a}"
    assert "second-message" in texts_a, \
        f"Client A missed second-message after B disconnected: {texts_a}"
    assert texts_b == ["first-message"], \
        f"Client B should only have first-message, got: {texts_b}"
    print(f"\n[disconnect-safety] Client A: {texts_a}, Client B: {texts_b} ✓")
