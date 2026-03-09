"""
rpc/hypha_client.py — Hypha ASGI service for the Jetson Whisper node.

Registers a FastAPI ASGI service on Hypha that exposes:
  GET /transcript_feed  — SSE stream: Whisper transcribes live mic audio and
                          pushes each segment as a Server-Sent Event.
                          Text grows longer and longer while speaking.
                          On disconnect: mic queue is drained so the next
                          connection starts with a clean slate.
  GET /health           — JSON status dict

Environment variables (set in /etc/hypha-whisper/config.env):
    HYPHA_SERVER  — e.g. https://hypha.aicell.io/
    HYPHA_TOKEN   — workspace token for public visibility

Usage:
    client = HyphaClient(server_url=..., token=...,
                         mic_capture=mic, whisper_engine=engine)
    await client.run()   # blocks; reconnects on disconnect
"""

import asyncio
import logging
import os
import queue
import socket
import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from hypha_rpc import connect_to_server

logger = logging.getLogger(__name__)

_WATCHDOG_INTERVAL = 10  # seconds between sd_notify WATCHDOG=1 calls

SERVICE_ID = "hypha-whisper"
SERVICE_NAME = "Jetson Whisper Node"
_RECONNECT_MAX_WAIT = 60  # seconds

# Module-level references injected by HyphaClient.__init__ so the FastAPI
# route handlers (which are module-level functions) can reach them.
_mic = None
_whisper = None
_start_time: float = 0.0

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
async def live_transcript_page():
    """Human-readable live transcript viewer."""
    return HTMLResponse("""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Hypha Whisper — Live Transcript</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, sans-serif;
      background: #f5f5f5;
      display: flex;
      flex-direction: column;
      height: 100vh;
      padding: 20px;
      gap: 12px;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    h1 { font-size: 1.1rem; font-weight: 600; color: #222; }
    #status {
      font-size: 0.8rem;
      padding: 3px 10px;
      border-radius: 99px;
      background: #e0e0e0;
      color: #555;
    }
    #status.connected { background: #d4edda; color: #155724; }
    #status.error     { background: #f8d7da; color: #721c24; }
    #transcript-box {
      flex: 1;
      background: #fff;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 20px;
      overflow-y: auto;
      font-size: 1.05rem;
      line-height: 1.75;
      color: #222;
      white-space: pre-wrap;
      word-break: break-word;
    }
    #transcript-box:empty::before {
      content: "Waiting for speech…";
      color: #aaa;
      font-style: italic;
    }
    footer {
      display: flex;
      justify-content: flex-end;
      flex-shrink: 0;
    }
    button {
      padding: 7px 18px;
      border: 1px solid #ccc;
      border-radius: 6px;
      background: #fff;
      cursor: pointer;
      font-size: 0.9rem;
      color: #444;
    }
    button:hover { background: #f0f0f0; }
  </style>
</head>
<body>
  <header>
    <h1>Hypha Whisper &mdash; Live Transcript</h1>
    <span id="status">Connecting…</span>
  </header>
  <div id="transcript-box"></div>
  <footer>
    <button onclick="document.getElementById('transcript-box').textContent=''">Clear</button>
  </footer>
  <script>
    const box    = document.getElementById('transcript-box');
    const status = document.getElementById('status');

    function connect() {
      const src = new EventSource('transcript_feed');

      src.onopen = () => {
        status.textContent = '● Connected';
        status.className = 'connected';
      };

      src.onmessage = (e) => {
        const text = e.data.trim();
        if (!text) return;
        box.textContent += (box.textContent ? ' ' : '') + text;
        box.scrollTop = box.scrollHeight;
      };

      src.onerror = () => {
        status.textContent = 'Disconnected — retrying…';
        status.className = 'error';
        src.close();
        setTimeout(connect, 3000);
      };
    }

    connect();
  </script>
</body>
</html>""")


@app.get("/transcript_feed")
async def transcript_feed():
    """
    SSE endpoint: streams live Whisper transcript segments.

    Each segment arrives as:
        data: <text>\\n\\n

    A keep-alive comment is sent every ~15 s when the mic is silent so that
    proxies and browsers do not close an idle connection.

    On client disconnect the mic queue is drained to ensure the next
    connection receives only fresh audio.
    """
    async def sse_gen():
        loop = asyncio.get_event_loop()
        try:
            while True:
                # Block up to 15 s waiting for a voiced audio chunk.
                try:
                    pcm = await asyncio.wait_for(
                        loop.run_in_executor(None, _mic.queue.get, True, 0.5),
                        timeout=15.0,
                    )
                except (asyncio.TimeoutError, queue.Empty):
                    yield ": keep-alive\n\n"
                    continue

                try:
                    text = await loop.run_in_executor(None, _whisper.transcribe, pcm)
                except Exception as exc:
                    logger.error("[transcript_feed] Transcription error: %s", exc, exc_info=True)
                    continue
                if text:
                    logger.info("[transcript] %s", text)
                    yield f"data: {text}\n\n"
        except Exception as exc:
            logger.error("[transcript_feed] SSE generator error: %s", exc, exc_info=True)
        finally:
            # Drain stale audio so the next connection starts clean.
            drained = 0
            while not _mic.queue.empty():
                _mic.queue.get_nowait()
                drained += 1
            if drained:
                logger.info("[transcript_feed] Drained %d stale chunk(s)", drained)

    return StreamingResponse(sse_gen(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": getattr(_whisper, "model_name", "unknown"),
        "uptime_seconds": round(time.time() - _start_time),
    }


def _sd_notify(msg: str) -> None:
    """Send a message to the systemd notification socket (no-op if not running under systemd)."""
    sock_path = os.environ.get("NOTIFY_SOCKET", "")
    if not sock_path:
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as s:
            s.connect(sock_path.lstrip("@"))
            s.sendall(msg.encode())
    except OSError:
        pass


class HyphaClient:
    """
    Registers a Hypha ASGI service exposing /transcript_feed and /health.

    Usage:
        client = HyphaClient(
            server_url=os.environ["HYPHA_SERVER"],
            token=os.environ["HYPHA_TOKEN"],
            mic_capture=mic,
            whisper_engine=engine,
        )
        await client.run()   # blocks; reconnects on disconnect
    """

    def __init__(self, server_url: str, token: str, mic_capture, whisper_engine,
                 workspace: str = ""):
        global _mic, _whisper, _start_time
        self.server_url = server_url.rstrip("/")
        self.workspace = workspace
        self.token = token
        _mic = mic_capture
        _whisper = whisper_engine
        _start_time = time.time()
        self._server = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self):
        """Connect to Hypha, register ASGI service; reconnect on disconnect."""
        first = True
        while True:
            await self._connect_with_backoff()
            await self._register()
            logger.info(
                "[hypha] ASGI service '%s' registered. Try: %s/%s/apps/%s/transcript_feed",
                SERVICE_ID, self.server_url,
                self._server.config.workspace, SERVICE_ID,
            )
            if first:
                _sd_notify("READY=1")
                first = False
            watchdog_task = asyncio.create_task(self._watchdog_loop())
            try:
                await self._keepalive()
            except asyncio.CancelledError:
                watchdog_task.cancel()
                raise
            finally:
                watchdog_task.cancel()
            logger.warning("[hypha] Connection lost — reconnecting in 5 s…")
            await asyncio.sleep(5)

    # ------------------------------------------------------------------
    # Keepalive / disconnect detection
    # ------------------------------------------------------------------

    async def _keepalive(self):
        """Block until the Hypha connection drops, then return."""
        while True:
            await asyncio.sleep(15)
            try:
                await asyncio.wait_for(self._server.list_services(), timeout=10.0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[hypha] Keepalive ping failed (%s) — reconnecting", exc)
                return

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    async def _watchdog_loop(self):
        """Notify systemd watchdog every _WATCHDOG_INTERVAL seconds."""
        while True:
            _sd_notify("WATCHDOG=1")
            await asyncio.sleep(_WATCHDOG_INTERVAL)

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _connect_with_backoff(self):
        wait = 1
        conn_config = {"server_url": self.server_url, "token": self.token}
        if self.workspace:
            conn_config["workspace"] = self.workspace
        while True:
            try:
                self._server = await connect_to_server(conn_config)
                logger.info("[hypha] Connected to %s (workspace: %s)",
                            self.server_url,
                            self._server.config.workspace)
                return
            except Exception as exc:
                logger.warning("[hypha] Connection failed (%s). Retrying in %ss…", exc, wait)
                await asyncio.sleep(wait)
                wait = min(wait * 2, _RECONNECT_MAX_WAIT)

    # ------------------------------------------------------------------
    # Service registration
    # ------------------------------------------------------------------

    async def _register(self):
        async def serve_asgi(args, context=None):
            scope = args["scope"]
            logger.info(
                "[asgi] %s %s %s",
                context["user"]["id"] if context else "-",
                scope.get("method", ""),
                scope.get("path", ""),
            )
            await app(args["scope"], args["receive"], args["send"])

        svc_info = await self._server.register_service({
            "id": SERVICE_ID,
            "name": SERVICE_NAME,
            "type": "asgi",
            "serve": serve_asgi,
            "config": {"visibility": "public", "require_context": True},
        })
        logger.info("[hypha] Service info: %s", dict(svc_info))
