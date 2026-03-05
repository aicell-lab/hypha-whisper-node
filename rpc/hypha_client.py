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
import queue
import time

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from hypha_rpc import connect_to_server

logger = logging.getLogger(__name__)

SERVICE_ID = "hypha-whisper"
SERVICE_NAME = "Jetson Whisper Node"
_RECONNECT_MAX_WAIT = 60  # seconds

# Module-level references injected by HyphaClient.__init__ so the FastAPI
# route handlers (which are module-level functions) can reach them.
_mic = None
_whisper = None
_start_time: float = 0.0

app = FastAPI()


@app.get("/transcript_feed")
async def transcript_feed(request: Request):
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
                if await request.is_disconnected():
                    logger.info("[transcript_feed] Client disconnected")
                    break
                # Block up to 15 s waiting for a voiced audio chunk.
                try:
                    pcm = await asyncio.wait_for(
                        loop.run_in_executor(None, _mic.queue.get, True, 0.5),
                        timeout=15.0,
                    )
                except (asyncio.TimeoutError, queue.Empty):
                    yield ": keep-alive\n\n"
                    continue

                text = await loop.run_in_executor(None, _whisper.transcribe, pcm)
                if text:
                    logger.debug("[transcript_feed] %s", text)
                    yield f"data: {text}\n\n"
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
        """Connect to Hypha, register ASGI service, keep alive."""
        await self._connect_with_backoff()
        await self._register()
        logger.info(
            "[hypha] ASGI service '%s' registered. Try: %s/%s/apps/%s/transcript_feed",
            SERVICE_ID, self.server_url,
            self._server.config.workspace, SERVICE_ID,
        )
        await asyncio.sleep(float("inf"))

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
