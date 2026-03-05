"""
rpc/hypha_client.py — Hypha RPC service for the Jetson Whisper node.

Connects to the Hypha server at https://hypha.aicell.io/ and registers
a protected service that streams live transcripts to remote clients.

Environment variables (set in /etc/hypha-whisper/config.env):
    HYPHA_SERVER  — e.g. https://hypha.aicell.io/
    HYPHA_TOKEN   — workspace token for protected visibility
"""

import asyncio
import logging
import time
from typing import Optional

from hypha_rpc import connect_to_server

logger = logging.getLogger(__name__)

SERVICE_ID = "hypha-whisper"
SERVICE_NAME = "Jetson Whisper Node"
_RECONNECT_MAX_WAIT = 60  # seconds


class HyphaClient:
    """
    Registers a Hypha service that exposes:
      - stream_transcripts() : async generator yielding {"text", "timestamp"}
      - health()             : returns status dict

    Usage:
        client = HyphaClient(
            server_url=os.environ["HYPHA_SERVER"],
            token=os.environ["HYPHA_TOKEN"],
            mic_capture=mic,
            whisper_engine=engine,
        )
        await client.run()   # blocks; reconnects on disconnect
    """

    def __init__(self, server_url: str, token: str, mic_capture, whisper_engine):
        self.server_url = server_url.rstrip("/")
        self.token = token
        self._mic = mic_capture
        self._whisper = whisper_engine
        self._server = None
        self._start_time = time.time()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self):
        """Connect, register, keep alive. Reconnects on startup failure."""
        await self._connect_with_backoff()
        await self._register()
        logger.info("[hypha] Service '%s' registered. Streaming transcripts.", SERVICE_ID)
        # hypha-rpc 0.20+ reconnects mid-session automatically;
        # this coroutine just needs to stay alive.
        await asyncio.sleep(float("inf"))

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def _connect_with_backoff(self):
        wait = 1
        while True:
            try:
                self._server = await connect_to_server({
                    "server_url": self.server_url,
                    "token": self.token,
                })
                logger.info("[hypha] Connected to %s", self.server_url)
                return
            except Exception as exc:
                logger.warning("[hypha] Connection failed (%s). Retrying in %ss…", exc, wait)
                await asyncio.sleep(wait)
                wait = min(wait * 2, _RECONNECT_MAX_WAIT)

    # ------------------------------------------------------------------
    # Service registration
    # ------------------------------------------------------------------

    async def _register(self):
        await self._server.register_service({
            "id": SERVICE_ID,
            "name": SERVICE_NAME,
            "config": {"visibility": "protected"},
            "stream_transcripts": self._stream_transcripts,
            "health": self._health,
        })

    # ------------------------------------------------------------------
    # Service methods
    # ------------------------------------------------------------------

    async def _stream_transcripts(self):
        """
        Async generator: pulls PCM chunks from MicCapture's threading.Queue,
        transcribes via WhisperEngine, yields transcript dicts.
        """
        loop = asyncio.get_event_loop()
        while True:
            try:
                # Bridge threading.Queue → async without blocking the event loop.
                # Timeout=0.1s so we don't stall forever if the queue is empty.
                pcm_bytes = await loop.run_in_executor(
                    None, self._mic.queue.get, True, 0.1
                )
            except Exception:
                # queue.get timed out or queue was empty — keep waiting
                await asyncio.sleep(0.05)
                continue

            text = self._whisper.transcribe(pcm_bytes)
            if text:
                yield {"text": text, "timestamp": time.time()}

    def _health(self) -> dict:
        return {
            "status": "ok",
            "model": getattr(self._whisper, "model_name", "unknown"),
            "uptime_seconds": round(time.time() - self._start_time),
        }
