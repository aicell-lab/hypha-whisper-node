"""
rpc/hypha_client.py — Hypha ASGI service for the Jetson Whisper node.

Registers a FastAPI ASGI service on Hypha that exposes:
  GET /transcript_feed  — SSE stream: streaming Whisper (LocalAgreement)
                          pushes committed transcript segments as SSE events.
                          Multiple simultaneous clients are supported — each
                          receives its own copy of every transcript item via
                          per-client asyncio queues fanned out from a single
                          background broadcast loop.
                          Session lifecycle (init/finish) is managed by
                          main.py at startup/shutdown, not per-client.
  GET /health           — JSON status dict
  GET /logs             — SSE stream of Python logging records; ?tail=N replays last N lines

Environment variables (set in /etc/hypha-whisper/config.env):
    HYPHA_SERVER  — e.g. https://hypha.aicell.io/
    HYPHA_TOKEN   — workspace token for public visibility

Usage:
    client = HyphaClient(server_url=..., token=...,
                         streaming_engine=engine)
    await client.run()   # blocks; reconnects on disconnect
"""

import asyncio
import collections
import json
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
_engine = None
_text_queue = None
_start_time: float = 0.0

# Multi-client fan-out state.
# Each connected SSE client gets its own asyncio.Queue in _subscribers.
# _broadcast_loop() drains _text_queue and copies each item to all subscribers.
_subscribers: set = set()
_broadcast_task = None
_CLEAR_SENTINEL = {"_clear": True}

# Client connection state callbacks (set by HyphaClient)
_on_first_client: callable = None
_on_last_client: callable = None

# ---------------------------------------------------------------------------
# Log streaming — captures all Python logging records and fans out to SSE
# ---------------------------------------------------------------------------

_log_queue: queue.Queue = queue.Queue(maxsize=2000)
_log_subscribers: set = set()
_log_broadcast_task = None
_log_buffer: collections.deque = collections.deque(maxlen=2000)  # rolling history for ?tail=N


class _LogQueueHandler(logging.Handler):
    """Logging handler that puts formatted records into _log_queue and _log_buffer."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            entry = {
                "ts": record.created,
                "level": record.levelname,
                "logger": record.name,
                "msg": msg,
            }
            _log_buffer.append(entry)
            _log_queue.put_nowait(entry)
        except queue.Full:
            pass  # drop silently if no consumer is draining fast enough
        except Exception:
            self.handleError(record)


_log_handler: _LogQueueHandler | None = None


def _install_log_handler() -> None:
    """Attach _LogQueueHandler to the root logger (idempotent)."""
    global _log_handler
    if _log_handler is not None:
        return
    _log_handler = _LogQueueHandler()
    _log_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(_log_handler)


def _push_to_subscribers(item) -> None:
    """Put item into all current subscriber queues (thread-safe via put_nowait)."""
    for q in list(_subscribers):
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

app = FastAPI()


def _item_to_json(item) -> str:
    """Convert a text_queue item to a JSON string for SSE payload.

    Accepts either:
      - dict: {"text": str, "speaker": str, "angle": int|None}
      - str:  legacy plain text (wrapped into {"text": ..., "speaker": "", "angle": null})
    """
    if isinstance(item, dict):
        return json.dumps({"text": item.get("text", ""),
                           "speaker": item.get("speaker", ""),
                           "angle": item.get("angle")})
    # Legacy str
    return json.dumps({"text": str(item), "speaker": "", "angle": None})


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
      word-break: break-word;
    }
    #transcript-box:empty::before {
      content: "Waiting for speech…";
      color: #aaa;
      font-style: italic;
    }
    .segment { margin-bottom: 6px; }
    .badge {
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 1px 7px;
      border-radius: 99px;
      margin-right: 6px;
      vertical-align: middle;
      opacity: 0.9;
    }
    .seg-text { vertical-align: middle; }
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
    <button onclick="clearSession()">Clear</button>
  </footer>
  <script>
    const box    = document.getElementById('transcript-box');
    const status = document.getElementById('status');

    // Speaker colour palette (cycle by speaker index)
    const PALETTE = [
      ['#1a73e8','#e8f0fe'], ['#e67c00','#fef3e2'], ['#188038','#e6f4ea'],
      ['#a142f4','#f3e8fd'], ['#c5221f','#fce8e6'], ['#007b83','#e4f7fb'],
    ];
    const speakerColors = {};
    let speakerCount = 0;

    function speakerColor(label) {
      if (!(label in speakerColors)) {
        speakerColors[label] = PALETTE[speakerCount % PALETTE.length];
        speakerCount++;
      }
      return speakerColors[label];
    }

    let lastSpeaker = null;
    let lastTextSpan = null;

    function appendSegment(data) {
      let text, speaker, angle;
      try {
        const obj = JSON.parse(data);
        text    = obj.text    || '';
        speaker = obj.speaker || '';
        angle   = obj.angle   != null ? obj.angle + '°' : null;
      } catch (_) {
        // legacy plain-text fallback
        text    = data;
        speaker = '';
        angle   = null;
      }
      if (!text.trim()) return;

      // If same speaker as last segment, just append text inline
      if (speaker && speaker === lastSpeaker && lastTextSpan) {
        lastTextSpan.textContent += ' ' + text.trim();
        box.scrollTop = box.scrollHeight;
        return;
      }

      // New speaker (or no speaker) — start a new segment line
      const seg = document.createElement('div');
      seg.className = 'segment';

      if (speaker) {
        const [fg, bg] = speakerColor(speaker);
        const badge = document.createElement('span');
        badge.className = 'badge';
        badge.style.color = fg;
        badge.style.background = bg;
        badge.textContent = speaker;
        seg.appendChild(badge);
      }

      const span = document.createElement('span');
      span.className = 'seg-text';
      span.textContent = text;
      seg.appendChild(span);

      box.appendChild(seg);
      box.scrollTop = box.scrollHeight;

      lastSpeaker = speaker || null;
      lastTextSpan = span;
    }

    function clearDisplay() {
      box.textContent = '';
      lastSpeaker = null;
      lastTextSpan = null;
      for (const k of Object.keys(speakerColors)) delete speakerColors[k];
      speakerCount = 0;
    }

    function clearSession() {
      fetch('clear', {method: 'POST'});
    }

    function connect() {
      const src = new EventSource('transcript_feed');

      src.onopen = () => {
        status.textContent = '● Connected';
        status.className = 'connected';
      };

      src.onmessage = (e) => {
        appendSegment(e.data);
      };

      src.addEventListener('clear', () => {
        clearDisplay();
      });

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


async def _broadcast_loop():
    """Background task: drain the engine's text_queue and fan out to all SSE clients."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            item = await asyncio.wait_for(
                loop.run_in_executor(None, _text_queue.get, True, 0.5),
                timeout=15.0,
            )
            payload = _item_to_json(item)
            # Note: Transcript payload is NOT logged for privacy
            # It is only sent to connected SSE clients
            logger.info("[transcript] Transcript sent to %d client(s)", len(_subscribers))
            for q in list(_subscribers):
                try:
                    q.put_nowait(item)
                except asyncio.QueueFull:
                    logger.warning("[broadcast] Client queue full — dropping item")
        except (asyncio.TimeoutError, queue.Empty):
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("[broadcast_loop] Unexpected error: %s", exc, exc_info=True)


def _ensure_broadcast_loop():
    """Start _broadcast_loop as a background task if not already running."""
    global _broadcast_task
    if _broadcast_task is None or _broadcast_task.done():
        _broadcast_task = asyncio.get_event_loop().create_task(_broadcast_loop())


@app.get("/transcript_feed")
async def transcript_feed():
    """
    SSE endpoint: streams committed transcript segments from the streaming engine.

    Each segment arrives as:
        data: <json>\\n\\n

    A keep-alive comment is sent every ~15 s when there is no committed text,
    so proxies and browsers do not close an idle connection.

    Multiple simultaneous clients are supported. Each client gets its own
    asyncio.Queue; the background _broadcast_loop fans out every transcript
    item to all connected clients. Session lifecycle (init/finish) is managed
    by main.py and is not affected by client connect/disconnect.
    """
    async def sse_gen():
        client_q: asyncio.Queue = asyncio.Queue(maxsize=256)
        _subscribers.add(client_q)
        is_first = len(_subscribers) == 1
        _ensure_broadcast_loop()
        logger.info("[transcript_feed] Client connected (%d total)", len(_subscribers))
        
        # Notify on first client connection
        if is_first and _on_first_client is not None:
            try:
                _on_first_client()
            except Exception as exc:
                logger.error("[transcript_feed] _on_first_client callback error: %s", exc)
        
        try:
            while True:
                try:
                    item = await asyncio.wait_for(client_q.get(), timeout=15.0)
                    if isinstance(item, dict) and item.get("_clear"):
                        yield "event: clear\ndata: {}\n\n"
                    else:
                        yield f"data: {_item_to_json(item)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        except Exception as exc:
            logger.error("[transcript_feed] SSE generator error: %s", exc, exc_info=True)
        finally:
            _subscribers.discard(client_q)
            remaining = len(_subscribers)
            logger.info("[transcript_feed] Client disconnected (%d remaining)", remaining)
            
            # Notify on last client disconnection
            if remaining == 0:
                if _on_last_client is not None:
                    try:
                        _on_last_client()
                    except Exception as exc:
                        logger.error("[transcript_feed] _on_last_client callback error: %s", exc)
                if _engine is not None:
                    _engine.init_session()
                    logger.info("[transcript_feed] Last client disconnected — engine session reset")

    return StreamingResponse(sse_gen(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": getattr(_engine, "model_name", "unknown"),
        "uptime_seconds": round(time.time() - _start_time),
    }


@app.post("/clear")
async def clear_session():
    """Reset the engine session state and notify all SSE clients to clear their display."""
    if _engine is not None:
        _engine.init_session()
        logger.info("[clear] Engine session reset via /clear")
    _push_to_subscribers(_CLEAR_SENTINEL)
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Log feed
# ---------------------------------------------------------------------------

async def _log_broadcast_loop():
    """Background task: drain _log_queue and fan out to all log SSE clients."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            record = await asyncio.wait_for(
                loop.run_in_executor(None, _log_queue.get, True, 0.5),
                timeout=15.0,
            )
            for q in list(_log_subscribers):
                try:
                    q.put_nowait(record)
                except asyncio.QueueFull:
                    pass
        except (asyncio.TimeoutError, queue.Empty):
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Use print to avoid infinite recursion via logging
            print(f"[log_broadcast_loop] error: {exc}")


def _ensure_log_broadcast_loop():
    global _log_broadcast_task
    if _log_broadcast_task is None or _log_broadcast_task.done():
        _log_broadcast_task = asyncio.get_event_loop().create_task(_log_broadcast_loop())


@app.get("/logs")
async def logs(tail: int = 0):
    """SSE stream of all Python logging records from this process.

    Query parameters:
        tail  — emit the last N buffered records before streaming live ones
                (0 = only new records, max capped at 2000)

    Each SSE event carries a newline-delimited JSON object:
        {"ts": <unix epoch float>, "level": "INFO"|"WARNING"|...,
         "logger": "<logger name>", "msg": "<formatted log line>"}

    A keep-alive comment (": keep-alive") is sent every 15 s when idle so
    the connection is not dropped by proxies or HTTP clients.
    """
    tail = max(0, min(tail, len(_log_buffer)))

    async def sse_gen():
        # Replay buffered records first
        for entry in list(_log_buffer)[-tail:]:
            yield f"data: {json.dumps(entry)}\n\n"

        # Then stream live records
        client_q: asyncio.Queue = asyncio.Queue(maxsize=512)
        _log_subscribers.add(client_q)
        _ensure_log_broadcast_loop()
        try:
            while True:
                try:
                    record = await asyncio.wait_for(client_q.get(), timeout=15.0)
                    yield f"data: {json.dumps(record)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            _log_subscribers.discard(client_q)

    return StreamingResponse(sse_gen(), media_type="text/event-stream")


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
            streaming_engine=engine,
        )
        await client.run()   # blocks; reconnects on disconnect
    """

    def __init__(self, server_url: str, token: str, streaming_engine,
                 workspace: str = "",
                 on_first_client: callable = None,
                 on_last_client: callable = None):
        global _engine, _text_queue, _start_time, _subscribers, _broadcast_task
        global _log_subscribers, _log_broadcast_task, _on_first_client, _on_last_client
        self.server_url = server_url.rstrip("/")
        self.workspace = workspace
        self.token = token
        _engine = streaming_engine
        _text_queue = streaming_engine.text_queue
        _start_time = time.time()
        self._server = None
        # Reset fan-out state so each HyphaClient instance starts clean
        # (important for test isolation and reconnects).
        _subscribers = set()
        if _broadcast_task is not None and not _broadcast_task.done():
            _broadcast_task.cancel()
        _broadcast_task = None
        # Install log handler so /log_feed captures all logging output.
        _install_log_handler()
        _log_subscribers = set()
        if _log_broadcast_task is not None and not _log_broadcast_task.done():
            _log_broadcast_task.cancel()
        _log_broadcast_task = None
        # Store callbacks for client connection state changes
        _on_first_client = on_first_client
        _on_last_client = on_last_client

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
                "[hypha] ASGI service '%s' registered. "
                "transcript: %s/%s/apps/%s/transcript_feed  "
                "logs: %s/%s/apps/%s/logs?tail=50",
                SERVICE_ID,
                self.server_url, self._server.config.workspace, SERVICE_ID,
                self.server_url, self._server.config.workspace, SERVICE_ID,
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
            except KeyError as exc:
                # "Service not found" means the server IS reachable but the service
                # query returned nothing (transient state after reconnect or service
                # refresh).  The connection is alive — do not reconnect.
                logger.debug("[hypha] Keepalive: service lookup returned KeyError (%s) — ignoring", exc)
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
