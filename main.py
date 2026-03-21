"""
main.py — Hypha Whisper Node entry point.

Wires: MicCapture (raw PyAudio) → audio_loop → StreamingEngine (whisper_streaming)
       → text_queue → HyphaClient (ASGI SSE)

Usage:
    python main.py [--server URL] [--workspace WS] [--token TOKEN]
                   [--model MODEL] [--backend BACKEND]

Environment variables (.env file or shell, in priority order):
    HYPHA_SERVER           — Hypha server URL (default: https://hypha.aicell.io/)
    HYPHA_WORKSPACE        — workspace name (e.g. reef-imaging)
    HYPHA_WORKSPACE_TOKEN  — workspace token

Offline mode (no --server / HYPHA_SERVER): transcribes locally and prints to stdout.
"""

import argparse
import asyncio
import concurrent.futures
import logging
import os
import queue
import signal
import sys


def _load_dotenv(path=".env"):
    """Load KEY=VALUE pairs from a .env file into os.environ (if not already set)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                os.environ.setdefault(key.strip(), value.strip())
    except FileNotFoundError:
        pass


_load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
for _lib in ("hypha_rpc", "websockets", "asyncio", "urllib3", "httpx"):
    logging.getLogger(_lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Jetson Whisper → Hypha ASGI node")
    p.add_argument("--server",
                   default=os.environ.get("HYPHA_SERVER", "https://hypha.aicell.io/"),
                   help="Hypha server URL (default: https://hypha.aicell.io/)")
    p.add_argument("--workspace",
                   default=os.environ.get("HYPHA_WORKSPACE", ""),
                   help="Hypha workspace name (default: $HYPHA_WORKSPACE)")
    p.add_argument("--token",
                   default=os.environ.get("HYPHA_WORKSPACE_TOKEN", ""),
                   help="Workspace token (default: $HYPHA_WORKSPACE_TOKEN)")
    p.add_argument("--model", default="small.en",
                   help="Whisper model name or HuggingFace ID (default: small.en)")
    p.add_argument("--backend", default="whisper-plain",
                   choices=["whisper-plain", "whisper-timestamped", "distil-whisper"],
                   help="ASR backend (default: whisper-plain)")
    p.add_argument("--device", default="",
                   help="PyTorch device: cuda or cpu (default: auto)")
    p.add_argument("--mic", default="",
                   help="Preferred mic name substring (default: auto-detect ReSpeaker then HIK)")
    p.add_argument("--prompt", default=os.environ.get("WHISPER_PROMPT", ""),
                   help="Domain vocabulary prompt prepended to every Whisper transcription call "
                        "(e.g. 'Hypha, AICell Lab, bioimaging'). "
                        "Also reads $WHISPER_PROMPT env var.")
    return p.parse_args()


async def audio_loop(mic, engine, shutdown: asyncio.Event, listening: asyncio.Event):
    """Feed raw audio chunks from the mic into the streaming engine when listening is set.

    Runs as an independent asyncio Task. Only processes audio when the
    'listening' Event is set (i.e., when at least one client is connected).
    The engine's process_audio() is blocking (runs Whisper inference), so it
    executes in a dedicated single-worker thread pool to prevent default pool
    saturation during inference. Committed text is placed into engine.text_queue
    by process_audio().
    """
    loop = asyncio.get_event_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    logger.info("[audio_loop] Started (waiting for clients)")
    try:
        while not shutdown.is_set():
            # Wait until listening is enabled (client connected)
            if not listening.is_set():
                try:
                    await asyncio.wait_for(listening.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
            
            # Drain the audio queue to avoid backlog after pausing
            while not mic.raw_audio_queue.empty():
                try:
                    mic.raw_audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Process audio while listening is enabled
            while listening.is_set() and not shutdown.is_set():
                try:
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(executor, mic.raw_audio_queue.get, True, 0.1),
                        timeout=0.5,
                    )
                except (asyncio.TimeoutError, queue.Empty):
                    # Check if we're still listening
                    continue
                try:
                    await loop.run_in_executor(executor, engine.process_audio, chunk)
                except Exception as exc:
                    logger.warning("[audio_loop] process_audio error: %s", exc)
            
            if not shutdown.is_set():
                logger.info("[audio_loop] Paused (no clients)")
    finally:
        executor.shutdown(wait=False)


async def run_offline(engine, shutdown: asyncio.Event):
    """Offline mode: print committed transcripts to stdout."""
    loop = asyncio.get_event_loop()
    logger.info("[offline] Transcribing — press Ctrl+C to stop")
    # Prime the session immediately in offline mode.
    engine.init_session()
    while not shutdown.is_set():
        try:
            text = await asyncio.wait_for(
                loop.run_in_executor(None, engine.text_queue.get, True, 0.5),
                timeout=1.0,
            )
            print(f"[transcript] {text}", flush=True)
        except (asyncio.TimeoutError, queue.Empty):
            continue


async def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Audio capture
    # ------------------------------------------------------------------
    logger.info("[main] Initialising microphone capture...")
    from audio.capture import MicCapture
    try:
        mic = MicCapture(preferred_mic=args.mic or None)
    except RuntimeError as e:
        logger.error("[main] %s", e)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Streaming engine (loads Whisper model)
    # ------------------------------------------------------------------
    logger.info("[main] Loading Whisper model '%s' (backend: %s)...", args.model, args.backend)
    from transcribe.streaming_engine import StreamingEngine
    engine = StreamingEngine(
        model_name=args.model,
        backend=args.backend,
        prompt_prefix=args.prompt,
    )

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    loop = asyncio.get_event_loop()
    shutdown = asyncio.Event()
    listening = asyncio.Event()  # Set when at least one client is connected

    def _sigint_handler():
        logger.info("[main] Shutdown signal received")
        shutdown.set()

    loop.add_signal_handler(signal.SIGINT, _sigint_handler)
    loop.add_signal_handler(signal.SIGTERM, _sigint_handler)

    # ------------------------------------------------------------------
    # Client connection callbacks for on-demand listening
    # ------------------------------------------------------------------
    def on_first_client():
        """Called when first client connects - start listening."""
        logger.info("[main] First client connected - starting microphone")
        mic.start()
        listening.set()

    def on_last_client():
        """Called when last client disconnects - stop listening."""
        logger.info("[main] Last client disconnected - stopping microphone")
        listening.clear()
        mic.stop()
        # Drain the text queue to avoid stale transcripts on reconnect
        while not engine.text_queue.empty():
            try:
                engine.text_queue.get_nowait()
            except queue.Empty:
                break

    # ------------------------------------------------------------------
    # Start audio processing loop (mic will be started on first client)
    # ------------------------------------------------------------------
    audio_task = loop.create_task(audio_loop(mic, engine, shutdown, listening))

    # ------------------------------------------------------------------
    # Hypha RPC or offline mode
    # ------------------------------------------------------------------
    if args.server:
        # Prime the session once here so the engine is ready before any client
        # connects.  Session lifecycle is no longer tied to individual SSE
        # client connect/disconnect events.
        engine.init_session()
        from rpc.hypha_client import HyphaClient
        client = HyphaClient(
            server_url=args.server,
            workspace=args.workspace,
            token=args.token,
            streaming_engine=engine,
            on_first_client=on_first_client,
            on_last_client=on_last_client,
        )
        logger.info("[main] Connecting to Hypha at %s (workspace: %s)...",
                    args.server, args.workspace or "<default>")
        rpc_task = loop.create_task(client.run())
    else:
        logger.info("[main] No server configured — running in offline mode")
        # In offline mode, start mic immediately
        mic.start()
        listening.set()
        rpc_task = loop.create_task(run_offline(engine, shutdown))

    # Wait until shutdown signal
    await shutdown.wait()

    # ------------------------------------------------------------------
    # Clean shutdown
    # ------------------------------------------------------------------
    logger.info("[main] Stopping...")
    rpc_task.cancel()
    audio_task.cancel()
    await asyncio.gather(rpc_task, audio_task, return_exceptions=True)

    # Flush any remaining audio context from the streaming engine.
    final = await loop.run_in_executor(None, engine.finish_session)
    if final:
        logger.info("[main] Final transcript: %s", final)

    # Only stop mic if it was started
    if listening.is_set():
        mic.stop()
    logger.info("[main] Done")


if __name__ == "__main__":
    asyncio.run(main())
