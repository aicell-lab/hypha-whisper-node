"""
main.py — Hypha Whisper Node entry point.

Wires: MicCapture (VAD) → WhisperEngine (GPU) → HyphaClient (ASGI stream)

Usage:
    python main.py [--server URL] [--workspace WS] [--token TOKEN] [--model MODEL]

Environment variables (.env file or shell, in priority order):
    HYPHA_SERVER           — Hypha server URL (default: https://hypha.aicell.io/)
    HYPHA_WORKSPACE        — workspace name (e.g. reef-imaging)
    HYPHA_WORKSPACE_TOKEN  — workspace token

Offline mode (no --server / HYPHA_SERVER): transcribes locally and prints to stdout.
"""

import argparse
import asyncio
import logging
import os
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
    p.add_argument("--model", default="base.en",
                   help="Whisper model name (default: base.en)")
    p.add_argument("--device", default="",
                   help="PyTorch device: cuda or cpu (default: auto)")
    return p.parse_args()


async def run_offline(mic, whisper_engine):
    """Offline mode: transcribe locally and print to stdout."""
    logger.info("[offline] Transcribing — press Ctrl+C to stop")
    loop = asyncio.get_event_loop()
    while True:
        try:
            pcm = await asyncio.wait_for(
                loop.run_in_executor(None, mic.queue.get, True, 0.5),
                timeout=1.0,
            )
        except asyncio.TimeoutError:
            continue
        text = await loop.run_in_executor(None, whisper_engine.transcribe, pcm)
        if text:
            print(f"[transcript] {text}", flush=True)


async def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Phase 2: audio capture
    # ------------------------------------------------------------------
    logger.info("[main] Initialising microphone capture...")
    from audio.capture import MicCapture
    try:
        mic = MicCapture()
    except RuntimeError as e:
        logger.error("[main] %s", e)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 3: Whisper engine
    # ------------------------------------------------------------------
    logger.info("[main] Loading Whisper model '%s'...", args.model)
    from transcribe.whisper_engine import WhisperEngine
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    whisper_engine = WhisperEngine(model_name=args.model)

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------
    loop = asyncio.get_event_loop()
    shutdown = asyncio.Event()

    def _sigint_handler():
        logger.info("[main] Shutdown signal received")
        shutdown.set()

    loop.add_signal_handler(signal.SIGINT, _sigint_handler)
    loop.add_signal_handler(signal.SIGTERM, _sigint_handler)

    # ------------------------------------------------------------------
    # Start mic capture
    # ------------------------------------------------------------------
    logger.info("[main] Starting mic capture (ambient noise calibration)...")
    mic.start()
    logger.info("[main] Mic capture running")

    # ------------------------------------------------------------------
    # Phase 4: Hypha RPC or offline mode
    # ------------------------------------------------------------------
    if args.server:
        from rpc.hypha_client import HyphaClient
        client = HyphaClient(
            server_url=args.server,
            workspace=args.workspace,
            token=args.token,
            mic_capture=mic,
            whisper_engine=whisper_engine,
        )
        logger.info("[main] Connecting to Hypha at %s (workspace: %s)…",
                    args.server, args.workspace or "<default>")
        rpc_task = loop.create_task(client.run())
    else:
        logger.info("[main] No server configured — running in offline mode")
        rpc_task = loop.create_task(run_offline(mic, whisper_engine))

    # Wait until shutdown signal
    await shutdown.wait()

    # ------------------------------------------------------------------
    # Clean shutdown
    # ------------------------------------------------------------------
    logger.info("[main] Stopping...")
    rpc_task.cancel()
    mic.stop()
    try:
        await rpc_task
    except asyncio.CancelledError:
        pass
    logger.info("[main] Done")


if __name__ == "__main__":
    asyncio.run(main())
