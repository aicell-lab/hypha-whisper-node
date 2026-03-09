"""
tests/test_stress.py — 30-minute stress test for the Jetson Whisper node.

Requires physical hardware (Jetson GPU + HIKVISION mic).
Run with:
    pytest -m "hardware and slow" tests/test_stress.py -s -v

Monitors:
  - CPU usage via psutil
  - RAM usage via psutil
  - GPU / memory via tegrastats (Jetson-specific)
  - Transcript latency per chunk
"""

import asyncio
import subprocess
import threading
import time

import psutil
import pytest

STRESS_DURATION_S = 30 * 60  # 30 minutes
SAMPLE_INTERVAL_S = 5         # resource poll interval
MAX_CPU_PERCENT = 90.0        # alert threshold (single-core %)
MAX_RAM_PERCENT = 85.0        # alert threshold


# ---------------------------------------------------------------------------
# tegrastats helpers
# ---------------------------------------------------------------------------

def _start_tegrastats(interval_ms: int = 2000):
    """
    Start tegrastats in a background thread; returns (thread, samples_list).
    Each entry in samples_list is a raw line from tegrastats stdout.
    """
    samples = []
    stop_event = threading.Event()

    def _reader():
        try:
            proc = subprocess.Popen(
                ["tegrastats", f"--interval", str(interval_ms)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            while not stop_event.is_set():
                line = proc.stdout.readline()
                if line:
                    samples.append(line.strip())
            proc.terminate()
        except FileNotFoundError:
            pass  # not on Jetson — skip GPU monitoring

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    return stop_event, t, samples


def _parse_gpu_percent(tegrastats_line: str) -> float | None:
    """
    Extract GPU utilisation % from a tegrastats line.
    Format contains: GR3D_FREQ XX%
    """
    try:
        idx = tegrastats_line.index("GR3D_FREQ")
        part = tegrastats_line[idx:].split()[1]
        return float(part.rstrip("%@"))
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# CI sustained test — no hardware, real Whisper + real Hypha
# ---------------------------------------------------------------------------

@pytest.mark.integration
async def test_sustained_pipeline_ci(tone_pcm):
    """
    CI comprehensive stress test (~60–120 s):
      - 15 synthetic audio chunks fed through real WhisperEngine (tiny.en)
      - Service registered on hypha.aicell.io via hypha-rpc
      - Client reads 15 SSE events from the live /transcript_feed URL
      - Tracks per-event latency and CPU/RAM throughout

    No physical hardware required — runs on any machine with network access
    and HYPHA_WORKSPACE_TOKEN set.
    """
    import os
    import uuid
    import httpx
    import rpc.hypha_client as _hc_module
    from tests.conftest import MockMicCapture
    from transcribe.whisper_engine import WhisperEngine
    from rpc.hypha_client import HyphaClient

    token = os.environ.get("HYPHA_WORKSPACE_TOKEN", "")
    workspace = os.environ.get("HYPHA_WORKSPACE", "")
    if not token:
        pytest.skip("HYPHA_WORKSPACE_TOKEN not set")

    N_CHUNKS = 15
    MAX_EVENT_LATENCY_S = 10.0   # generous for CPU-only CI runners

    svc_id = f"hypha-whisper-test-{uuid.uuid4().hex[:8]}"
    _hc_module.SERVICE_ID = svc_id

    engine = WhisperEngine(model_name="tiny.en")
    mic = MockMicCapture([tone_pcm] * N_CHUNKS)

    client = HyphaClient(
        server_url="https://hypha.aicell.io/",
        workspace=workspace,
        token=token,
        mic_capture=mic,
        whisper_engine=engine,
    )
    await client._connect_with_backoff()
    await client._register()

    ws = client._server.config.workspace
    url = f"https://hypha.aicell.io/{ws}/apps/{svc_id}/transcript_feed"
    print(f"\n[ci-stress] {N_CHUNKS} chunks → {url}")

    events = []
    latencies = []
    cpu_samples = []
    ram_samples = []

    try:
        async with httpx.AsyncClient() as http:
            async with http.stream(
                "GET", url,
                timeout=httpx.Timeout(120.0, connect=15.0),
            ) as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers["content-type"]

                t_last = time.monotonic()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    now = time.monotonic()
                    gap = now - t_last
                    t_last = now
                    latencies.append(gap)
                    events.append(line)
                    cpu_samples.append(psutil.cpu_percent())
                    ram_samples.append(psutil.virtual_memory().percent)
                    print(f"  [{len(events):2d}/{N_CHUNKS}] +{gap:.2f}s  {line[:80]!r}")
                    assert gap < MAX_EVENT_LATENCY_S, (
                        f"Event {len(events)} took {gap:.1f}s > {MAX_EVENT_LATENCY_S}s"
                    )
                    if len(events) >= N_CHUNKS:
                        break
    finally:
        await asyncio.sleep(0.6)
        await client._server.disconnect()

    assert len(events) >= N_CHUNKS, f"Only {len(events)}/{N_CHUNKS} events received"

    avg_lat = sum(latencies) / len(latencies)
    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else 0

    print(f"\n--- CI stress summary ---")
    print(f"  Events received : {len(events)}")
    print(f"  Avg event gap   : {avg_lat:.2f}s")
    print(f"  Max event gap   : {max(latencies):.2f}s")
    print(f"  Avg CPU         : {avg_cpu:.1f}%")
    print(f"  Avg RAM         : {avg_ram:.1f}%")

    assert avg_lat < MAX_EVENT_LATENCY_S, (
        f"Average event gap {avg_lat:.2f}s > {MAX_EVENT_LATENCY_S}s"
    )


# ---------------------------------------------------------------------------
# Stress test — 30-min hardware run
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.hardware
def test_30min_continuous_transcription():
    """
    Run continuous mic capture → Whisper for 30 minutes.

    Assertions:
    - At least one transcript produced (mic is live)
    - CPU never exceeds MAX_CPU_PERCENT for 3+ consecutive samples
    - RAM never exceeds MAX_RAM_PERCENT
    - No unhandled exceptions
    """
    from audio.capture import MicCapture
    from transcribe.streaming_engine import StreamingEngine

    mic = MicCapture()
    engine = StreamingEngine(model_name="base.en", backend="whisper-timestamped")
    engine.init_session()

    mic.start()

    # Resource monitoring
    stop_teg, teg_thread, teg_samples = _start_tegrastats(interval_ms=5000)
    cpu_samples = []
    ram_samples = []
    latencies = []
    transcripts = []
    high_cpu_streak = 0

    deadline = time.monotonic() + STRESS_DURATION_S

    try:
        while time.monotonic() < deadline:
            # Poll resources every SAMPLE_INTERVAL_S
            cpu = psutil.cpu_percent(interval=SAMPLE_INTERVAL_S)
            ram = psutil.virtual_memory().percent
            cpu_samples.append(cpu)
            ram_samples.append(ram)

            # Check CPU streak
            if cpu > MAX_CPU_PERCENT:
                high_cpu_streak += 1
            else:
                high_cpu_streak = 0

            assert high_cpu_streak < 3, (
                f"CPU above {MAX_CPU_PERCENT}% for 3+ consecutive samples "
                f"(last={cpu:.1f}%)"
            )
            assert ram < MAX_RAM_PERCENT, (
                f"RAM above {MAX_RAM_PERCENT}% ({ram:.1f}%)"
            )

            # Drain any available audio chunks (non-blocking)
            while not mic.raw_audio_queue.empty():
                chunk = mic.raw_audio_queue.get_nowait()
                t0 = time.monotonic()
                engine.process_audio(chunk)
                latency = time.monotonic() - t0
                latencies.append(latency)
            # Drain committed transcripts
            while not engine.text_queue.empty():
                text = engine.text_queue.get_nowait()
                if text:
                    transcripts.append(text)

    finally:
        mic.stop()
        stop_teg.set()
        teg_thread.join(timeout=5)

    # --- Assertions ---
    assert len(transcripts) > 0, "No transcripts produced during 30-min run"

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else 0
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    max_lat = max(latencies) if latencies else 0

    # GPU stats from tegrastats
    gpu_vals = [v for line in teg_samples for v in [_parse_gpu_percent(line)] if v is not None]
    avg_gpu = sum(gpu_vals) / len(gpu_vals) if gpu_vals else None

    print("\n--- Stress test summary ---")
    print(f"  Duration        : {STRESS_DURATION_S // 60} min")
    print(f"  Transcripts     : {len(transcripts)}")
    print(f"  Avg CPU         : {avg_cpu:.1f}%")
    print(f"  Avg RAM         : {avg_ram:.1f}%")
    if avg_gpu is not None:
        print(f"  Avg GPU         : {avg_gpu:.1f}%")
    print(f"  Avg latency     : {avg_lat:.3f}s")
    print(f"  Max latency     : {max_lat:.3f}s")
    print(f"  tegrastats lines: {len(teg_samples)}")

    assert avg_lat < 2.0, f"Average transcription latency {avg_lat:.3f}s exceeds 2s target"
