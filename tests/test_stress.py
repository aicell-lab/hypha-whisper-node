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
# Stress test
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
    from transcribe.whisper_engine import WhisperEngine

    mic = MicCapture()
    engine = WhisperEngine(model_name="base.en", device="cuda")

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
            while not mic.queue.empty():
                pcm = mic.queue.get_nowait()
                t0 = time.monotonic()
                text = engine.transcribe(pcm)
                latency = time.monotonic() - t0
                latencies.append(latency)
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
