"""
tests/conftest.py — Shared pytest fixtures.

Provides hardware-free test doubles so unit and ASGI tests can run on any
machine without a microphone, GPU, or Hypha connection.
"""

import queue
import math
import subprocess
import threading
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Service management — stop hypha-whisper before hardware tests, restore after
# ---------------------------------------------------------------------------

def _service_is_active() -> bool:
    result = subprocess.run(
        ["systemctl", "is-active", "--quiet", "hypha-whisper"],
        capture_output=True,
    )
    return result.returncode == 0


def _service_is_enabled() -> bool:
    result = subprocess.run(
        ["systemctl", "is-enabled", "--quiet", "hypha-whisper"],
        capture_output=True,
    )
    return result.returncode == 0


def _sudo_passwordless() -> bool:
    """Return True if sudo systemctl stop can run without a password prompt."""
    result = subprocess.run(
        ["sudo", "-n", "systemctl", "stop", "hypha-whisper"],
        capture_output=True,
    )
    return b"password" not in result.stderr


def _service_control(action: str):
    subprocess.run(["sudo", "systemctl", action, "hypha-whisper"],
                   capture_output=True)


@pytest.fixture(scope="session", autouse=False)
def suspend_service():
    """Stop hypha-whisper before hardware tests; restart it afterward if it was enabled.

    Requires passwordless sudo for systemctl stop/start. Set up once with:
        echo "YOUR_USER ALL=(ALL) NOPASSWD: /bin/systemctl start hypha-whisper, /bin/systemctl stop hypha-whisper" \\
            | sudo tee /etc/sudoers.d/hypha-whisper-tests

    Runs a background keeper thread that unconditionally stops the service every 3 s,
    preventing systemd's Restart=always (including the activating phase) from
    reclaiming the microphone during tests.

    If passwordless sudo is not configured, the fixture skips with instructions.
    """
    # Check enabled state BEFORE _sudo_passwordless() which stops the service as a side-effect.
    was_enabled = _service_is_enabled()
    can_sudo = _sudo_passwordless()  # also stops the service if running

    if _service_is_active():
        if not can_sudo:
            pytest.skip(
                "hypha-whisper is running and passwordless sudo is not configured. "
                "Either stop it manually ('sudo systemctl stop hypha-whisper') "
                "or add a sudoers rule — see conftest.py suspend_service docstring."
            )

    if not can_sudo:
        yield
        return

    print("\n[fixture] hypha-whisper stopped; keeping it down during tests...")

    # Keeper: stop unconditionally every 3 s — catches activating+active phases.
    _stop_keeper = threading.Event()

    def _keep_stopped():
        while not _stop_keeper.is_set():
            _service_control("stop")
            _stop_keeper.wait(3)

    keeper_thread = threading.Thread(target=_keep_stopped, daemon=True)
    keeper_thread.start()

    yield

    _stop_keeper.set()
    keeper_thread.join(timeout=5)

    if was_enabled:
        print("\n[fixture] Restarting hypha-whisper service...")
        _service_control("start")
        print("[fixture] hypha-whisper restarted.")


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000  # Hz


def _make_silence_f32(duration_s: float = 2.0) -> np.ndarray:
    """Return `duration_s` seconds of float32 silence at 16 kHz."""
    return np.zeros(int(SAMPLE_RATE * duration_s), dtype=np.float32)


def _make_tone_f32(freq_hz: float = 440.0, duration_s: float = 2.0,
                   amplitude: float = 0.25) -> np.ndarray:
    """Return a sine-wave tone as float32 in [-1, 1]."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * math.pi * freq_hz * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures: raw numpy audio chunks
# ---------------------------------------------------------------------------

@pytest.fixture
def silence_f32() -> np.ndarray:
    """2 seconds of float32 silence."""
    return _make_silence_f32(2.0)


@pytest.fixture
def tone_f32() -> np.ndarray:
    """2-second 440 Hz float32 sine wave chunk."""
    return _make_tone_f32(440.0, 2.0)


# ---------------------------------------------------------------------------
# Mock objects
# ---------------------------------------------------------------------------

class MockMicCapture:
    """Drop-in for audio.capture.MicCapture — feeds pre-loaded np.float32 chunks."""

    def __init__(self, chunks: list):
        self.raw_audio_queue: queue.Queue = queue.Queue()
        for chunk in chunks:
            self.raw_audio_queue.put(chunk)

    def start(self):
        pass

    def stop(self):
        pass


class MockStreamingEngine:
    """Drop-in for transcribe.streaming_engine.StreamingEngine.

    init_session() pre-loads one response into text_queue so SSE tests
    can immediately receive a transcript without running real Whisper.
    """

    model_name = "mock"

    def __init__(self, responses: list = None):
        self.text_queue: queue.Queue = queue.Queue()
        self._responses = responses or ["Hello from mock streaming."]
        self._idx = 0
        self._session_active = False

    def init_session(self, offset=None) -> None:
        self._session_active = True
        # Pre-load the next response so SSE gen can yield it immediately.
        text = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        self.text_queue.put(text)

    def process_audio(self, chunk: np.ndarray):
        return None

    def finish_session(self):
        self._session_active = False
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_mic(tone_f32) -> MockMicCapture:
    """MicCapture stub pre-loaded with one tone chunk."""
    return MockMicCapture([tone_f32])


@pytest.fixture
def mock_engine() -> MockStreamingEngine:
    """StreamingEngine stub that pre-loads one transcript on init_session()."""
    return MockStreamingEngine(["Hello from mock streaming."])
