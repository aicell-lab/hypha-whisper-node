"""
tests/conftest.py — Shared pytest fixtures.

Provides hardware-free test doubles so unit and ASGI tests can run on any
machine without a microphone, GPU, or Hypha connection.
"""

import queue
import struct
import math
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic audio helpers
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000  # Hz


def _make_silence_pcm(duration_s: float = 2.0) -> bytes:
    """Return `duration_s` seconds of S16LE silence at 16 kHz."""
    n = int(SAMPLE_RATE * duration_s)
    return struct.pack(f"<{n}h", *([0] * n))


def _make_tone_pcm(freq_hz: float = 440.0, duration_s: float = 2.0,
                   amplitude: float = 8000.0) -> bytes:
    """Return a sine-wave tone as S16LE PCM.  Passes webrtcvad's energy check."""
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    samples = (amplitude * np.sin(2 * math.pi * freq_hz * t)).astype(np.int16)
    return samples.tobytes()


# ---------------------------------------------------------------------------
# Fixtures: raw PCM bytes
# ---------------------------------------------------------------------------

@pytest.fixture
def silence_pcm() -> bytes:
    """2 seconds of silence — should be rejected by VAD."""
    return _make_silence_pcm(2.0)


@pytest.fixture
def tone_pcm() -> bytes:
    """2-second 440 Hz sine wave — energetic enough to pass VAD energy gate."""
    return _make_tone_pcm(440.0, 2.0)


# ---------------------------------------------------------------------------
# Fixtures: mock objects
# ---------------------------------------------------------------------------

class MockMicCapture:
    """Drop-in for audio.capture.MicCapture — feeds pre-loaded PCM from queue."""

    def __init__(self, chunks: list[bytes]):
        self.queue: queue.Queue[bytes] = queue.Queue()
        for chunk in chunks:
            self.queue.put(chunk)

    def start(self):
        pass

    def stop(self):
        pass


class MockWhisperEngine:
    """Drop-in for transcribe.whisper_engine.WhisperEngine — returns preset text."""

    model_name = "mock"

    def __init__(self, responses: list[str] | None = None):
        # cycle through responses; default to a single fixed string
        self._responses = responses or ["Hello from mock Whisper."]
        self._idx = 0

    def transcribe(self, pcm_bytes: bytes) -> str:
        text = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return text


@pytest.fixture
def mock_mic(tone_pcm) -> MockMicCapture:
    """MicCapture stub pre-loaded with one tone chunk."""
    return MockMicCapture([tone_pcm])


@pytest.fixture
def mock_whisper() -> MockWhisperEngine:
    """WhisperEngine stub that always returns a fixed transcript."""
    return MockWhisperEngine(["Hello from mock Whisper."])
