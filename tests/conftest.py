"""
tests/conftest.py — Shared pytest fixtures.

Provides hardware-free test doubles so unit and ASGI tests can run on any
machine without a microphone, GPU, or Hypha connection.
"""

import queue
import math
import pytest
import numpy as np


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
