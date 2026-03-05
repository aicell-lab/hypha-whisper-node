"""
tests/test_vad.py — Unit tests for VAD logic in audio/capture.py.

No hardware required.  Pure Python + webrtcvad.
"""

import numpy as np
import pytest

from audio.capture import _vad_has_speech, SAMPLE_RATE, VAD_FRAME_MS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_bytes() -> int:
    return int(SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2  # S16LE: 2 bytes/sample


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_silence_rejected(silence_pcm):
    """Pure silence must not pass the VAD gate."""
    assert _vad_has_speech(silence_pcm) is False


def test_empty_pcm_rejected():
    """Empty byte string must not crash and must return False."""
    assert _vad_has_speech(b"") is False


def test_single_frame_silence():
    """Exactly one silent frame — below ratio threshold → rejected."""
    n_samples = int(SAMPLE_RATE * VAD_FRAME_MS / 1000)
    pcm = bytes(n_samples * 2)  # all zeros
    assert _vad_has_speech(pcm) is False


def test_tone_may_pass(tone_pcm):
    """A loud 440 Hz sine wave has enough energy to be classified as voiced
    by webrtcvad (aggressiveness=2).  If webrtcvad disagrees on a given
    platform the test is skipped rather than failed — VAD is statistical."""
    result = _vad_has_speech(tone_pcm)
    # We don't assert True because webrtcvad may reject non-speech tones;
    # we assert it doesn't crash and returns a bool.
    assert isinstance(result, bool)


def test_high_amplitude_pcm_does_not_crash():
    """Maximum-amplitude square wave — must not raise."""
    n = int(SAMPLE_RATE * 2)
    samples = np.tile(np.array([32767, -32768], dtype=np.int16), n // 2)
    pcm = samples.tobytes()
    result = _vad_has_speech(pcm)
    assert isinstance(result, bool)


def test_odd_length_pcm_handled():
    """PCM with an odd number of bytes — partial last frame is silently ignored."""
    silence = bytes(int(SAMPLE_RATE * 2) * 2 + 1)  # one extra byte
    assert _vad_has_speech(silence) is False


def test_vad_aggressiveness_range():
    """VAD should work for all supported aggressiveness levels 0–3."""
    pcm = bytes(int(SAMPLE_RATE * 2) * 2)  # 2 s of silence
    for level in range(4):
        result = _vad_has_speech(pcm, aggressiveness=level)
        assert result is False, f"Silence passed VAD at aggressiveness={level}"
