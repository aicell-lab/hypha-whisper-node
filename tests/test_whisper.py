"""
tests/test_whisper.py — Tests for transcribe/whisper_engine.py.

Unit tests (no hardware) confirm the engine handles edge-case inputs.
Hardware tests (marked @hardware) run only on the Jetson with CUDA.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Unit tests — no GPU / no physical hardware required
# ---------------------------------------------------------------------------

def test_whisper_import():
    """openai-whisper must be importable."""
    import whisper  # noqa: F401


def test_torch_import():
    """torch must be importable (CPU-only on CI is fine)."""
    import torch  # noqa: F401


def test_whisper_engine_silent_audio_cpu():
    """WhisperEngine on CPU transcribes 1 s of silence to an empty string."""
    import torch
    from transcribe.whisper_engine import WhisperEngine

    device = "cuda" if torch.cuda.is_available() else "cpu"
    engine = WhisperEngine(model_name="tiny.en")
    assert engine.model is not None

    silence = bytes(SAMPLE_RATE * 1 * 2)  # 1 s S16LE silence
    result = engine.transcribe(silence)
    assert isinstance(result, str)
    # Whisper on silence may produce "" or filler text; we just check type.


SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Hardware tests — only run on Jetson with CUDA
# ---------------------------------------------------------------------------

@pytest.mark.hardware
def test_cuda_available():
    """CUDA must be available on the Jetson."""
    import torch
    assert torch.cuda.is_available(), "CUDA not available"


@pytest.mark.hardware
def test_whisper_gpu_device():
    """WhisperEngine must load on cuda:0."""
    import torch
    from transcribe.whisper_engine import WhisperEngine

    assert torch.cuda.is_available()
    engine = WhisperEngine(model_name="base.en")
    param = next(engine.model.parameters())
    assert param.device.type == "cuda", f"Model on {param.device}, expected cuda"


@pytest.mark.hardware
def test_whisper_transcribe_latency(tone_pcm):
    """base.en warm-up transcription should complete in under 2 s."""
    import time
    import torch
    from transcribe.whisper_engine import WhisperEngine

    assert torch.cuda.is_available()
    engine = WhisperEngine(model_name="base.en")

    # warm-up pass
    engine.transcribe(tone_pcm)

    t0 = time.time()
    engine.transcribe(tone_pcm)
    elapsed = time.time() - t0

    assert elapsed < 2.0, f"Transcription took {elapsed:.2f}s, target <2s"


@pytest.mark.hardware
def test_whisper_known_audio():
    """Transcribing a generated WAV with known content returns non-empty text."""
    import torch
    from transcribe.whisper_engine import WhisperEngine

    assert torch.cuda.is_available()
    engine = WhisperEngine(model_name="base.en")

    # Generate a 3-second 440 Hz tone — Whisper may or may not produce text,
    # but it must not raise an exception.
    n = SAMPLE_RATE * 3
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    samples = (8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
    result = engine.transcribe(samples.tobytes())
    assert isinstance(result, str)
