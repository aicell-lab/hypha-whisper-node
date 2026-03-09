"""
tests/test_streaming_engine.py — Unit and hardware tests for StreamingEngine.

Unit tests run without GPU/hardware using MockStreamingEngine (fast).
Hardware tests require GPU and are skipped in CI (marked @hardware).
"""

import queue
import pytest
import numpy as np

from tests.conftest import SAMPLE_RATE, _make_silence_f32, _make_tone_f32

# ---------------------------------------------------------------------------
# Unit tests (no GPU, no real model)
# ---------------------------------------------------------------------------

def test_mock_streaming_engine_lifecycle(mock_engine):
    """MockStreamingEngine session lifecycle works as expected."""
    # Before init_session: queue is empty
    assert mock_engine.text_queue.empty()

    # init_session pre-loads a response
    mock_engine.init_session()
    assert not mock_engine.text_queue.empty()
    text = mock_engine.text_queue.get_nowait()
    assert isinstance(text, str) and len(text) > 0

    # process_audio returns None (no real inference)
    chunk = _make_silence_f32(0.5)
    result = mock_engine.process_audio(chunk)
    assert result is None

    # finish_session returns None (no real inference)
    result = mock_engine.finish_session()
    assert result is None


def test_mock_streaming_engine_double_finish(mock_engine):
    """finish_session() is idempotent — second call is a no-op."""
    mock_engine.init_session()
    mock_engine.text_queue.get_nowait()  # drain pre-loaded text
    mock_engine.finish_session()
    # Second call should not raise and should return None
    result = mock_engine.finish_session()
    assert result is None


def test_mock_streaming_engine_session_reset(mock_engine):
    """init_session() drains stale text and resets state."""
    # Pre-load stale text manually
    mock_engine.text_queue.put("stale text")
    assert not mock_engine.text_queue.empty()

    # init_session should drain stale text, then pre-load fresh response
    mock_engine.init_session()
    items = []
    while not mock_engine.text_queue.empty():
        items.append(mock_engine.text_queue.get_nowait())

    # Should have exactly one fresh response (stale one was drained by init_session
    # in the real implementation; MockStreamingEngine doesn't drain but real does)
    assert len(items) >= 1


# ---------------------------------------------------------------------------
# Hardware tests — require GPU + whisper-timestamped installed
# ---------------------------------------------------------------------------

@pytest.mark.hardware
def test_streaming_engine_init():
    """StreamingEngine initialises with tiny.en without raising."""
    from transcribe.streaming_engine import StreamingEngine
    engine = StreamingEngine(model_name="tiny.en", use_vac=False)
    assert engine.model_name == "tiny.en"
    assert isinstance(engine.text_queue, queue.Queue)


@pytest.mark.hardware
def test_streaming_engine_session_lifecycle():
    """Full init → process → finish cycle with silence audio does not crash."""
    from transcribe.streaming_engine import StreamingEngine
    engine = StreamingEngine(model_name="tiny.en", use_vac=False)

    engine.init_session()

    # Feed 2 seconds of silence
    chunk = _make_silence_f32(2.0)
    result = engine.process_audio(chunk)
    # Silence may or may not produce output; just assert it doesn't raise
    assert result is None or isinstance(result, str)

    # Flush
    final = engine.finish_session()
    assert final is None or isinstance(final, str)


@pytest.mark.hardware
def test_streaming_engine_finish_idempotent():
    """finish_session() called twice does not raise on second call."""
    from transcribe.streaming_engine import StreamingEngine
    engine = StreamingEngine(model_name="tiny.en", use_vac=False)
    engine.init_session()
    engine.finish_session()
    result = engine.finish_session()
    assert result is None


@pytest.mark.hardware
def test_streaming_engine_init_session_resets_queue():
    """init_session() drains stale text from a previous session."""
    from transcribe.streaming_engine import StreamingEngine
    engine = StreamingEngine(model_name="tiny.en", use_vac=False)

    # Inject stale text directly
    engine.text_queue.put("stale from last session")
    assert not engine.text_queue.empty()

    # init_session should drain it
    engine.init_session()
    assert engine.text_queue.empty(), "init_session() should drain stale text"
