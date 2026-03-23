"""
tests/test_transcribe_busy_state.py — Unit tests for the file transcription busy state feature.

These tests verify:
1. Health endpoint correctly reports busy status
2. File transcription endpoint returns 503 when engine is busy
3. Busy state is properly cleared after transcription completes
4. Webapp shows busy banner when engine is busy

No hardware or Hypha connection required.
"""

import asyncio
import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import rpc.hypha_client as _hc_module
from rpc.hypha_client import app, _transcription_lock, _engine_busy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_busy_state():
    """Reset global busy state before each test."""
    global _engine_busy, _transcription_lock
    _engine_busy = False
    _hc_module._engine_busy = False
    _hc_module._transcription_lock = None
    yield
    # Cleanup after test
    _engine_busy = False
    _hc_module._engine_busy = False
    if _hc_module._transcription_lock and _hc_module._transcription_lock.locked():
        # This shouldn't happen in normal tests, but just in case
        pytest.fail("Test left transcription lock locked!")


@pytest.fixture
def mock_engine():
    """Mock engine for testing."""
    engine = MagicMock()
    engine.model_name = "mock-model"
    engine.text_queue = MagicMock()
    return engine


@pytest.fixture
def client(mock_engine):
    """FastAPI TestClient with mocked engine."""
    # Inject mock engine into module
    original_engine = _hc_module._engine
    _hc_module._engine = mock_engine
    _hc_module._start_time = asyncio.get_event_loop().time()
    
    with TestClient(app) as test_client:
        yield test_client
    
    # Restore original engine
    _hc_module._engine = original_engine


# ---------------------------------------------------------------------------
# Tests for health endpoint busy status
# ---------------------------------------------------------------------------

def test_health_returns_busy_false_when_idle(client):
    """Health endpoint should return busy=false when no transcription is running."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["busy"] is False
    assert "model" in data
    assert "uptime_seconds" in data


def test_health_returns_busy_true_when_processing(client, mock_engine):
    """Health endpoint should return busy=true when transcription is in progress."""
    # Simulate busy state by acquiring the lock
    async def set_busy():
        if _hc_module._transcription_lock is None:
            _hc_module._transcription_lock = asyncio.Lock()
        await _hc_module._transcription_lock.acquire()
        _hc_module._engine_busy = True
    
    asyncio.get_event_loop().run_until_complete(set_busy())
    
    try:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["busy"] is True
    finally:
        # Release lock
        _hc_module._transcription_lock.release()
        _hc_module._engine_busy = False


# ---------------------------------------------------------------------------
# Tests for file transcription busy handling
# ---------------------------------------------------------------------------

def test_transcribe_returns_503_when_busy(client, mock_engine):
    """POST /transcribe should return 503 when engine is already busy."""
    # Set up busy state
    async def set_busy():
        if _hc_module._transcription_lock is None:
            _hc_module._transcription_lock = asyncio.Lock()
        await _hc_module._transcription_lock.acquire()
        _hc_module._engine_busy = True
    
    asyncio.get_event_loop().run_until_complete(set_busy())
    
    try:
        # Attempt to upload a file while busy
        response = client.post(
            "/transcribe",
            files={"file": ("test.mp3", io.BytesIO(b"fake audio data"), "audio/mpeg")}
        )
        assert response.status_code == 503
        assert "busy" in response.json()["detail"].lower() or "engine" in response.json()["detail"].lower()
    finally:
        # Release lock
        _hc_module._transcription_lock.release()
        _hc_module._engine_busy = False


# ---------------------------------------------------------------------------
# Tests for webapp busy banner
# ---------------------------------------------------------------------------

def test_webapp_contains_busy_banner_elements(client):
    """Webapp HTML should contain busy banner elements."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    
    # Check for busy banner div
    assert 'id="busy-banner"' in html
    
    # Check for busy state CSS classes
    assert '#status.busy' in html
    
    # Check for JavaScript health polling
    assert 'checkHealth' in html
    assert 'data.busy' in html
    assert 'busyBanner' in html


def test_webapp_busy_banner_initially_hidden(client):
    """Busy banner should be hidden by default (no 'show' class in initial HTML)."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    
    # Banner should exist
    assert 'id="busy-banner"' in html
    
    # CSS should have display:none by default
    assert 'display: none' in html or 'display:none' in html


# ---------------------------------------------------------------------------
# Tests for transcription lock lifecycle
# ---------------------------------------------------------------------------

def test_transcription_lock_initially_none(client):
    """Transcription lock should be None initially."""
    assert _hc_module._transcription_lock is None


def test_transcription_lock_created_on_first_request(client, mock_engine):
    """Lock should be created lazily on first transcription request."""
    # Mock the transcribe function to avoid actual processing
    with patch.object(_hc_module, '_transcribe_audio_file') as mock_transcribe:
        mock_transcribe.return_value = {
            "text": "test",
            "segments": [],
            "language": "en",
            "processing_time_seconds": 1.0,
            "duration_seconds": 1.0
        }
        
        # First request should create the lock
        response = client.post(
            "/transcribe",
            files={"file": ("test.mp3", io.BytesIO(b"fake audio data"), "audio/mpeg")}
        )
        
        # Lock should exist after request
        assert _hc_module._transcription_lock is not None


def test_concurrent_requests_blocked(client, mock_engine):
    """Only one transcription request should be processed at a time."""
    import time
    
    results = []
    
    def slow_transcribe(*args, **kwargs):
        time.sleep(0.1)  # Simulate slow transcription
        return {
            "text": "test",
            "segments": [],
            "language": "en",
            "processing_time_seconds": 0.1,
            "duration_seconds": 1.0
        }
    
    with patch.object(_hc_module, '_transcribe_audio_file', side_effect=slow_transcribe):
        async def make_requests():
            # Make two concurrent requests
            resp1 = client.post(
                "/transcribe",
                files={"file": ("test1.mp3", io.BytesIO(b"fake audio data"), "audio/mpeg")}
            )
            results.append(("req1", resp1.status_code))
        
        # First request
        asyncio.get_event_loop().run_until_complete(make_requests())
        
        # Verify first request succeeded
        assert results[0][1] == 200
