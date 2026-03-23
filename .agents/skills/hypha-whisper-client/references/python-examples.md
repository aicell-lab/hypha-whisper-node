# Python Examples for Hypha Whisper Client

## Check Health Status

```python
import requests

def check_health():
    """Check if service is healthy and get busy status."""
    url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/health"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    
    print(f"Status: {data['status']}")
    print(f"Model: {data['model']}")
    print(f"Busy: {data['busy']}")
    print(f"Uptime: {data['uptime_seconds']} seconds")
    return data
```

## Upload Audio File

```python
import requests
import time

def transcribe_file(filepath: str, language: str = "en") -> dict:
    """
    Upload an audio file for transcription.
    
    Args:
        filepath: Path to audio file (wav, mp3, m4a, ogg, flac)
        language: Language code hint (e.g., 'en', 'zh')
    
    Returns:
        Transcription result dict with text, segments, metadata
    """
    url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcribe"
    
    with open(filepath, 'rb') as f:
        files = {'file': f}
        data = {'language': language}
        
        resp = requests.post(url, files=files, data=data, timeout=300)
        
    if resp.status_code == 503:
        raise RuntimeError("Engine is busy processing another file. Please retry.")
    
    resp.raise_for_status()
    return resp.json()

# Example usage
result = transcribe_file("recording.mp3")
print(f"Text: {result['text']}")
print(f"Duration: {result['duration_seconds']}s")
print(f"Processing time: {result['processing_time_seconds']}s")
for seg in result['segments']:
    print(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['text']}")
```

## Stream Live Transcripts (SSE)

```python
import requests
import json

def stream_transcripts():
    """
    Connect to SSE stream and print transcripts as they arrive.
    Press Ctrl+C to stop.
    """
    url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcript_feed"
    
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        
        for line in resp.iter_lines():
            if not line:
                continue
            
            line = line.decode('utf-8')
            
            # Skip keep-alive comments
            if line.startswith(':'):
                continue
            
            # Parse data: lines
            if line.startswith('data: '):
                data = line[6:]  # Remove 'data: ' prefix
                
                try:
                    event = json.loads(data)
                    speaker = event.get('speaker', '')
                    angle = event.get('angle')
                    text = event.get('text', '')
                    
                    if speaker:
                        angle_str = f" ({angle}°)" if angle else ""
                        print(f"[{speaker}{angle_str}] {text}")
                    else:
                        print(text)
                        
                except json.JSONDecodeError:
                    print(f"Raw: {data}")

# Run with timeout or Ctrl+C handling
try:
    stream_transcripts()
except KeyboardInterrupt:
    print("\nStopped.")
```

## Wait for Busy Engine

```python
import requests
import time

def wait_until_idle(poll_interval: float = 2.0, timeout: float = 300.0):
    """
    Poll health endpoint until engine is not busy.
    
    Args:
        poll_interval: Seconds between polls
        timeout: Maximum seconds to wait
    """
    url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/health"
    start = time.time()
    
    while time.time() - start < timeout:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        if not data.get('busy', False):
            print("Engine is idle.")
            return True
        
        print(f"Engine busy... waiting {poll_interval}s")
        time.sleep(poll_interval)
    
    raise TimeoutError("Engine remained busy beyond timeout")

# Usage: wait then upload
wait_until_idle()
result = transcribe_file("my-audio.wav")
```

## Stream Logs

```python
import requests
import json

def stream_logs(tail: int = 50):
    """Stream service logs via SSE."""
    url = f"https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/logs?tail={tail}"
    
    with requests.get(url, stream=True, timeout=60) as resp:
        for line in resp.iter_lines():
            if not line:
                continue
            
            line = line.decode('utf-8')
            if line.startswith('data: '):
                try:
                    log_entry = json.loads(line[6:])
                    ts = log_entry.get('ts', 0)
                    level = log_entry.get('level', 'INFO')
                    msg = log_entry.get('msg', '')
                    print(f"[{level}] {msg}")
                except json.JSONDecodeError:
                    pass
```

## Clear Session

```python
import requests

def clear_session():
    """Reset the transcription session."""
    url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/clear"
    resp = requests.post(url, timeout=10)
    resp.raise_for_status()
    return resp.json()
```

## Async Examples (asyncio)

```python
import aiohttp
import asyncio
import json

async def check_health_async():
    """Async health check."""
    url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/health"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

async def transcribe_file_async(filepath: str, language: str = "en"):
    """Async file upload."""
    url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcribe"
    
    async with aiohttp.ClientSession() as session:
        data = aiohttp.FormData()
        data.add_field('language', language)
        
        with open(filepath, 'rb') as f:
            data.add_field('file', f, filename=filepath)
            async with session.post(url, data=data) as resp:
                if resp.status == 503:
                    raise RuntimeError("Engine is busy")
                return await resp.json()

async def stream_transcripts_async():
    """Async SSE transcript streaming."""
    url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcript_feed"
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    try:
                        event = json.loads(line[6:])
                        print(f"[{event.get('speaker', '?')}] {event.get('text', '')}")
                    except json.JSONDecodeError:
                        pass

# Run async examples
# asyncio.run(check_health_async())
```
