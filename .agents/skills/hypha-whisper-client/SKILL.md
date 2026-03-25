---
name: hypha-whisper-client
description: Interact with the hypha-whisper speech-to-text service at https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/. Use when users need to check transcription status, upload audio files for transcription, monitor live transcripts, or troubleshoot the whisper node service.
---

# Hypha Whisper Client

This skill provides tools to interact with the hypha-whisper-node real-time speech-to-text service.

## Service Overview

The hypha-whisper-node is a portable, privacy-first speech-to-text edge node running on NVIDIA Jetson hardware. It uses **whisper-timestamped** (OpenAI Whisper) as the ASR backend with optimized fp16 inference on GPU.

### Important Hardware Requirement

**To use this service, you must have the actual hardware running:**
- NVIDIA Jetson Orin Nano (or similar edge device) - for on-device inference
- Microphone array (6-channel XMOS XVF-3000 DSP or ReSpeaker 4 Mic Array) - required for real-time streaming
- Physical computer with the hardware connected and powered on

The endpoints in this skill (live transcription, file upload, health checks) only work when the actual hardware node is deployed and running. This is not a cloud service - it's a physical edge device that must be on-site and operational.

### Backend Details
- **ASR Engine:** whisper-timestamped (OpenAI Whisper)
- **Optimization:** fp16 on CUDA, beam_size=3, condition_on_previous_text=False
- **VAD:** Silero VAD for voice activity detection
- **Hallucination Filter:** Built-in filtering for repetition patterns

### Capabilities

- **Live real-time transcription** via SSE stream at `/transcript_feed`
- **File upload transcription** via POST `/transcribe`
- **Health monitoring** via GET `/health`
- **Live transcript viewer** webapp at `/`

Base URL: `https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/`

## Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Live transcript viewer (HTML page) |
| `/transcript_feed` | GET | SSE stream of real-time transcripts |
| `/transcribe` | POST | Upload audio file for transcription |
| `/health` | GET | Service health status + busy state |
| `/logs?tail=N` | GET | SSE stream of service logs |
| `/clear` | POST | Reset session and clear transcripts |

## Common Tasks

### Check Service Status

```bash
curl https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/health
```

Response:
```json
{
  "status": "ok",
  "busy": false,
  "model": "small.en",
  "uptime_seconds": 3600
}
```

**Note:** When `busy: true`, the engine is processing a file upload and real-time transcription is paused.

### Upload Audio File for Transcription

```bash
curl -X POST \
  -F "file=@recording.mp3" \
  -F "language=en" \
  https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcribe
```

**Parameters:**
- `file` (required): Audio file (wav, mp3, m4a, ogg, flac)
- `language` (optional): Language code hint (e.g., 'en', 'zh')
- `response_format` (optional): 'json' (default) or 'text'

**Limits:**
- Max file size: 500MB
- Only one file processed at a time (mutual exclusion with streaming)
- Returns HTTP 503 if engine is busy

### Monitor Live Transcripts (curl)

```bash
curl https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcript_feed
```

### Clear Session

```bash
curl -X POST https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/clear
```

## Python Examples

See [references/python-examples.md](references/python-examples.md) for complete Python code examples.

## Local File Transcription (Offline)

For transcribing audio files locally using the same backend (without the service):

```python
#!/usr/bin/env python3
"""Local transcription using whisper-timestamped backend."""
import sys
sys.path.insert(0, '/path/to/hypha-whisper-node')

import numpy as np
import librosa
from transcribe.streaming_engine import StreamingEngine

# Load audio
audio, sr = librosa.load("recording.m4a", sr=16000, mono=True)

# Create engine with whisper-timestamped backend
engine = StreamingEngine(model_name="base.en", use_vac=True)
engine.init_session()

# Process audio chunks
chunk_size = int(0.5 * 16000)  # 0.5s chunks
for i in range(0, len(audio), chunk_size):
    chunk = audio[i:i+chunk_size]
    if len(chunk) > 0:
        engine.process_audio(chunk)

# Get final transcript
final = engine.finish_session()
print(final)
```

**Key differences from faster-whisper:**
- Uses OpenAI Whisper + whisper-timestamped for word-level timestamps
- Optimized with fp16 on CUDA, beam_size=3
- Built-in hallucination filtering for streaming

## Troubleshooting

### Engine is Busy

If `/health` returns `busy: true`, the engine is processing a file upload:
- Wait for current transcription to complete
- Real-time streaming is paused during file processing
- The webapp shows a "🔄 Whisper engine is busy" banner

### Service Unavailable

If health check fails:
1. Check Jetson is powered on and connected
2. Verify systemd service: `systemctl status hypha-whisper`
3. Check logs: `journalctl -u hypha-whisper -n 50`

### No Transcripts Appearing

- Ensure microphone is connected (ReSpeaker 4 Mic Array or HIK camera)
- Check audio is being captured: look for logs with `[audio_loop]`
- Verify SSE connection is active in browser dev tools

## Data Privacy

- **No audio stored** — all processing is in-memory only
- **No cloud processing** — Whisper runs locally on Jetson GPU
- **No transcript history** — transcripts stream via SSE and are discarded
- See [PRIVACY.md](https://github.com/reef-imaging/hypha-whisper-node/blob/main/PRIVACY.md) for full details
