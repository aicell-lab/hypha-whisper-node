---
name: hypha-whisper
description: |
  Transcribe audio files using the Hypha Whisper Node - a GPU-accelerated 
  Whisper deployment on NVIDIA Jetson. Supports real-time streaming and 
  file-based transcription via HTTP endpoints.
  
  Use this skill when you need to:
  - Transcribe audio recordings to text
  - Convert speech in audio files to written text
  - Process audio files for speech-to-text conversion
  - Access real-time streaming transcription services
---

# Hypha Whisper Node Skill

This skill enables agents to transcribe audio files using the Hypha Whisper Node deployed on NVIDIA Jetson hardware with GPU acceleration.

## Capabilities

1. **File Transcription** - Upload audio files and get text transcriptions with timestamps
2. **Real-time Streaming** - Connect to live transcript feed via SSE

## Usage

### File Transcription

Transcribe an audio file using the `/transcribe` endpoint:

```bash
curl -X POST \
  -F "file=@/path/to/audio.mp3" \
  -F "language=en" \
  -F "response_format=json" \
  https://hypha.aicell.io/{workspace}/apps/hypha-whisper/transcribe
```

**Parameters:**
- `file` (required): Audio file (wav, mp3, m4a, ogg, flac, etc.)
- `language` (optional): Language code hint (e.g., 'en', 'zh', 'es')
- `response_format` (optional): 'json' (default) or 'text'

**Response (JSON format):**
```json
{
  "success": true,
  "filename": "audio.mp3",
  "text": "The full transcription text...",
  "segments": [
    {"start": 0.0, "end": 5.2, "text": "First segment..."},
    {"start": 5.2, "end": 10.1, "text": "Second segment..."}
  ],
  "language": "en",
  "processing_time_seconds": 2.145,
  "duration_seconds": 45.2
}
```

### Real-time Streaming

Connect to the live transcript feed:

```bash
curl https://hypha.aicell.io/{workspace}/apps/hypha-whisper/transcript_feed
```

The endpoint returns Server-Sent Events (SSE) with transcription segments as they are recognized.

### Health Check

Check service status:

```bash
curl https://hypha.aicell.io/{workspace}/apps/hypha-whisper/health
```

## Supported Audio Formats

- WAV (16kHz mono preferred)
- MP3
- M4A / AAC
- OGG / Vorbis
- FLAC
- And any other format supported by ffmpeg

## Hardware Requirements

- NVIDIA Jetson Orin Nano (8GB) or Jetson AGX Orin (64GB)
- JetPack 6.x with CUDA 12.x
- ReSpeaker 4 Mic Array (for real-time streaming with DOA)

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Live transcript viewer (HTML page) |
| `/transcript_feed` | GET | SSE stream of live transcriptions |
| `/transcribe` | POST | Upload audio file for transcription |
| `/health` | GET | Service health status |
| `/logs` | GET | SSE stream of application logs |
| `/clear` | POST | Reset session state |

## Example Workflows

### Transcribe a Meeting Recording

```python
import requests

url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcribe"
with open("meeting.mp3", "rb") as f:
    response = requests.post(
        url,
        files={"file": f},
        data={"language": "en", "response_format": "json"}
    )
result = response.json()
print(f"Transcription: {result['text']}")
print(f"Processing time: {result['processing_time_seconds']}s")
```

### Batch Transcribe Multiple Files

```python
import requests
import glob

url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcribe"
for audio_file in glob.glob("recordings/*.mp3"):
    with open(audio_file, "rb") as f:
        response = requests.post(url, files={"file": f})
        result = response.json()
        print(f"{audio_file}: {result['text'][:100]}...")
```

### Monitor Live Transcription

```python
import requests

def stream_transcripts(workspace="reef-imaging"):
    url = f"https://hypha.aicell.io/{workspace}/apps/hypha-whisper/transcript_feed"
    response = requests.get(url, stream=True)
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))

stream_transcripts()
```

## Error Handling

Common HTTP status codes:
- `200` - Success
- `400` - Bad request (invalid parameters)
- `413` - File too large (max 500MB)
- `500` - Transcription error or internal server error

## Notes

- File uploads are limited to 500MB
- Audio is automatically converted to 16kHz mono WAV
- GPU acceleration provides ~10x faster than real-time transcription
- The service requires ffmpeg to be installed for audio format conversion
