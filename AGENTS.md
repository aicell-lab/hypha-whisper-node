# AGENTS.md — hypha-whisper-node

This document provides essential context for AI coding agents working on the hypha-whisper-node project.

---

## Project Overview

**hypha-whisper-node** is a portable, real-time speech-to-text edge node that:
1. Captures audio via ReSpeaker 4 Mic Array (beamformed, hardware noise suppression) or HIK camera mic
2. Transcribes locally using OpenAI Whisper (on NVIDIA Jetson GPU)
3. Streams transcripts via Hypha RPC to remote agents/dashboards using Server-Sent Events (SSE)

The system runs continuously as a systemd service, providing a live transcript viewer and SSE endpoints via Hypha's ASGI service infrastructure.

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| ML Framework | PyTorch 2.5.0 (NVIDIA JetPack wheel) |
| ASR Engine | OpenAI Whisper, whisper-timestamped, faster-whisper |
| Streaming Algorithm | whisper_streaming LocalAgreement (vendored) |
| VAD | Silero VAD (vendored iterator) |
| Audio I/O | PyAudio |
| RPC/Streaming | Hypha RPC 0.21.31, FastAPI |
| Deployment | systemd (with watchdog) |
| Testing | pytest |

### Target Hardware
- **Compute**: 
  - NVIDIA Jetson Orin Nano (JetPack 6.2, L4T R36.5, CUDA 12.6) — 8GB RAM
  - NVIDIA Jetson AGX Orin 64GB (JetPack 6.x, L4T R36.x, CUDA 12.x) — 64GB RAM, 275 TOPS
- **Microphone**: ReSpeaker 4 Mic Array v2.0 (6-channel UAC1.0, 16 kHz, beamformed ch0)
- **Fallback Mic**: HIK 1080P Camera USB microphone
- **Test Speaker**: Dell AC511 USB SoundBar **or** HDMI/DisplayPort monitor speakers (with auto-detection fallback)

---

## Project Structure

```
hypha-whisper-node/
├── main.py                      # Entry point; orchestrates audio → engine → Hypha
├── watchdog.py                  # Health watchdog (separate systemd service)
├── silero_vad_iterator.py       # Vendored from snakers4/silero-vad (MIT license)
│
├── audio/                       # Audio capture modules
│   ├── capture.py               # MicCapture: PyAudio-based microphone capture
│   ├── doa_reader.py            # DOAReader: USB DOA angle from ReSpeaker
│   ├── led_control.py           # led_off(): turn off ReSpeaker LEDs
│   └── tuning.py                # Vendored from respeaker/usb_4_mic_array
│
├── transcribe/                  # Speech recognition modules
│   ├── streaming_engine.py      # StreamingEngine: wraps OnlineASRProcessor
│   ├── speaker_registry.py      # Direction-based speaker identification
│   └── whisper_online.py        # Vendored from ufal/whisper_streaming (do not modify)
│
├── rpc/                         # Network communication
│   └── hypha_client.py          # HyphaClient: ASGI service registration, SSE endpoints
│
├── tests/                       # Test suite
│   ├── conftest.py              # Shared fixtures (MockMicCapture, MockStreamingEngine)
│   ├── test_streaming_engine.py # Unit + hardware tests for StreamingEngine
│   ├── test_hypha.py            # ASGI endpoint + Hypha integration tests
│   ├── test_speaker_registry.py # Speaker identification tests
│   ├── test_hardware_loopback.py # Acoustic loopback tests (speaker → mic → WER)
│   ├── test_stress.py           # 30-min stress test (@slow @hardware)
│   └── test_multi_client_sse.py # Multi-client SSE fan-out tests
│
├── scripts/
│   └── run_hardware_tests.sh    # Hardware test runner with service management
│
├── deploy/                      # Deployment configuration
│   ├── hypha-whisper.service        # Main systemd unit
│   └── hypha-whisper-watchdog.service  # Watchdog systemd unit
│
├── docs/
│   └── hardware.md              # Hardware setup guide
│
├── setup.sh                     # One-shot install script for Jetson
├── requirements.txt             # Production dependencies (numpy==1.26.4 pinned!)
├── requirements-dev.txt         # Development dependencies
└── pytest.ini                  # pytest configuration with markers
```

---

## Build and Installation

### Prerequisites
- JetPack 6.x (L4T R36.x) with CUDA 12.x
- Python 3.10
- System packages: `portaudio19-dev`, `ffmpeg`, `libsndfile1`

### Installation Steps

```bash
# 1. Clone and run setup
git clone https://github.com/reef-imaging/hypha-whisper-node
cd hypha-whisper-node
sudo ./setup.sh

# 2. Configure secrets
sudo nano /etc/hypha-whisper/config.env
# Add:
#   HYPHA_SERVER=https://hypha.aicell.io/
#   HYPHA_WORKSPACE=my-workspace
#   HYPHA_WORKSPACE_TOKEN=my-token

# 3. Install systemd service
sudo ./setup.sh --install-service

# 4. Start the service
sudo systemctl enable --now hypha-whisper
```

### Critical Dependency Notes

1. **numpy MUST stay at 1.26.4**: whisper-timestamped upgrades to numpy 2.x which breaks PyTorch ABI on Jetson. Do not upgrade.

2. **PyTorch from NVIDIA wheel**: Use the JetPack 6.1 wheel (works on JP6.2):
   ```
   https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
   ```

3. **libcusparseLt missing on JP6.2**: Extract from NVIDIA CUDA sbsa repo `.deb` into `~/.local/lib/` and add to `LD_LIBRARY_PATH`.

4. **LD_LIBRARY_PATH in systemd**: The service file sets (automatically configured by `setup.sh --install-service`):
   ```
   Environment="LD_LIBRARY_PATH=/home/<user>/.local/lib:/usr/local/cuda/lib64:/usr/local/cuda-12.6/lib64"
   ```

---

## Running the Application

### Manual (development)
```bash
# With Hypha streaming
python3 main.py \
  --server https://hypha.aicell.io/ \
  --workspace my-workspace \
  --token my-token \
  --model small.en \
  --backend whisper-timestamped

# Override microphone
python3 main.py --mic "ReSpeaker"     # Force ReSpeaker
python3 main.py --mic "HIK"           # Force HIK camera mic

# Offline mode (stdout only, no Hypha)
python3 main.py --server ""
```

### systemd Service Management
```bash
sudo systemctl start hypha-whisper      # Start
sudo systemctl stop hypha-whisper       # Stop (watchdog stops too)
sudo systemctl restart hypha-whisper    # Restart
systemctl status hypha-whisper          # Status
journalctl -u hypha-whisper -f          # Live logs
```

---

## Testing

### Test Markers (pytest.ini)
- `@hardware`: Requires Jetson GPU and physical mic (deselect with `-m "not hardware"`)
- `@integration`: Requires live Hypha server and `HYPHA_WORKSPACE_TOKEN` env var
- `@slow`: Takes more than 5 minutes (30-min stress test)

### Unit Tests (no hardware)
```bash
pip install -r requirements-dev.txt
pytest tests/ -m "not hardware and not integration and not slow" -v
```

### Hardware Tests (requires ReSpeaker + Speaker: USB SoundBar or HDMI/DP)
```bash
# One-time: passwordless sudo for service management
# Replace <username> with your actual username
echo "<username> ALL=(ALL) NOPASSWD: /bin/systemctl start hypha-whisper, /bin/systemctl stop hypha-whisper" \
    | sudo tee /etc/sudoers.d/hypha-whisper-tests

# Run via script (handles service stop/start)
./scripts/run_hardware_tests.sh

# Or run directly with pytest
pytest tests/test_hardware_loopback.py -m hardware -v
```

### Hardware Test Coverage
| Test | Description |
|------|-------------|
| `test_speaker_playback_only` | Speaker plays without error (USB SoundBar or HDMI/DP) |
| `test_mic_capture_rms` | ReSpeaker picks up speaker audio (RMS > 0.001) |
| `test_acoustic_loopback_wer` | Full pipeline WER < 30% against reference transcript |
| `test_speaker_identification` | Left vs right channel → 2 distinct angle labels |
| `test_speaker_stability_under_variation` | Same direction, varied audio → same angle label |

---

## Code Style Guidelines

### Python Style
- Follow PEP 8
- Use type hints for function signatures
- Docstrings for all public modules, classes, and functions
- Structured logging via `logging` module (captured by journald)

### Logging Format
```python
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
```

### Async Patterns
- Use `asyncio` for concurrency
- Blocking operations (Whisper inference) run in `ThreadPoolExecutor`
- Queue-based communication between sync and async components

### Vendored Code
The following files are vendored from external projects. **Do not modify** unless upgrading the upstream:
- `transcribe/whisper_online.py` — from ufal/whisper_streaming
- `silero_vad_iterator.py` — from snakers4/silero-vad

---

## Architecture Notes

### Audio Pipeline
```
MicCapture (PyAudio) → raw_audio_queue → audio_loop (async) → 
ThreadPoolExecutor → StreamingEngine.process_audio() → text_queue → 
_broadcast_loop → SSE clients
```

### Streaming Engine
- Uses `VACOnlineASRProcessor` (Silero VAD + OnlineASRProcessor)
- Implements LocalAgreement algorithm: text commits only when two consecutive passes agree
- Commit latency: ~3-5 seconds (trade-off for accuracy, eliminates word-boundary errors)
- Hallucination filter detects: word loops, phrase loops, exact duplicates, n-gram loops, hyphen-stutter

### Speaker Identification
- Direction-based only (no voice embeddings)
- Uses ReSpeaker USB DOA angle via `DOAReader`
- `SpeakerRegistry` assigns labels like "45°" based on angular proximity (30° threshold)
- **Known limitation**: LocalAgreement's 3-5s commit latency can cause angle misattribution if speakers overlap

### Hypha ASGI Service
Endpoints exposed via Hypha RPC:
- `GET /` — Live transcript viewer (HTML + SSE)
- `GET /transcript_feed` — SSE stream of transcripts
- `POST /transcribe` — Upload audio file for transcription (wav, mp3, m4a, etc.)
- `GET /health` — JSON status
- `GET /logs?tail=N` — SSE stream of Python logs
- `POST /clear` — Reset session and notify clients

#### File Transcription API

Upload audio files for batch transcription:

```bash
# Transcribe an audio file
curl -X POST \
  -F "file=@recording.mp3" \
  -F "language=en" \
  https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcribe
```

**Parameters:**
- `file` (required): Audio file (any format ffmpeg supports: wav, mp3, m4a, ogg, flac)
- `language` (optional): Language code hint (e.g., 'en', 'zh', 'es')
- `response_format` (optional): 'json' (default with segments + metadata) or 'text' (plain text only)

**Response (JSON):**
```json
{
  "success": true,
  "filename": "recording.mp3",
  "text": "The full transcription...",
  "segments": [
    {"start": 0.0, "end": 5.2, "text": "First segment..."}
  ],
  "language": "en",
  "processing_time_seconds": 2.145,
  "duration_seconds": 45.2
}
```

**Limits:**
- Maximum file size: 500MB
- Audio is automatically converted to 16kHz mono WAV

### Session Lifecycle
- `init_session()` — Called at startup (main.py), resets LocalAgreement state
- `finish_session()` — Called at shutdown, flushes remaining audio
- Not tied to SSE client connect/disconnect (multi-client fan-out)

---

## Security Considerations

1. **Secrets**: Stored in `/etc/hypha-whisper/config.env` (mode 600), never in code
2. **Hypha Token**: Workspace token grants access to Hypha workspace; keep confidential
3. **USB Permissions**: ReSpeaker DOA requires udev rule for USB access:
   ```
   SUBSYSTEM=="usb", ATTR{idVendor}=="2886", ATTR{idProduct}=="0018", MODE="0666", GROUP="plugdev"
   ```
4. **sudo Access**: Hardware tests require passwordless sudo for systemctl; configure via sudoers.d

---

## Deployment

### systemd Units
- **hypha-whisper.service**: Main service with `Restart=always`, `WatchdogSec=180`
- **hypha-whisper-watchdog.service**: Separate health monitoring service
- Relationship: `Wants=` and `BindsTo=` ensures watchdog lifecycle is tied to main service

### Watchdog Mechanism
1. Application calls `sd_notify("WATCHDOG=1")` every ~10s
2. systemd kills/restarts process if no watchdog signal within 180s
3. 3-minute window accommodates model loading (~26s for small.en)
4. Separate watchdog.py polls health endpoint and restarts service if unhealthy

### Environment Variables
| Variable | Purpose |
|----------|---------|
| `HYPHA_SERVER` | Hypha server URL |
| `HYPHA_WORKSPACE` | Workspace name |
| `HYPHA_WORKSPACE_TOKEN` | Authentication token |
| `WHISPER_PROMPT` | Domain vocabulary prompt for Whisper |
| `LD_LIBRARY_PATH` | CUDA and libcusparseLt library paths |
| `NOTIFY_SOCKET` | systemd notification socket (auto-set) |

---

## Troubleshooting

### Common Issues

**CUDA not available**
- Check `LD_LIBRARY_PATH` includes `~/.local/lib` and `/usr/local/cuda/lib64`
- Verify `libcusparseLt.so.0` exists in `~/.local/lib/`

**Microphone not found**
- Check USB connection: `lsusb | grep -i respeaker`
- Verify PyAudio sees device: `python3 -c "import pyaudio; pa=pyaudio.PyAudio(); [print(i, pa.get_device_info_by_index(i)['name']) for i in range(pa.get_device_count()) if pa.get_device_info_by_index(i)['maxInputChannels']>0]"`

**numpy ABI errors**
- Ensure `numpy==1.26.4` is installed: `pip show numpy`
- Check whisper-timestamped didn't upgrade it: `pip check`

**Service won't start (watchdog timeout)**
- Check model loading time in logs: `journalctl -u hypha-whisper -n 50`
- Increase `WatchdogSec` if using larger model

### Log Locations
- Application logs: `journalctl -u hypha-whisper -f`
- Watchdog logs: `journalctl -u hypha-whisper-watchdog -f`

---

## Development Workflow

1. **Make changes** to source files
2. **Run unit tests**: `pytest -m "not hardware and not slow" -v`
3. **Test on hardware** (if applicable): `./scripts/run_hardware_tests.sh`
4. **Update CLAUDE.md** if completing planned tasks (mark `[x]`)
5. **Commit** changes with AI co-authors (see below)

### AI Co-authored Commits 🤖

To credit Kimi (and other AI assistants) as contributors, use co-authored commits:

```bash
# Quick commit with Kimi as co-author
./scripts/git-commit-with-kimi.sh "Your commit message"

# Or manually:
git commit -m "Your message

Co-authored-by: Kimi <kimi@moonshot.cn>"
```

**Enable automatic co-authoring for all commits in this repo:**
```bash
# The .gitmessage template is already configured
git config commit.template .gitmessage
```

Now every commit will automatically include Kimi as a co-author. GitHub will recognize this and display 🤖 in the contributors graph.

### CI/CD
- GitHub Actions runs on push/PR to `main`
- Runs on `ubuntu-latest` with CPU-only PyTorch
- Excludes `@hardware`, `@slow` markers
- Includes `@integration` tests when `HYPHA_WORKSPACE_TOKEN` secret is set

---

## Useful Commands

```bash
# Check CUDA availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Quick Whisper latency test
python3 -c "
import torch, whisper, time
model = whisper.load_model('tiny.en', device='cuda')
audio = whisper.load_audio('tests/fixtures/reference.wav') if os.path.exists('tests/fixtures/reference.wav') else torch.randn(32000)
start = time.time()
model.transcribe(audio)
print(f'Latency: {time.time()-start:.2f}s')
"

# Monitor GPU usage
watch -n 1 tegrastats

# Test DOA reading
python3 -c "
from audio.doa_reader import DOAReader
doa = DOAReader()
doa.start()
import time; time.sleep(1)
print(f'DOA: {doa.current_direction()}°')
doa.stop()
"
```

---

## References

- [Whisper](https://github.com/openai/whisper) — OpenAI ASR model
- [whisper-timestamped](https://github.com/linto-ai/whisper-timestamped) — Word-level timestamps
- [whisper_streaming](https://github.com/ufal/whisper_streaming) — LocalAgreement streaming algorithm
- [Hypha RPC](https://pypi.org/project/hypha-rpc/) — RPC framework for bioimaging
- [ReSpeaker Mic Array v2.0](https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/) — Hardware docs
- [Jetson Orin Nano](https://developer.nvidia.com/embedded/jetson-orin-nano-developer-kit) — NVIDIA docs
- [Jetson AGX Orin](https://developer.nvidia.com/embedded/jetson-agx-orin-developer-kit) — NVIDIA docs
