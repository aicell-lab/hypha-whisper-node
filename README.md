# hypha-whisper-node

Portable real-time speech-to-text node powered by Whisper and NVIDIA Jetson.
Captures speech via ReSpeaker 4 Mic Array, transcribes on-device using the LocalAgreement streaming algorithm, and streams results through [Hypha RPC](https://pypi.org/project/hypha-rpc/).

<p align="center">
  <img src="https://img.shields.io/badge/%F0%9F%94%92%20Privacy-First-blue" alt="Privacy-First">
  <img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Local%20Processing-no%20cloud-green" alt="Local Processing">
  <img src="https://img.shields.io/badge/%F0%9F%93%A1%20No%20Telemetry-none-orange" alt="No Telemetry">
  <img src="https://img.shields.io/badge/%E2%9C%85%20Open%20Source-MIT-success" alt="Open Source MIT">
</p>

> **🔒 Privacy Guarantee:** Your voice is NEVER recorded or stored. All processing happens on-device. [Read our Privacy Policy →](PRIVACY.md)

**It's live! Give it a try:**

> [**🎙️ Open Live Transcript Viewer →**](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/) &nbsp;&nbsp; [**📡 SSE Stream**](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcript_feed) &nbsp;&nbsp; [**💚 Health**](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/health)&nbsp; 
[**📋 Logs**](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/logs)

---

## Hardware

| Component | Details |
|---|---|
| Compute | Jetson Orin Nano **or** Jetson AGX Orin 64GB (JetPack 6.x, CUDA 12.x) |
| Microphone | ReSpeaker 4 Mic Array v2.0 (UAC1.0, 16 kHz, beamformed ch0) |
| Speaker (test) | Dell AC511 USB SoundBar **or** HDMI/DisplayPort monitor speakers |
| Power | USB-C PD power bank |
| Enclosure | 3D printed shell (ventilation + antenna ports) |

**Mic auto-detection:** ReSpeaker is tried first; falls back to HIK 1080P Camera if not found. Override with `--mic "name-substring"`.

**Speaker auto-detection:** Tests use Dell AC511 USB SoundBar first, falling back to HDMI/DisplayPort monitor speakers if not found.

---

## Features

- Continuous streaming transcription using **whisper_streaming LocalAgreement** — no word-boundary errors from chunk splitting
- On-device Whisper inference via PyTorch (GPU, works offline)
- Real-time transcript streaming via Hypha ASGI service (SSE)
- Live transcript viewer — browser-based HTML page at `/`
- ReSpeaker 4 Mic Array: hardware beamforming + 4-mic noise suppression via ch0
- **Direction annotation** — ReSpeaker USB DOA angle tags each utterance with the speaker's direction (e.g. `45°`); note: speaker grouping is best-effort only (see known limitations)
- Auto-reconnect to Hypha on network loss (exponential backoff)
- systemd service with watchdog (`WatchdogSec=180`) and auto-restart

---

## Architecture & Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         hypha-whisper-node Architecture                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌──────────┐
│  ReSpeaker  │     │   MicCapture│     │  StreamingEngine│     │ Hypha    │
│  4-Mic Array│────▶│   (PyAudio) │────▶│  (Whisper ASR)  │────▶│  Client  │
│  (USB Audio)│     │             │     │                 │     │          │
└─────────────┘     └─────────────┘     └─────────────────┘     └────┬─────┘
       │                    │                    │                   │
       │                    │                    │                   │
       ▼                    ▼                    ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐     ┌──────────┐
│ XMOS XVF-   │     │ 6-channel   │     │ LocalAgreement  │     │ ASGI     │
│ 3000 DSP    │     │ audio:      │     │ Algorithm       │     │ Service  │
│             │     │ • ch0=ASR   │     │                 │     │          │
│ • Beamform  │     │ • ch1-4=DOA │     │ • VAD (Silero)  │     │ Endpoints│
│ • DOA       │     │             │     │ • Buffer 3-5s   │     │          │
│ • AEC/NS    │     │             │     │ • Commit text   │     │ /        │
└─────────────┘     └─────────────┘     └─────────────────┘     │ /transcript_feed
                                                               │ /health
                                                               │ /logs
                                                               └────┬─────┘
                                                                    │
                                                                    ▼
                                                            ┌──────────────┐
                                                            │   Browser    │
                                                            │   (SSE)      │
                                                            └──────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         Data Flow (Privacy-First)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Speech ──▶ Memory ──▶ Whisper (local GPU) ──▶ Text ──▶ SSE ──▶ Discard   │
│              (temp)        (no cloud)              │                        │
│                                                    ▼                        │
│                                              ┌─────────┐                    │
│                                              │  DOA    │                    │
│                                              │  Buffer │                    │
│                                              │ (timing │                    │
│                                              │  fix)   │                    │
│                                              └─────────┘                    │
│                                                                             │
│   🔒 No audio stored  🔒 No transcript history  🔒 No cloud processing     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    DOA Time-Alignment (Key Fix)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Problem: LocalAgreement buffers 3-5s, so text commits AFTER audio captured │
│                                                                             │
│  WRONG (old):  DOA at commit time ──▶ misattributes if speaker changes      │
│                                                                             │
│  CORRECT (new):                                                             │
│    1. DOA polled from firmware every 50ms ──▶ store as time intervals       │
│    2. When text commits, get its actual time range [begin, end]             │
│    3. Query: which DOA angle had longest overlap with [begin, end]?         │
│                                                                             │
│    Speaker 1: 0-4.9s @ 90°  ═══╤═══  Dominant = 90° (4.9s > 0.1s)         │
│    Speaker 2: 4.9-5s @ 341° ───╯                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
hypha-whisper-node/
├── main.py                      # Entry point, orchestrates all components
├── audio/
│   ├── capture.py               # PyAudio microphone capture
│   └── doa_reader.py            # ReSpeaker USB DOA + IntervalBuffer
├── transcribe/
│   ├── streaming_engine.py      # Whisper + LocalAgreement + DOA alignment
│   ├── speaker_registry.py      # Direction-based speaker labeling
│   └── whisper_online.py        # Vendored from whisper_streaming
├── rpc/
│   └── hypha_client.py          # Hypha RPC ASGI service (SSE endpoints)
└── tests/
    └── test_hardware_loopback.py # Acoustic WER + DOA verification
```

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Audio Capture** | PyAudio | ReSpeaker 6-channel (ch0=ASR, ch1-4=raw) |
| **DOA Estimation** | XMOS XVF-3000 | On-chip direction detection via USB |
| **ASR Engine** | Whisper + LocalAgreement | Streaming transcription with buffering |
| **VAD** | Silero VAD | Voice activity detection |
| **DOA Alignment** | Duration-weighted overlap | Correct attribution during speaker changes |
| **Streaming** | Hypha RPC + SSE | Real-time text delivery to browsers |
| **Watchdog** | systemd + sd_notify | Auto-restart on hang or crash |

---

## 🔒 Privacy & Security

**hypha-whisper-node** is built with privacy as a foundational principle:

| Privacy Feature | Status |
|----------------|--------|
| 🎤 **Audio Storage** | ❌ Never saved — audio is processed in real-time and immediately discarded |
| 📝 **Transcript Storage** | ❌ Never persisted — transcripts exist only in memory |
| ☁️ **Cloud Transcription** | ❌ None — all processing is on-device (Jetson GPU) |
| 📡 **Telemetry** | ❌ None — no analytics or usage data collected |
| 🔓 **Open Source** | ✅ 100% — fully auditable codebase |
| 🏠 **Offline Mode** | ✅ Yes — works without any network connection |

### How It Works
```
Microphone → Memory → Whisper (local) → SSE Stream → Discard
                ↓            ↓              ↓
           (temporary)   (no cloud)    (live only)
```

- **No audio files** are ever written to disk
- **No transcript history** is stored — when the service restarts, everything is gone
- **No voice data** is sent to external servers for transcription
- Optional Hypha streaming only sends **text** (never audio) to your configured endpoint

Run completely offline:
```bash
python3 main.py --server ""
```

📖 **[Read the full Privacy Policy →](PRIVACY.md)**

---

## Whisper Benchmarks

### Jetson Orin Nano (2 s audio, GPU)

| Model | Avg latency | Load time |
|-------|------------|-----------|
| tiny.en | 0.19 s | 6 s |
| base.en | 0.40 s | 4 s |
| **small.en** (default) | 0.92 s | 26 s |

`small.en` with `whisper-timestamped` backend (PyTorch + CUDA) is the default.
---

## Installation

```bash
git clone https://github.com/reef-imaging/hypha-whisper-node
cd hypha-whisper-node
sudo ./setup.sh
```

`setup.sh` installs system packages, PyTorch (NVIDIA JP6.1 wheel), all Python deps, and creates `/etc/hypha-whisper/config.env`.

> **numpy pin:** `numpy==1.26.4` is pinned in `requirements.txt`. Do not upgrade — `whisper-timestamped` pulls numpy 2.x which breaks PyTorch's ABI on Jetson.

### Configure secrets

```bash
sudo nano /etc/hypha-whisper/config.env
```

```env
HYPHA_SERVER=https://hypha.aicell.io/
HYPHA_WORKSPACE=my-workspace
HYPHA_WORKSPACE_TOKEN=my-token
```

### Install the systemd service

```bash
# Option 1: Auto-install with setup script (recommended)
sudo ./setup.sh --install-service

# Option 2: Manual install
sudo cp deploy/hypha-whisper.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now hypha-whisper
```

---

## Service Management

All commands below also start/stop/restart the watchdog automatically.

| Task | Command |
|------|---------|
| Start | `sudo systemctl start hypha-whisper` |
| Stop | `sudo systemctl stop hypha-whisper` |
| Restart | `sudo systemctl restart hypha-whisper` |
| Status | `systemctl status hypha-whisper` |
| Watchdog status | `systemctl status hypha-whisper-watchdog` |
| Live logs | `journalctl -u hypha-whisper -f` |
| Watchdog logs | `journalctl -u hypha-whisper-watchdog -f` |
| Last 100 lines | `journalctl -u hypha-whisper -n 100` |
| Disable autostart | `sudo systemctl disable hypha-whisper hypha-whisper-watchdog` |

Logs look like:

```
2026-03-09T12:55:19 INFO [MicCapture] Found 'ReSpeaker 4 Mic Array...' at device index 24 (capture ch=6, use ch=0)
2026-03-09T12:55:25 INFO [StreamingEngine] Ready
2026-03-09T12:55:31 INFO [hypha] Connected to https://hypha.aicell.io (workspace: reef-imaging)
2026-03-09T12:55:54 INFO [transcript] Transcript sent to 1 client(s)
```

> **🔒 Privacy Note:** Transcript text is intentionally NOT logged. Only metadata (timestamps, client counts, connection events) appears in logs.

If the Hypha server drops, the service reconnects automatically (exponential backoff, max 60 s).

---

## Endpoints

Once running, the service exposes these endpoints via Hypha:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Live transcript viewer — open in any browser |
| `GET /transcript_feed` | SSE stream — one `data: <json>` event per committed phrase |
| `GET /health` | JSON: `{"status":"ok","model":"small.en","uptime_seconds":123}` |
| `GET /logs?tail=N` | SSE stream of all Python log records; `tail=N` replays last N lines first |

**Live deployment (reef-imaging workspace):**

| URL | Description |
|-----|-------------|
| [hypha.aicell.io/reef-imaging/apps/hypha-whisper/](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/) | Live transcript viewer |
| [hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcript_feed](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcript_feed) | SSE transcript stream |
| [hypha.aicell.io/reef-imaging/apps/hypha-whisper/health](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/health) | Health check |
| [hypha.aicell.io/reef-imaging/apps/hypha-whisper/logs?tail=100](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/logs?tail=100) | Live log stream |

Full URL pattern:
```
https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/
https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/transcript_feed
https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/health
https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/logs?tail=100
```

### Live transcript viewer

Open `https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/` in a browser. The page connects automatically to `transcript_feed` via `EventSource`, accumulates text, and auto-scrolls. A **Clear** button resets the display. The connection indicator shows green when live and retries automatically on disconnect.

Each transcript segment is tagged with a coloured direction badge (e.g. **45°**) showing the DOA angle when the speech was detected. Consecutive segments from the same direction are grouped into one line.

> **✅ Fixed — DOA attribution:** Previously, speaker angles could be misattributed when multiple people spoke during LocalAgreement's 3-5s buffering period. The fix uses duration-weighted overlap: each transcript segment is tagged with the DOA angle that had the longest overlap with its actual time range (learned from WhisperX's interval tree approach).

### SSE payload format

Each `data:` event is a JSON object:

```json
{"text": "Hello everyone.", "speaker": "45°", "angle": 45}
```

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Committed transcript phrase |
| `speaker` | string | Speaker direction label, e.g. `"45°"` (empty if DOA unavailable) |
| `angle` | int \| null | Raw DOA angle in degrees, or `null` |

### Consuming the SSE stream programmatically

```python
import httpx

url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcript_feed"
with httpx.stream("GET", url) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            print(line[6:])
```

### Log stream (`/logs`)

Designed for AI agents and automated monitoring. Each SSE event is a JSON object:

```json
{"ts": 1741694400.123, "level": "INFO", "logger": "transcribe.streaming_engine", "msg": "12:00:00 INFO     ... — Engine ready"}
```

| Field | Description |
|-------|-------------|
| `ts` | Unix timestamp (float) |
| `level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `logger` | Python logger name (e.g. `rpc.hypha_client`) |
| `msg` | Fully formatted log line |

Query parameter `tail=N` replays the last N buffered records (up to 2000) before streaming live — useful for catching up after connecting:

```python
import httpx, json

url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/logs"
with httpx.stream("GET", url, params={"tail": 100}) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            record = json.loads(line[6:])
            print(f"[{record['level']}] {record['msg']}")
```

---

## Run manually (without systemd)

```bash
python3 main.py \
  --server https://hypha.aicell.io/ \
  --workspace my-workspace \
  --token my-token \
  --model small.en \
  --backend whisper-timestamped
```

Override microphone:
```bash
python3 main.py --mic "ReSpeaker"     # force ReSpeaker
python3 main.py --mic "HIK"           # force HIK camera mic
```

Offline mode (transcribe to stdout, no Hypha):
```bash
python3 main.py --server ""
```

---

## Testing

### Unit tests (no hardware required)

```bash
pip install -r requirements-dev.txt
pytest tests/ -m "not hardware and not integration and not slow"
```

### Hardware loopback tests (ReSpeaker + Dell AC511 required)

Plays pre-recorded reference audio (`tests/test-audio-male.wav`) through the Dell AC511 speaker, records via ReSpeaker, transcribes, and measures Word Error Rate.

```bash
# One-time: allow passwordless sudo for service management during tests
# Replace <username> with your actual username
echo "<username> ALL=(ALL) NOPASSWD: /bin/systemctl start hypha-whisper, /bin/systemctl stop hypha-whisper" \
    | sudo tee /etc/sudoers.d/hypha-whisper-tests

# Run all hardware tests (auto stops/restarts hypha-whisper service)
pytest tests/test_hardware_loopback.py -m hardware -v

# Run a specific test
pytest tests/test_hardware_loopback.py -m hardware -k wer
pytest tests/test_hardware_loopback.py -m hardware -k "rms or playback"
```

The `suspend_service` pytest fixture automatically stops `hypha-whisper` at the start of the test session (to release the mic) and restarts it when done. It runs a background keeper thread that re-stops the service every 3 s to prevent `Restart=always` from reclaiming the microphone mid-test.

| Test | What it checks |
|------|---------------|
| `test_speaker_playback_only` | Speaker plays without error (USB SoundBar or HDMI/DP) |
| `test_mic_capture_rms` | ReSpeaker picks up speaker audio (RMS > 0.001) |
| `test_acoustic_loopback_wer` | Full pipeline WER < 30% against reference transcript |
| `test_speaker_identification` | Left vs right channel → 2 distinct angle labels |
| `test_speaker_stability_under_variation` | Same direction, varied audio → same angle label |

---

## Use Cases

- Live conference captioning
- AI agent voice interfaces
- Edge speech processing nodes
- Lab automation voice control

---

## Contributors

<a href="https://github.com/aicell-lab/hypha-whisper-node/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=aicell-lab/hypha-whisper-node" />
</a>

*Including AI assistants — [Claude](https://claude.ai) and [Kimi](https://kimi.moonshot.cn) contribute via co-authored commits.* 🤖

---

## License

MIT
