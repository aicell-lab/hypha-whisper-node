# hypha-whisper-node

Portable real-time speech-to-text node powered by Whisper and Jetson Orin Nano.
Captures speech via ReSpeaker 4 Mic Array, transcribes on-device using the LocalAgreement streaming algorithm, and streams results through [Hypha RPC](https://pypi.org/project/hypha-rpc/).

**It's live! Give it a try:**

> [**🎙️ Open Live Transcript Viewer →**](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/) &nbsp;&nbsp; [**📡 SSE Stream**](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcript_feed) &nbsp;&nbsp; [**💚 Health**](https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/health)

---

## Hardware

| Component | Details |
|---|---|
| Compute | Jetson Orin Nano (JetPack 6.2, CUDA 12.6) |
| Microphone | ReSpeaker 4 Mic Array v2.0 (UAC1.0, 16 kHz, beamformed ch0) |
| Speaker (test) | Dell AC511 USB SoundBar |
| Power | USB-C PD power bank |
| Enclosure | 3D printed shell (ventilation + antenna ports) |

**Mic auto-detection:** ReSpeaker is tried first; falls back to HIK 1080P Camera if not found. Override with `--mic "name-substring"`.

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

## Whisper Benchmarks (Jetson Orin Nano, 2 s audio, GPU)

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
2026-03-09T12:55:54 INFO [transcript] hello, are you there?
```

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

> **Known limitation — speaker/angle attribution:** LocalAgreement (the streaming ASR algorithm) introduces 3–5 s commit latency. If a second speaker starts talking before the first speaker's text is committed, both speakers' audio overlap in the pending buffer and the DOA angle at commit time may reflect the second speaker rather than the first. The raw `angle` field is the most reliable signal; the `speaker` grouping label is best-effort only.

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
echo "reef-orinnano ALL=(ALL) NOPASSWD: /bin/systemctl start hypha-whisper, /bin/systemctl stop hypha-whisper" \
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
| `test_speaker_playback_only` | Dell AC511 plays without error |
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

## License

MIT
