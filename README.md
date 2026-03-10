# hypha-whisper-node

Portable real-time speech-to-text node powered by Whisper and Jetson Orin Nano.
Captures speech via ReSpeaker 4 Mic Array, transcribes on-device using the LocalAgreement streaming algorithm, and streams results through [Hypha RPC](https://pypi.org/project/hypha-rpc/).

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

| Task | Command |
|------|---------|
| Start | `sudo systemctl start hypha-whisper` |
| Stop | `sudo systemctl stop hypha-whisper` |
| Restart | `sudo systemctl restart hypha-whisper` |
| Status | `systemctl status hypha-whisper` |
| Live logs | `journalctl -u hypha-whisper -f` |
| Last 100 lines | `journalctl -u hypha-whisper -n 100` |
| Disable autostart | `sudo systemctl disable hypha-whisper` |

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

Once running, the service exposes three endpoints via Hypha:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Live transcript viewer — open in any browser |
| `GET /transcript_feed` | SSE stream — one `data: <text>` event per committed phrase |
| `GET /health` | JSON: `{"status":"ok","model":"small.en","uptime_seconds":123}` |

Full URL pattern:
```
https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/
https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/transcript_feed
https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/health
```

### Live transcript viewer

Open `https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/` in a browser. The page connects automatically to `transcript_feed` via `EventSource`, accumulates text, and auto-scrolls. A **Clear** button resets the display. The connection indicator shows green when live and retries automatically on disconnect.

### Consuming the SSE stream programmatically

```python
import httpx

url = "https://hypha.aicell.io/reef-imaging/apps/hypha-whisper/transcript_feed"
with httpx.stream("GET", url) as r:
    for line in r.iter_lines():
        if line.startswith("data: "):
            print(line[6:])
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

Generates a TTS reference clip (via gTTS — natural Google-quality voice), plays it through the Dell AC511 speaker, records via ReSpeaker, transcribes, and measures Word Error Rate. The audio file is generated once and cached in `tests/fixtures/reference.wav`.

**Prerequisites:**

```bash
pip install gtts                          # natural TTS (requires internet on first run)
sudo apt-get install -y espeak-ng         # fallback TTS if gTTS unavailable
```

```bash
# One-time: allow passwordless sudo for service management
echo "reef-orinnano ALL=(ALL) NOPASSWD: /bin/systemctl start hypha-whisper, /bin/systemctl stop hypha-whisper" \
    | sudo tee /etc/sudoers.d/hypha-whisper-tests

# Run all hardware tests (auto stops/restarts service)
./scripts/run_hardware_tests.sh

# Run a specific test
./scripts/run_hardware_tests.sh -k wer
./scripts/run_hardware_tests.sh -k "rms or playback"
```

| Test | What it checks |
|------|---------------|
| `test_speaker_playback_only` | Dell AC511 plays without error |
| `test_mic_capture_rms` | ReSpeaker picks up speaker audio (RMS > 0.001) |
| `test_acoustic_loopback_wer` | Full pipeline WER < 40% against gTTS reference transcript |

---

## Use Cases

- Live conference captioning
- AI agent voice interfaces
- Edge speech processing nodes
- Lab automation voice control

---

## License

MIT
