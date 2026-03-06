# hypha-whisper-node

Portable real-time speech-to-text node powered by Whisper and Jetson Orin Nano.
Captures speech via ReSpeaker mic array, transcribes on-device, and streams results through [Hypha RPC](https://pypi.org/project/hypha-rpc/).

---

## Hardware

| Component | Details |
|---|---|
| Compute | Jetson Orin Nano |
| Microphone | ReSpeaker Mic Array v2.0 (USB) |
| Power | USB-C PD power bank |
| Enclosure | 3D printed shell (ventilation + antenna ports) |

**Device stack:** ReSpeaker (top) → Jetson Orin Nano (bottom) inside 3D printed shell.

---

## Features

- On-device Whisper transcription (GPU, works offline)
- Real-time transcript streaming via Hypha ASGI service (SSE)
- Voice Activity Detection (webrtcvad) — silence is ignored
- Auto-reconnect to Hypha on network loss
- systemd service with watchdog and auto-restart

---

## Whisper Benchmarks (Jetson Orin Nano, 2 s audio, GPU)

| Model | Avg latency | Load time |
|-------|------------|-----------|
| tiny.en | 0.19 s | 6 s |
| **base.en** (default) | **0.40 s** | 4 s |
| small.en | 0.92 s | 26 s |

---

## Installation

```bash
git clone https://github.com/reef-imaging/hypha-whisper-node
cd hypha-whisper-node
sudo ./setup.sh
```

`setup.sh` installs system packages, PyTorch (NVIDIA JP6.1 wheel), all Python deps, and creates `/etc/hypha-whisper/config.env`.

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
2026-03-06T09:00:01 INFO [main] Initialising microphone capture...
2026-03-06T09:00:05 INFO [hypha] Connected to https://hypha.aicell.io (workspace: reef-imaging)
2026-03-06T09:00:05 INFO [hypha] ASGI service 'hypha-whisper' registered.
2026-03-06T09:00:15 INFO [transcript] Hello, how are you today
```

If the Hypha server drops, the service reconnects automatically (exponential backoff, max 60 s).

---

## Endpoints

Once running, the service exposes two endpoints via Hypha:

| Endpoint | Description |
|----------|-------------|
| `GET /transcript_feed` | SSE stream — one `data: <text>` event per utterance |
| `GET /health` | JSON: `{"status":"ok","model":"base.en","uptime_seconds":123}` |

Full URL pattern:
```
https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/transcript_feed
https://<HYPHA_SERVER>/<WORKSPACE>/apps/hypha-whisper/health
```

---

## Run manually (without systemd)

```bash
python3 main.py \
  --server https://hypha.aicell.io/ \
  --workspace my-workspace \
  --token my-token \
  --model base.en
```

Offline mode (transcribe to stdout, no Hypha):

```bash
python3 main.py --server ""
```

---

## Use Cases

- Live conference captioning
- AI agent voice interfaces
- Edge speech processing nodes
- Lab automation voice control

---

## License

MIT
