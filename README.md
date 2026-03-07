# hypha-whisper-node

Portable real-time speech-to-text node powered by Whisper and Jetson Orin Nano.
Captures speech via HIKVISION USB camera mic, transcribes on-device, and streams results through [Hypha RPC](https://pypi.org/project/hypha-rpc/).

---

## Hardware

| Component | Details |
|---|---|
| Compute | Jetson Orin Nano (JetPack 6.2, CUDA 12.6) |
| Microphone | HIKVISION USB camera built-in mic (16 kHz mono) |
| Power | USB-C PD power bank |
| Enclosure | 3D printed shell (ventilation + antenna ports) |

**Device stack:** HIKVISION USB camera → Jetson Orin Nano inside 3D printed shell.

---

## Features

- On-device Whisper transcription (GPU, works offline)
- Real-time transcript streaming via Hypha ASGI service (SSE)
- Live transcript viewer — browser-based HTML page at `/`
- Two-stage noise rejection: webrtcvad VAD + bandpass filter (300–3400 Hz) + RMS normalisation
- Hallucination suppression: `condition_on_previous_text=False` + post-processing regex
- Auto-reconnect to Hypha on network loss (exponential backoff)
- systemd service with watchdog and auto-restart

---

## Whisper Benchmarks (Jetson Orin Nano, 2 s audio, GPU)

| Model | Avg latency | Load time |
|-------|------------|-----------|
| tiny.en | 0.19 s | 6 s |
| base.en | 0.40 s | 4 s |
| **small.en** (default) | **0.92 s** | 26 s |

`small.en` is the default — it offers significantly better accuracy for natural speech with acceptable latency. Use `--model base.en` if lower latency is required.

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

Once running, the service exposes three endpoints via Hypha:

| Endpoint | Description |
|----------|-------------|
| `GET /` | Live transcript viewer — open in any browser |
| `GET /transcript_feed` | SSE stream — one `data: <text>` event per utterance |
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
  --model small.en
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
