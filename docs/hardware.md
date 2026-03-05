# Hardware Guide — Hypha Whisper Node

## Current Hardware

| Component | Model | Notes |
|-----------|-------|-------|
| Compute | NVIDIA Jetson Orin Nano (8 GB) | JetPack 6.2, L4T R36.5, CUDA 12.6 |
| Microphone | HIKVISION USB camera (built-in mic) | USB 2.0, mono, 16 kHz |
| Storage | microSD or NVMe (recommended) | NVMe gives faster model load |
| Power | 19 V DC barrel jack | Jetson Orin Nano dev kit PSU |

---

## USB Audio — HIKVISION Camera

### Physical Connection

1. Connect the HIKVISION camera USB cable to any USB-A port on the Jetson.
2. The camera enumerates two USB interfaces: video (UVC) and audio (UAC).
3. No additional drivers are required — the kernel UAC driver handles it.

### Verify Detection

```bash
# List ALSA capture devices
arecord -l
# Expected: card 0: HD-Camera [...], device 0: USB Audio [USB Audio]

# Quick mic test — record 3 s, play back
arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 -d 3 /tmp/test.wav
aplay /tmp/test.wav
```

### PyAudio Device Index

PyAudio assigns its own device indices independently of ALSA card numbers.
Run this to find the correct index:

```python
import pyaudio
pa = pyaudio.PyAudio()
for i in range(pa.get_device_count()):
    info = pa.get_device_info_by_index(i)
    if info["maxInputChannels"] > 0:
        print(i, info["name"])
```

Look for a line containing `"HD-Camera"` or `"USB Audio"` — that index is the
one to pass to `MicCapture(device_index=N)`.

On a fresh JetPack 6.2 install the HIKVISION camera appeared at **ALSA card 0**
(`hw:0,0`). The PyAudio index may differ; re-run the snippet above after every
OS reinstall.

---

## Enclosure & Wiring

### Minimal Bench Setup

```
[HIKVISION USB Camera]
        |
        | USB 2.0 cable
        |
[Jetson Orin Nano dev kit]
        |
        | Ethernet or Wi-Fi
        |
[Network / Internet] ──► Hypha server (https://hypha.aicell.io/)
```

### Recommended Deployment

For a permanent edge deployment:

1. **Enclosure** — mount the Jetson in a ventilated enclosure (e.g. Waveshare
   metal case). Allow ≥ 5 cm clearance on the heatsink side.
2. **Camera position** — point the HIKVISION camera at the subject area.
   Mount at head height for best direct-speech capture.
3. **USB cable** — use a cable ≤ 3 m to avoid USB 2.0 signal degradation.
   If a longer run is needed, use a powered USB hub.
4. **Power** — use the official 19 V / 4 A PSU. Underpowering the Jetson
   causes throttling that increases Whisper latency.
5. **Network** — wired Ethernet is strongly preferred. Wi-Fi introduces
   variable latency on the Hypha streaming connection.

---

## Future: ReSpeaker Mic Array v2.0

See [Phase 8 in CLAUDE.md](../CLAUDE.md) for the upgrade plan.

The ReSpeaker Mic Array v2.0 provides:
- 4-mic circular array with beamforming
- On-chip noise suppression (XMOS XVF-3000)
- USB UAC interface (no custom driver required on Linux)
- 16 kHz mono beamformed output — drop-in replacement for the HIKVISION mic

When the array is available, update `audio/capture.py` to accept a
`--device respeaker` flag that selects the beamformed channel.
