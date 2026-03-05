# CLAUDE.md — hypha-whisper-node Action Plan

This file tracks the build plan. Mark tasks `[x]` as they are completed.

---

## Project Goal

Build a portable, real-time speech-to-text edge node that:
1. Captures audio via HIKVISION USB camera (built-in mic) — current hardware
2. Transcribes locally using Whisper (on Jetson Orin Nano)
3. Streams transcripts via Hypha RPC to remote agents/dashboards

---

## Action Plan

### Phase 1 — Environment Setup
- [x] Verify CUDA / TensorRT availability — CUDA 11.4, TensorRT 8.5.2 confirmed
- [x] Create Conda virtual environment and install base deps(with miniconda)
  - env: `hypha-whisper` (Python 3.8, miniconda)
  - PyTorch 2.1.0 from NVIDIA JetPack 5.1.2 wheel (CUDA on Orin GPU confirmed)
  - openai-whisper 20250625, SpeechRecognition 3.10.4, pyaudio, ffmpeg
  - Run with: `LD_LIBRARY_PATH=/home/reef-orinnano/miniconda3/envs/hypha-whisper/lib:$LD_LIBRARY_PATH`
- [x] Verify HIKVISION USB camera mic is detected — card 2 `hw:2,0`, 16kHz mono S16_LE

### Phase 2 — Audio Capture (HIKVISION USB mic)
- [x] Write `audio/capture.py` — MicCapture class, SpeechRecognition + PyAudio, Queue-based
- [x] Test HIKVISION mic device index — PyAudio index 20, hw:2,0, 16kHz mono, confirmed
- [x] Add VAD — webrtcvad 2.0.10, aggressiveness=2, 20ms frames, 30% voiced ratio threshold

### Phase 3 — Whisper Transcription
- [x] Benchmark Whisper model sizes on Jetson — results (2s audio, GPU):
  - tiny.en:  0.19s avg ✅  (load 6s)
  - base.en:  0.40s avg ✅  (load 4s)  ← default
  - small.en: 0.92s avg ✅  (load 26s)
- [x] Write `transcribe/whisper_engine.py` — WhisperEngine class, fp16=False, language="en"
- [x] Integrate openai-whisper — GPU confirmed, model on cuda:0
- [x] Latency target <2s — all model sizes PASS

### Phase 4 — Hypha RPC Integration
- [x] Install hypha-rpc 0.20.48 (already present)
- [x] Write `rpc/hypha_client.py` — HyphaClient class, protected visibility, exponential backoff reconnect
- [x] Expose `stream_transcripts()` async generator + `health()` via Hypha RPC
- [x] Connection to https://hypha.aicell.io/ confirmed ✅

### ⚠️ PENDING: JetPack Upgrade (do this before Phase 5)

**Current**: JetPack 5.1.3 (L4T R35.5) → Python 3.8 → hypha-rpc max 0.20.48
**Target**:  JetPack 6.1 or 6.2 (L4T R36) → Python 3.10 → hypha-rpc latest + CUDA 12

**Upgrade requires physical access — full OS reflash via NVIDIA SDK Manager:**
1. Host PC: install SDK Manager (`https://developer.nvidia.com/sdk-manager`)
2. Connect Jetson via USB-C data cable
3. Force Recovery mode: hold `FORCE REC` → press `RESET` → release `FORCE REC`
4. SDK Manager: select Jetson Orin Nano, JetPack 6.1 or 6.2, flash

**After reflash, reinstall env:**
```bash
conda create -n hypha-whisper python=3.10 -y
conda activate hypha-whisper
conda install -c conda-forge portaudio pyaudio ffmpeg -y
pip install https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/<wheel>.whl
pip install -r requirements.txt   # versions will need updating for py3.10
```
- New PyTorch wheel URL: `https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/`
- Remove `LD_LIBRARY_PATH` workaround — no longer needed on JP6
- All source code (audio/, transcribe/, rpc/) carries over unchanged

---

### Phase 5 — Main Orchestration
- [ ] Write `main.py` — wire audio capture → VAD → Whisper → Hypha RPC
- [ ] Add CLI args: `--server`, `--token`, `--model`, `--device`
- [ ] Add graceful shutdown (Ctrl+C, reconnect on disconnect)

### Phase 6 — Packaging & Deployment

#### 6a — Install & Config
- [ ] Write `requirements.txt` with pinned versions
- [ ] Write `setup.sh` for one-shot install on fresh Jetson
- [ ] Store secrets in `/etc/hypha-whisper/config.env` (not in code): `HYPHA_SERVER`, `HYPHA_TOKEN`
- [ ] Document USB wiring and enclosure assembly in `docs/hardware.md`

#### 6b — systemd Service (auto-start + auto-restart)
- [ ] Write `deploy/hypha-whisper.service` with:
  - `After=network-online.target` — wait for internet before starting
  - `Wants=network-online.target`
  - `Restart=always` — restart automatically on crash
  - `RestartSec=5` — wait 5 s before retry
  - `EnvironmentFile=/etc/hypha-whisper/config.env` — load secrets
  - `WatchdogSec=30` — systemd kills and restarts if process hangs
- [ ] Enable with `systemctl enable --now hypha-whisper`
- [ ] Verify `systemctl status hypha-whisper` and `journalctl -u hypha-whisper -f`

#### 6c — Application-level Resilience
- [ ] Implement reconnect loop in `rpc/hypha_client.py`:
  - On disconnect: wait, then reconnect with exponential backoff (1 s → 2 s → 4 s … max 60 s)
  - On internet loss: keep transcribing locally, buffer last N transcripts, flush when reconnected
- [ ] Implement watchdog heartbeat: call `systemd.daemon.notify("WATCHDOG=1")` every ~10 s so systemd knows the process is alive
- [ ] Add structured logging to stdout (captured by journald): timestamp, level, event

#### 6d — Network-awareness
- [ ] Use `After=network-online.target` + `systemd-networkd-wait-online` so the service only starts when the interface is actually up (not just configured)
- [ ] Handle DNS failures gracefully — retry Hypha connection, don't crash

### Phase 7 — Testing & Polish
- [ ] Write integration test: simulate audio file → verify Hypha transcript delivery
- [ ] Stress test: 30-min continuous run, monitor CPU/GPU/RAM
- [ ] Add local HDMI UI (optional) — display live transcript on screen
- [ ] Update README with actual benchmark numbers

### Phase 8 — ReSpeaker Mic Array Upgrade (future)
- [ ] Procure ReSpeaker Mic Array v2.0
- [ ] Install ReSpeaker USB driver
- [ ] Test beamforming / noise suppression vs. HIKVISION baseline
- [ ] Tune channel selection and sample rate (16 kHz mono, beamformed output)
- [ ] Update `audio/capture.py` with ReSpeaker device support (switchable via CLI flag)

---

## Key Files (target layout)

```
hypha-whisper-node/
  main.py               # entry point
  audio/
    capture.py          # mic capture + VAD
  transcribe/
    whisper_engine.py   # Whisper inference wrapper
  rpc/
    hypha_client.py     # Hypha RPC service
  docs/
    hardware.md         # wiring, assembly
  setup.sh              # one-shot install
  requirements.txt
  README.md
  CLAUDE.md             # this file
```

---

## Notes

- Target latency: < 2 s per utterance
- Whisper model default: `base.en` (adjust based on Jetson benchmarks)
- Hypha server: configurable via CLI; supports offline mode (transcribe only, no streaming)
- Current mic: HIKVISION USB camera built-in mic (mono, 16 kHz)
- Future mic: ReSpeaker Mic Array v2.0 (see Phase 8)
