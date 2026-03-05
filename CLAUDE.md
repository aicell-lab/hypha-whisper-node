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
- [x] Verify CUDA availability — **JetPack 6.2 (L4T R36.5), CUDA 12.6, Python 3.10.12** confirmed
- [x] Install Python deps for new environment (python3-pip via get-pip.py; portaudio19-dev, ffmpeg via apt; PyTorch from NVIDIA JP6.1 wheel)
  - pip installed to `~/.local/bin` via `python3 get-pip.py --user`
  - PyTorch 2.5.0 NVIDIA wheel from `https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/` (JP6.2 wheel not yet published; JP6.1 wheel works on CUDA 12.6)
  - `libcusparseLt.so.0` missing on JP6.2 — extracted from NVIDIA CUDA sbsa repo `.deb` into `~/.local/lib/`
  - `LD_LIBRARY_PATH="$HOME/.local/lib:/usr/local/cuda/lib64"` added to `~/.bashrc`
  - openai-whisper 20250625, SpeechRecognition 3.10.4, webrtcvad 2.0.10, hypha-rpc 0.21.31 installed ✅
  - portaudio19-dev + ffmpeg still needed (`sudo apt-get install -y portaudio19-dev ffmpeg`) → then `pip install --user pyaudio`
  - CUDA confirmed: torch.cuda.is_available()=True, device=Orin ✅
  - Whisper base.en GPU latency ~3.5s first run (JIT warm-up), ~0.4s subsequent ✅
- [x] Verify HIKVISION USB camera mic is detected — **card 0** `hw:0,0` on new OS

### Phase 2 — Audio Capture (HIKVISION USB mic)
- [x] Write `audio/capture.py` — MicCapture class, SpeechRecognition + PyAudio, Queue-based
- [ ] Test HIKVISION mic device index on new OS — was PyAudio index 20 hw:2,0; now card 0 hw:0,0 (re-verify PyAudio index)
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
- [x] Install hypha-rpc 0.21.31 (latest, installed fresh on JP6.2)
- [x] Write `rpc/hypha_client.py` — HyphaClient class, public visibility, exponential backoff reconnect
- [x] Expose `GET /transcript_feed` (SSE) + `GET /health` (JSON) via Hypha **ASGI** service (`type="asgi"`)
  - Replaced RPC `stream_transcripts()` with FastAPI StreamingResponse (`text/event-stream`)
  - On disconnect: mic queue drained so next session starts clean
- [x] Connection to https://hypha.aicell.io/ confirmed ✅

---

### Phase 5 — Main Orchestration
- [x] Write `main.py` — wire audio capture → VAD → Whisper → Hypha RPC
- [x] Add CLI args: `--server`, `--token`, `--model`, `--device`
- [x] Add graceful shutdown (Ctrl+C / SIGTERM → mic.stop() + task cancel)

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

#### 7a — pytest suite
- [x] `pytest.ini` + `requirements-dev.txt` (pytest-asyncio, httpx, psutil)
- [x] `tests/conftest.py` — shared fixtures: synthetic PCM, MockMicCapture, MockWhisperEngine
- [x] `tests/test_vad.py` — VAD unit tests (silence rejected, voiced frames accepted); no hardware
- [x] `tests/test_whisper.py` — WhisperEngine: silent-audio returns empty string; `@hardware` GPU test
- [x] `tests/test_hypha.py` — ASGI `/transcript_feed` + `/health` with mock mic/whisper; `@integration` live Hypha test
- [x] `tests/test_stress.py` — `@slow @hardware` 30-min run; log CPU/GPU/RAM via psutil + tegrastats

#### 7b — GitHub Actions CI
- [x] `.github/workflows/test.yml` — runs on push/PR to `main`
  - standard `ubuntu-latest` runner: install deps (CPU-only torch), run unit tests only
  - excludes `@hardware`, `@integration`, `@slow` markers

#### 7c — Polish
- [ ] Update README with actual benchmark numbers
- [ ] Add local HDMI UI (optional) — display live transcript on screen

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
    hypha_client.py     # Hypha ASGI service (FastAPI SSE)
  tests/
    conftest.py         # shared fixtures
    test_vad.py         # VAD unit tests
    test_whisper.py     # Whisper unit + hardware tests
    test_hypha.py       # ASGI endpoint + Hypha integration tests
    test_stress.py      # 30-min stress test (slow + hardware)
  .github/
    workflows/
      test.yml          # CI: unit tests on ubuntu-latest
  docs/
    hardware.md         # wiring, assembly
  setup.sh              # one-shot install
  pytest.ini
  requirements.txt
  requirements-dev.txt
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
