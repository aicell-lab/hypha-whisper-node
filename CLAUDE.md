# CLAUDE.md — hypha-whisper-node Action Plan

This file tracks the build plan. Mark tasks `[x]` as they are completed.

---

## Project Goal

Build a portable, real-time speech-to-text edge node that:
1. Captures audio via ReSpeaker 4 Mic Array (beamformed, hardware noise suppression)
2. Transcribes locally using Whisper (on Jetson Orin Nano, GPU)
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
  - openai-whisper 20250625, whisper-timestamped, hypha-rpc 0.21.31 installed ✅
  - portaudio19-dev + ffmpeg still needed (`sudo apt-get install -y portaudio19-dev ffmpeg`) → then `pip install --user pyaudio`
  - CUDA confirmed: torch.cuda.is_available()=True, device=Orin ✅
  - Whisper base.en GPU latency ~0.4s ✅
  - **numpy pinned to 1.26.4** — whisper-timestamped upgrades to 2.x which breaks PyTorch ABI + system scipy

### Phase 2 — Audio Capture
- [x] Write `audio/capture.py` — MicCapture class, PyAudio, raw numpy float32 chunks
- [x] Auto-detect ReSpeaker (priority) then HIK camera mic by name substring
- [x] ReSpeaker 4 Mic Array: open 6-ch stream, extract ch0 (beamformed) via `raw[0::6]`
- [x] `--mic` CLI flag to override auto-detection

### Phase 3 — Whisper Transcription (streaming)
- [x] Benchmark Whisper model sizes on Jetson — results (2s audio, GPU):
  - tiny.en:  0.19s avg ✅  (load 6s)
  - base.en:  0.40s avg ✅  (load 4s)
  - small.en: 0.92s avg ✅  (load 26s) ← default
- [x] Vendor `whisper_online.py` + `silero_vad_iterator.py` from ufal/whisper_streaming (commit 6da90b44)
- [x] Write `transcribe/streaming_engine.py` — StreamingEngine wrapping OnlineASRProcessor (LocalAgreement)
  - Backend: `whisper-timestamped` (uses PyTorch, CUDA on Jetson) — default
  - Backend: `faster-whisper` (CTranslate2, CPU only on Jetson — no CUDA build available via pip)
  - LocalAgreement commits text only when two consecutive passes agree → no word-boundary errors
  - ~3–5 s commit latency (trade-off for accuracy)
- [x] Latency target <2s per chunk inference — PASS (separate from commit latency)

### Phase 4 — Hypha RPC Integration
- [x] Install hypha-rpc 0.21.31 (latest, installed fresh on JP6.2)
- [x] Write `rpc/hypha_client.py` — HyphaClient class, public visibility, exponential backoff reconnect
- [x] Expose `GET /transcript_feed` (SSE) + `GET /health` (JSON) via Hypha **ASGI** service (`type="asgi"`)
  - SSE generator: `except (asyncio.TimeoutError, queue.Empty)` → yield keep-alive (prevents crash)
  - On connect: `engine.init_session()` resets LocalAgreement state
  - On disconnect: `engine.finish_session()` flushes remaining audio
- [x] Connection to https://hypha.aicell.io/ confirmed ✅
- [x] Fix `KeyError: 'Service not found'` keepalive false reconnects — treat KeyError as non-fatal

---

### Phase 5 — Main Orchestration
- [x] Write `main.py` — wire audio capture → StreamingEngine → Hypha RPC
- [x] Add CLI args: `--server`, `--token`, `--model`, `--backend`, `--device`, `--mic`
- [x] Add graceful shutdown (Ctrl+C / SIGTERM → mic.stop() + task cancel)

### Phase 6 — Packaging & Deployment

#### 6a — Install & Config
- [x] Write `requirements.txt` with pinned versions (numpy==1.26.4 critical)
- [x] Write `setup.sh` for one-shot install on fresh Jetson
- [x] Store secrets in `/etc/hypha-whisper/config.env` (not in code): `HYPHA_SERVER`, `HYPHA_TOKEN`
- [x] Document USB wiring and enclosure assembly in `docs/hardware.md`

#### 6b — systemd Service (auto-start + auto-restart)
- [x] Write `deploy/hypha-whisper.service` with:
  - `After=network-online.target` — wait for internet before starting
  - `Restart=always` — restart automatically on crash
  - `RestartSec=5` — wait 5 s before retry
  - `EnvironmentFile=/etc/hypha-whisper/config.env` — load secrets
  - `WatchdogSec=180` — 3-min window to survive model loading (~4s base.en, up to 26s small.en)
  - `TimeoutStartSec=300` — 5-min start timeout
  - `ExecStart=... --backend whisper-timestamped`

#### 6c — Application-level Resilience
- [x] Implement reconnect loop in `rpc/hypha_client.py` — exponential backoff (1s → 2s → 4s … max 60s)
- [x] Implement watchdog heartbeat: `_sd_notify("WATCHDOG=1")` every ~10 s; `READY=1` on service start
- [x] Structured logging to stdout (captured by journald)

#### 6d — Network-awareness
- [x] `After=network-online.target` in systemd unit
- [x] Exponential backoff covers DNS failures

### Phase 7 — Testing & Polish

#### 7a — pytest suite
- [x] `pytest.ini` + `requirements-dev.txt` (pytest-asyncio, httpx, psutil)
- [x] `tests/conftest.py` — shared fixtures: MockMicCapture, MockStreamingEngine, `suspend_service`
  - `suspend_service` fixture: stops hypha-whisper service before hardware tests, restores after
  - Requires passwordless sudo or skips automatically with instructions
- [x] `tests/test_streaming_engine.py` — StreamingEngine unit + `@hardware` GPU tests
- [x] `tests/test_hypha.py` — ASGI `/transcript_feed` + `/health` with mock engine; `@integration` live Hypha test
- [x] `tests/test_stress.py` — `@slow @hardware` 30-min run; log CPU/GPU/RAM via psutil + tegrastats
- [x] `tests/test_hardware_loopback.py` — `@hardware` acoustic loopback tests (speaker → mic → WER)

#### 7b — GitHub Actions CI
- [x] `.github/workflows/test.yml` — runs on push/PR to `main`
  - standard `ubuntu-latest` runner: install deps (CPU-only torch), run unit tests only
  - excludes `@hardware`, `@integration`, `@slow` markers

#### 7c — Polish
- [x] Update README with actual benchmark numbers and hardware
- [x] Switch to whisper_streaming LocalAgreement — eliminates word-boundary errors
- [x] `whisper-timestamped` backend uses PyTorch CUDA on Jetson (faster-whisper CTranslate2 has no CUDA pip wheel for aarch64)
- [x] Add `GET /` live transcript viewer (HTML + SSE `EventSource`, browser-ready)
- [x] Default model updated to `small.en`

### Phase 8 — ReSpeaker Mic Array Upgrade
- [x] Procure ReSpeaker Mic Array v2.0
- [x] Install ReSpeaker USB driver — UAC1.0 class-compliant, no custom driver needed on Linux
- [x] Test beamforming / noise suppression vs. HIKVISION baseline (use `test_hardware_loopback.py`)
- [x] Tune channel selection and sample rate — 16 kHz native, ch0 = beamformed output
- [x] Update `audio/capture.py` with ReSpeaker device support (auto-detect; switchable via `--mic` CLI flag)

### Phase 9 — Hardware Testing Infrastructure
- [x] `tests/test_hardware_loopback.py` — play audio through Dell AC511 → record via ReSpeaker → WER check
  - `test_speaker_playback_only` — smoke test speaker
  - `test_mic_capture_rms` — verify ReSpeaker picks up audio (RMS threshold)
  - `test_acoustic_loopback_wer` — full pipeline WER < 40% vs gTTS reference transcript
- [x] TTS audio: generated via gTTS (Google TTS, natural voice) on first run; cached as `tests/fixtures/reference.wav`; falls back to espeak-ng if gTTS unavailable
  - Install: `pip install gtts` (internet required on first run); fallback: `sudo apt-get install -y espeak-ng`
- [x] `scripts/run_hardware_tests.sh` — wrapper script: generate audio if missing, stop/restore service, check hardware, run pytest
- [x] `suspend_service` pytest fixture — auto stop/restart hypha-whisper around hardware tests
  - Passwordless sudo: `echo "USER ALL=(ALL) NOPASSWD: /bin/systemctl start hypha-whisper, /bin/systemctl stop hypha-whisper" | sudo tee /etc/sudoers.d/hypha-whisper-tests`

---

## Key Files

```
hypha-whisper-node/
  main.py                        # entry point; CLI args including --mic, --backend
  audio/
    capture.py                   # MicCapture: ReSpeaker (ch0 beamformed) or HIK fallback
  transcribe/
    streaming_engine.py          # StreamingEngine wrapping OnlineASRProcessor
    whisper_online.py            # vendored from ufal/whisper_streaming (do not modify)
  silero_vad_iterator.py         # vendored alongside whisper_online.py (project root)
  rpc/
    hypha_client.py              # Hypha ASGI service (FastAPI SSE)
  tests/
    conftest.py                  # fixtures: mocks + suspend_service
    test_streaming_engine.py     # StreamingEngine unit + hardware tests
    test_hypha.py                # ASGI endpoint + Hypha integration tests
    test_stress.py               # 30-min stress test (slow + hardware)
    test_hardware_loopback.py    # acoustic loopback + WER tests
    fixtures/
      reference.wav              # gTTS reference audio (gitignored, auto-generated)
  scripts/
    run_hardware_tests.sh        # hardware test runner with service management
  .github/
    workflows/
      test.yml                   # CI: unit tests on ubuntu-latest
  deploy/
    hypha-whisper.service        # systemd unit (copy to /etc/systemd/system/)
  docs/
    hardware.md                  # wiring, assembly
  setup.sh                       # one-shot install
  pytest.ini
  requirements.txt               # numpy==1.26.4 pinned
  requirements-dev.txt
  README.md
  CLAUDE.md                      # this file
```

---

## Working Convention

- **sudo commands**: If a `sudo` command is needed during implementation, STOP and ask the user to run it — do not attempt to run it yourself or assume it was already done.

---

## Notes

- Target latency: ~3–5 s commit latency (LocalAgreement; trade-off for accuracy)
- Whisper model default: `small.en`
- Backend default: `whisper-timestamped` (PyTorch CUDA); `faster-whisper` runs CPU-only on Jetson
- Hypha server: configurable via CLI; supports offline mode (transcribe only, no streaming)
- Current mic: ReSpeaker 4 Mic Array v2.0 (6-ch UAC1.0, 16 kHz, beamformed ch0)
- numpy must stay at 1.26.4 — whisper-timestamped upgrades it to 2.x breaking PyTorch ABI
- `sudo python3` on Jetson has no pip and no user site-packages; run sudo scripts with user packages via:
  `sudo PYTHONPATH=/home/reef-orinnano/.local/lib/python3.10/site-packages python3 <script.py>`
- **ReSpeaker DOA**: USB hardware DOA (firmware v4.0) broken on Linux/Jetson; firmware downgrade confirmed same v4.0. Fixed via **software GCC-PHAT DOA** (`_SoftwareDOAProcessor` in `audio/doa_reader.py`, pyroomacoustics SRP-PHAT on ch1–4 raw mics). Requires `pip install --user pyroomacoustics`. Wired: `MicCapture.multichannel_queue` → `StreamingEngine(multichannel_queue=...)` → `DOAReader`.
