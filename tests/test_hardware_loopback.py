"""
tests/test_hardware_loopback.py — End-to-end acoustic loopback tests.

Audio source:
  tests/test-audio-male.wav   — voice reading the reference text

Tests:
  test_speaker_playback_only     — smoke test: play audio through speaker bar
  test_mic_capture_rms           — verify ReSpeaker picks up audio (RMS threshold)
  test_acoustic_loopback_wer     — full pipeline WER < threshold
  test_speaker_identification    — play from LEFT then RIGHT; assert 2 distinct
                                   speakers identified (DOA angle difference)
  test_speaker_stability_under_variation
                                 — audio-processed variants from the same
                                   direction must map to the SAME speaker

Run with:
    pytest tests/test_hardware_loopback.py -v -s -m hardware
"""

import queue
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pyaudio
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIXTURES_DIR   = Path(__file__).parent / "fixtures"
MALE_WAV       = Path(__file__).parent / "test-audio-male.wav"

# Text spoken in the audio file
REFERENCE_TEXT = (
    "In microscopy laboratories, researchers often need to annotate experiments, "
    "describe observations, or record notes while working at the microscope, which "
    "can interrupt the workflow if done manually. A portable real-time speech-to-text "
    "device can capture spoken descriptions and automatically convert them into text "
    "during imaging sessions."
)

REFERENCE = " ".join(
    re.sub(r"[^\w\s']", " ", REFERENCE_TEXT.lower()).split()  # hyphens → spaces
)

# Speaker detection fallback list (in priority order)
# 1. Dell AC511 USB SoundBar (original hardware)
# 2. HDMI/DisplayPort audio (monitor speakers)
# 3. Generic ALSA outputs
SPEAKER_CANDIDATES = [
    "Dell AC511",
    "HDMI",
    "DisplayPort", 
    "alsa_output.pci",
]
MIC_NAME = "ReSpeaker"   # substring match for input device

SPEAKER_RATE     = 44100
SPEAKER_CHANNELS = 2
POST_PLAYBACK_WAIT   = 45    # seconds to flush VAC/LocalAgreement buffer after playback
WER_PASS_THRESHOLD   = 0.30  # Never change this! Otherwise this is cheating and we can't trust the test!

# Speaker identification test settings
SEGMENT_POST_WAIT     = 6    # seconds to flush each clip after playback
MIN_DISTINCT_SPEAKERS = 2    # left vs right channel must give distinct DOA angles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_wav(path: Path) -> None:
    """Skip test if audio file is missing."""
    if not path.exists():
        pytest.skip(f"Audio file not found: {path}. Place it in the tests/ directory.")


def _find_output_device(name_sub: str = None) -> tuple:
    """Return PyAudio device index and name for an output device.
    
    If name_sub is provided, searches for that specific device.
    Otherwise tries SPEAKER_CANDIDATES in order and returns the first match.
    
    Returns:
        tuple: (device_index, device_name)
    """
    pa = pyaudio.PyAudio()
    try:
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxOutputChannels", 0) > 0:
                devices.append((i, info.get("name", "")))
        
        # Search candidates in priority order
        candidates = [name_sub] if name_sub else SPEAKER_CANDIDATES
        for candidate in candidates:
            if not candidate:
                continue
            for idx, name in devices:
                if candidate in name:
                    return idx, name
    finally:
        pa.terminate()
    
    available = [name for _, name in devices]
    raise RuntimeError(
        f"No output device found. Tried: {candidates}. "
        f"Available: {available}"
    )


def _decode_audio_to_pcm(path: Path, rate: int, channels: int) -> np.ndarray:
    """Use ffmpeg to decode any audio file to interleaved int16 PCM."""
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-i", str(path),
        "-f", "s16le",
        "-ac", str(channels),
        "-ar", str(rate),
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.int16)


def _decode_audio_to_float32_mono(path: Path, rate: int = 44100) -> np.ndarray:
    """Decode audio file to float32 mono PCM (for sounddevice channel routing)."""
    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(path),
        "-f", "f32le", "-ac", "1", "-ar", str(rate), "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def _word_error_rate(ref: str, hyp: str) -> float:
    """Compute WER: edit_distance(ref_words, hyp_words) / len(ref_words)."""
    def normalise(s):
        s = s.lower()
        s = re.sub(r"[^\w\s']", " ", s)  # hyphens/punct → spaces
        return s.split()

    r = normalise(ref)
    h = normalise(hyp)
    if not r:
        return 0.0 if not h else 1.0

    d = list(range(len(h) + 1))
    for i, rw in enumerate(r):
        prev = d[:]
        d[0] = i + 1
        for j, hw in enumerate(h):
            d[j + 1] = min(prev[j] + (0 if rw == hw else 1),
                           d[j] + 1,
                           prev[j + 1] + 1)
    return d[len(h)] / len(r)


def _find_sd_output_device(name_sub: str = None) -> tuple:
    """Return sounddevice device index and name for an output device.
    
    If name_sub is provided, searches for that specific device.
    Otherwise tries SPEAKER_CANDIDATES in order and returns the first match.
    
    Returns:
        tuple: (device_index, device_name)
    """
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        
        # Search candidates in priority order
        candidates = [name_sub] if name_sub else SPEAKER_CANDIDATES
        for candidate in candidates:
            if not candidate:
                continue
            for i, d in enumerate(devs):
                if candidate in d["name"] and d["max_output_channels"] > 0:
                    return i, d["name"]
    except Exception as exc:
        raise RuntimeError(f"sounddevice not available: {exc}")
    
    available = [d["name"] for d in devs if d["max_output_channels"] > 0]
    raise RuntimeError(
        f"sounddevice output device not found. Tried: {candidates}. "
        f"Available: {available}"
    )


def _play_mono_to_channel_sd(wav_path: Path, channel: int, device_idx: int,
                              rate: int = 44100) -> None:
    """Play a mono WAV into one stereo channel of the speaker via sounddevice.

    channel: 0 = left, 1 = right (the other channel is silence).
    Blocks until playback is complete.
    """
    import sounddevice as sd

    mono = _decode_audio_to_float32_mono(wav_path, rate)
    stereo = np.zeros((len(mono), 2), dtype=np.float32)
    stereo[:, channel] = mono
    sd.play(stereo, samplerate=rate, device=device_idx, blocking=True)


def _extract_text(item) -> str:
    if isinstance(item, dict):
        return item.get("text", "")
    return str(item)


def _extract_speaker(item) -> str:
    if isinstance(item, dict):
        return item.get("speaker", "?")
    return "?"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_speaker_playback_only():
    """Smoke test: verify speaker plays audio without error."""
    _require_wav(MALE_WAV)
    speaker_idx, speaker_name = _find_output_device()
    pcm = _decode_audio_to_pcm(MALE_WAV, SPEAKER_RATE, SPEAKER_CHANNELS)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=SPEAKER_CHANNELS,
        rate=SPEAKER_RATE,
        output=True,
        output_device_index=speaker_idx,
        frames_per_buffer=4096,
    )
    try:
        chunk_size = 4096 * SPEAKER_CHANNELS
        for offset in range(0, len(pcm), chunk_size):
            stream.write(pcm[offset: offset + chunk_size].tobytes())
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

    print(f"\n[test] Played {len(pcm)/SPEAKER_CHANNELS/SPEAKER_RATE:.1f}s audio through {speaker_name}")


@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_mic_capture_rms():
    """Verify ReSpeaker captures non-silent audio during playback."""
    _require_wav(MALE_WAV)
    speaker_idx, speaker_name = _find_output_device()

    from audio.capture import MicCapture
    mic = MicCapture(preferred_mic=MIC_NAME)
    mic.start()

    pcm = _decode_audio_to_pcm(MALE_WAV, SPEAKER_RATE, SPEAKER_CHANNELS)

    playback_done = threading.Event()

    def _play():
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=SPEAKER_CHANNELS,
            rate=SPEAKER_RATE,
            output=True,
            output_device_index=speaker_idx,
            frames_per_buffer=4096,
        )
        chunk_size = 4096 * SPEAKER_CHANNELS
        for offset in range(0, len(pcm), chunk_size):
            stream.write(pcm[offset: offset + chunk_size].tobytes())
        stream.stop_stream(); stream.close(); pa.terminate()
        playback_done.set()

    threading.Thread(target=_play, daemon=True).start()

    rms_values = []
    deadline = time.monotonic() + len(pcm) / SPEAKER_CHANNELS / SPEAKER_RATE + 2
    while time.monotonic() < deadline:
        try:
            chunk = mic.raw_audio_queue.get(timeout=0.2)
            rms = float(np.sqrt(np.mean(chunk ** 2)))
            rms_values.append(rms)
        except queue.Empty:
            pass

    mic.stop()

    avg_rms = sum(rms_values) / len(rms_values) if rms_values else 0
    print(f"\n[test] Captured {len(rms_values)} chunks, avg RMS = {avg_rms:.4f}")
    print(f"[test] Speaker used: {speaker_name}")

    assert avg_rms > 0.001, (
        f"RMS too low ({avg_rms:.5f}) — mic may not be picking up speaker audio. "
        "Check volume and physical proximity."
    )


@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_acoustic_loopback_wer():
    """
    Play audio through Dell AC511, capture via ReSpeaker, transcribe via
    StreamingEngine, and assert WER < threshold.
    """
    _require_wav(MALE_WAV)

    from audio.capture import MicCapture
    from transcribe.streaming_engine import StreamingEngine

    speaker_idx, speaker_name = _find_output_device()
    print(f"\n[test] Speaker: {speaker_name} at index {speaker_idx}")

    pcm = _decode_audio_to_pcm(MALE_WAV, SPEAKER_RATE, SPEAKER_CHANNELS)
    audio_duration = len(pcm) / SPEAKER_CHANNELS / SPEAKER_RATE
    print(f"[test] Audio duration: {audio_duration:.1f}s")

    mic = MicCapture(preferred_mic=MIC_NAME)
    engine = StreamingEngine(model_name="small.en", use_vac=True,
                              enable_doa=False, enable_speaker_id=False)
    engine.init_session()
    mic.start()

    playback_done = threading.Event()

    def _play():
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=SPEAKER_CHANNELS,
            rate=SPEAKER_RATE,
            output=True,
            output_device_index=speaker_idx,
            frames_per_buffer=4096,
        )
        try:
            chunk_size = 4096 * SPEAKER_CHANNELS
            for offset in range(0, len(pcm), chunk_size):
                stream.write(pcm[offset: offset + chunk_size].tobytes())
        finally:
            stream.stop_stream()
            stream.close()
            pa.terminate()
            playback_done.set()

    playback_thread = threading.Thread(target=_play, daemon=True)
    playback_thread.start()
    print("[test] Playback started...")

    transcripts = []

    def _drain_text():
        while not engine.text_queue.empty():
            item = engine.text_queue.get_nowait()
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            if text.strip():
                print(f"[transcript] {text}")
                transcripts.append(text)

    try:
        while not playback_done.is_set():
            try:
                chunk = mic.raw_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            engine.process_audio(chunk)
            _drain_text()

        print("[test] Playback complete — flushing buffer...")

        flush_deadline = time.monotonic() + POST_PLAYBACK_WAIT
        while time.monotonic() < flush_deadline:
            try:
                chunk = mic.raw_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            engine.process_audio(chunk)
            _drain_text()

    finally:
        engine.finish_session()
        _drain_text()
        mic.stop()

    playback_thread.join(timeout=5)

    full_hypothesis = " ".join(transcripts)
    wer = _word_error_rate(REFERENCE, full_hypothesis)

    print("\n--- Loopback test results ---")
    print(f"  Reference : {REFERENCE[:80]}...")
    print(f"  Hypothesis: {full_hypothesis[:80]}...")
    print(f"  WER       : {wer:.1%}  (threshold: {WER_PASS_THRESHOLD:.0%})")

    assert transcripts, "No transcript produced — check speaker volume and mic position"
    assert wer < WER_PASS_THRESHOLD, (
        f"WER {wer:.1%} exceeds {WER_PASS_THRESHOLD:.0%} threshold.\n"
        f"  Ref: {REFERENCE}\n"
        f"  Hyp: {full_hypothesis}"
    )


# ---------------------------------------------------------------------------
# Audio transform helpers (used by stability test)
# ---------------------------------------------------------------------------

STABILITY_TRANSFORMS = [
    ("noise_20db",   "white noise SNR 20 dB (light)"),
    ("noise_10db",   "white noise SNR 10 dB (heavy)"),
    ("speed_fast",   "10% faster speaking pace (atempo=1.10)"),
    ("speed_slow",   "10% slower speaking pace (atempo=0.90)"),
    ("reverb",       "room echo simulation"),
    ("volume_low",   "50% volume — quieter speaker"),
    ("volume_high",  "150% volume — louder speaker"),
]

# Allow this fraction of items per variant to be mis-attributed (accounts for
# initial utterances before DOA history builds up).
STABILITY_WRONG_FRACTION_LIMIT = 0.35

STABILITY_POST_WAIT = 5


def _apply_transform(path: Path, transform: str, rate: int = 44100) -> np.ndarray:
    """Return float32 mono array at *rate* Hz with audio transform applied."""
    if transform in ("noise_20db", "noise_10db"):
        audio = _decode_audio_to_float32_mono(path, rate)
        snr_db = 20 if transform == "noise_20db" else 10
        snr_linear = 10 ** (snr_db / 20)
        signal_rms = float(np.sqrt(np.mean(audio ** 2))) + 1e-9
        noise_rms = signal_rms / snr_linear
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_rms, len(audio)).astype(np.float32)
        return np.clip(audio + noise, -1.0, 1.0)

    af_filters = {
        "speed_fast":  "atempo=1.10",
        "speed_slow":  "atempo=0.90",
        "reverb":      "aecho=0.8:0.88:60:0.4",
        "volume_low":  "volume=0.5",
        "volume_high": "volume=1.5",
    }
    if transform not in af_filters:
        raise ValueError(f"Unknown transform: {transform}")

    cmd = [
        "ffmpeg", "-loglevel", "error", "-i", str(path),
        "-af", af_filters[transform],
        "-f", "f32le", "-ac", "1", "-ar", str(rate), "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(result.stdout, dtype=np.float32)


def _run_clip_through_engine(
    audio_mono: np.ndarray,
    channel: int,
    sd_device_idx: int,
    mic,
    engine,
    post_wait: float = STABILITY_POST_WAIT,
) -> list:
    """Play float32 mono audio through one stereo channel, capture via mic,
    feed to engine, and return all text_queue items produced."""
    import sounddevice as sd

    stereo = np.zeros((len(audio_mono), 2), dtype=np.float32)
    stereo[:, channel] = audio_mono

    playback_done = threading.Event()

    def _play():
        try:
            sd.play(stereo, samplerate=44100, device=sd_device_idx, blocking=True)
        finally:
            playback_done.set()

    items: list = []

    def _drain():
        while not engine.text_queue.empty():
            try:
                items.append(engine.text_queue.get_nowait())
            except queue.Empty:
                break

    threading.Thread(target=_play, daemon=True).start()

    while not playback_done.is_set():
        try:
            chunk = mic.raw_audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        engine.process_audio(chunk)
        _drain()

    flush_end = time.monotonic() + post_wait
    while time.monotonic() < flush_end:
        try:
            chunk = mic.raw_audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        engine.process_audio(chunk)
        _drain()

    _drain()
    return items


# ---------------------------------------------------------------------------
# Speaker identification test (DOA-based: left vs right channel)
# ---------------------------------------------------------------------------

# 2 clips: same voice, different stereo channel → different DOA angle
#   channel 0 = left speaker  → DOA should point roughly left  (~270°)
#   channel 1 = right speaker → DOA should point roughly right (~90°)
SPEAKER_ID_CLIPS = [
    ("left",  MALE_WAV, 0),
    ("right", MALE_WAV, 1),
]


@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_speaker_identification():
    """
    Play the same voice from LEFT then RIGHT stereo channel.
    DOA angles will differ by ~180°, so SpeakerRegistry must register 2 distinct
    speakers purely from direction.
    """
    try:
        import sounddevice as sd  # noqa: F401
    except ImportError:
        pytest.skip("sounddevice not installed: pip install sounddevice")

    _require_wav(MALE_WAV)

    from audio.capture import MicCapture
    from transcribe.streaming_engine import StreamingEngine

    sd_device_idx, speaker_name = _find_sd_output_device()
    print(f"\n[test] sounddevice output: {speaker_name} at index={sd_device_idx}")

    mic = MicCapture(preferred_mic=MIC_NAME)
    engine = StreamingEngine(model_name="small.en", use_vac=True,
                              enable_doa=True, enable_speaker_id=True)
    engine.init_session()
    mic.start()

    segment_results = []

    def _drain_items() -> list:
        items = []
        while not engine.text_queue.empty():
            try:
                items.append(engine.text_queue.get_nowait())
            except queue.Empty:
                break
        return items

    try:
        for label, wav_path, channel in SPEAKER_ID_CLIPS:
            ch_name = "LEFT" if channel == 0 else "RIGHT"
            print(f"\n[test] Playing: {label} ({ch_name} channel)")

            playback_done = threading.Event()

            def _play(w=wav_path, c=channel, dev=sd_device_idx):
                try:
                    _play_mono_to_channel_sd(w, c, dev)
                finally:
                    playback_done.set()

            play_thread = threading.Thread(target=_play, daemon=True)
            play_thread.start()

            seg_items = []

            while not playback_done.is_set():
                try:
                    chunk = mic.raw_audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                engine.process_audio(chunk)
                seg_items.extend(_drain_items())

            flush_end = time.monotonic() + SEGMENT_POST_WAIT
            while time.monotonic() < flush_end:
                try:
                    chunk = mic.raw_audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                engine.process_audio(chunk)
                seg_items.extend(_drain_items())

            play_thread.join(timeout=5)

            speakers = list({_extract_speaker(it) for it in seg_items})
            angles   = [it.get("angle") for it in seg_items
                        if isinstance(it, dict) and it.get("angle") is not None]
            avg_angle = int(np.mean(angles)) if angles else None

            print(f"  Speakers: {speakers}  DOA avg: {avg_angle}°")

            segment_results.append({
                "label":    label,
                "speakers": speakers,
                "angle":    avg_angle,
            })

            time.sleep(1.5)

    finally:
        engine.finish_session()
        _drain_items()
        mic.stop()

    print("\n--- Speaker identification results ---")
    all_speakers: set = set()
    for r in segment_results:
        all_speakers.update(r["speakers"])
        print(f"  {r['label']:6s}  speakers={r['speakers']}  DOA={r['angle']}°")

    n_distinct = len(all_speakers)
    print(f"\n  Distinct speakers: {n_distinct}  (required >= {MIN_DISTINCT_SPEAKERS})")

    assert n_distinct >= MIN_DISTINCT_SPEAKERS, (
        f"Speaker registry identified only {n_distinct} distinct speaker(s) "
        f"from left/right channels (expected >= {MIN_DISTINCT_SPEAKERS}). "
        f"DOA angles: {[r['angle'] for r in segment_results]}"
    )


# ---------------------------------------------------------------------------
# Speaker stability under audio variation
# ---------------------------------------------------------------------------

@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_speaker_stability_under_variation():
    """
    For left and right channels, verify that audio-processed variants of the
    same voice from the same direction are still identified as the SAME speaker.

    Sub-session per channel:
      1. Play original → registers baseline speaker label (angle, e.g. "45°").
      2. Play each variant from the SAME channel → must map back to baseline.

    Pass condition per variant: <= STABILITY_WRONG_FRACTION_LIMIT of committed
    items are assigned to a different speaker than the baseline.
    """
    try:
        import sounddevice as sd  # noqa: F401
    except ImportError:
        pytest.skip("sounddevice not installed: pip install sounddevice")

    _require_wav(MALE_WAV)

    from audio.capture import MicCapture
    from transcribe.streaming_engine import StreamingEngine

    sd_device_idx, speaker_name = _find_sd_output_device()
    print(f"\n[stability] sounddevice output: {speaker_name} at index={sd_device_idx}")

    failures: list = []
    all_results: list = []

    for base_label, wav_path, channel in SPEAKER_ID_CLIPS:
        ch_name = "LEFT" if channel == 0 else "RIGHT"
        print(f"\n[stability] ══════ {base_label} ({ch_name}) ══════")

        mic = MicCapture(preferred_mic=MIC_NAME)
        engine = StreamingEngine(
            model_name="small.en", use_vac=True,
            enable_doa=True, enable_speaker_id=True,
        )
        engine.init_session()
        mic.start()

        clip_results: list = []

        try:
            # ── Step 1: original clip → establish baseline speaker ──────────
            print(f"  [original] loading and playing...")
            orig_audio = _decode_audio_to_float32_mono(wav_path)
            orig_items = _run_clip_through_engine(
                orig_audio, channel, sd_device_idx, mic, engine,
            )

            if not orig_items:
                print(f"  [original] WARNING: no transcript — skipping {base_label}")
                continue

            baseline_label = _extract_speaker(orig_items[0])
            print(f"  [original] DONE  baseline={baseline_label}")

            clip_results.append({
                "variant": "original",
                "wrong": 0,
                "total": len(orig_items),
                "status": "BASELINE",
            })

            # ── Step 2: variants ────────────────────────────────────────────
            for transform, description in STABILITY_TRANSFORMS:
                print(f"  [{transform}] applying transform ({description})...")
                try:
                    variant_audio = _apply_transform(wav_path, transform)
                except Exception as exc:
                    print(f"  [{transform}] SKIP — transform failed: {exc}")
                    continue

                print(f"  [{transform}] playing {len(variant_audio)/44100:.1f}s of audio...")
                var_items = _run_clip_through_engine(
                    variant_audio, channel, sd_device_idx, mic, engine,
                )

                if not var_items:
                    print(f"  [{transform}] WARNING: no transcript produced")
                    clip_results.append({
                        "variant": transform,
                        "wrong": 0,
                        "total": 0,
                        "status": "NO_TRANSCRIPT",
                    })
                    continue

                wrong_items = [
                    it for it in var_items
                    if _extract_speaker(it) != baseline_label
                ]
                wrong_frac = len(wrong_items) / len(var_items)
                passed = wrong_frac <= STABILITY_WRONG_FRACTION_LIMIT
                status = "PASS" if passed else "FAIL"

                print(
                    f"  [{transform}] {status}  "
                    f"wrong={len(wrong_items)}/{len(var_items)} ({wrong_frac:.0%})"
                )

                clip_results.append({
                    "variant": transform,
                    "wrong": len(wrong_items),
                    "total": len(var_items),
                    "status": status,
                })

                if not passed:
                    failures.append(
                        f"{base_label}/{transform}: {wrong_frac:.0%} mis-attributed "
                        f"(expected '{baseline_label}')"
                    )

                time.sleep(0.5)

        finally:
            engine.finish_session()
            mic.stop()

        all_results.append({"base": base_label, "clips": clip_results})
        time.sleep(1.0)

    print("\n══════ Speaker stability summary ══════")
    for entry in all_results:
        print(f"  {entry['base']}:")
        for c in entry["clips"]:
            wrong_str = (
                f"wrong={c['wrong']}/{c['total']} ({c['wrong']/c['total']:.0%})"
                if c["total"] > 0 else "no transcript"
            )
            print(f"    {c['variant']:15s}  {c['status']:12s}  {wrong_str}")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  ✗ {f}")

    assert not failures, (
        f"Speaker stability failed for {len(failures)} variant(s):\n"
        + "\n".join(f"  - {f}" for f in failures)
    )
