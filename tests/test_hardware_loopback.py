"""
tests/test_hardware_loopback.py — End-to-end acoustic loopback tests.

Audio sources:
  tests/test-audio-male.wav   — male voice reading the reference text
  tests/test-audio-femal.wav  — female voice reading the reference text

Both files were provided as pre-recorded custom audio (no TTS generation needed).

Tests:
  test_speaker_playback_only    — smoke test: play male audio through speaker bar
  test_mic_capture_rms          — verify ReSpeaker picks up audio (RMS threshold)
  test_acoustic_loopback_wer    — full pipeline WER < threshold (male voice)
  test_speaker_identification   — 4 clips (male/female × left/right channel);
                                  assert >= 2 distinct speakers identified

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
FEMALE_WAV     = Path(__file__).parent / "test-audio-femal.wav"

# Text spoken in both audio files
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

SPEAKER_NAME = "Dell AC511"   # substring match for output device
MIC_NAME     = "ReSpeaker"   # substring match for input device

SPEAKER_RATE     = 44100
SPEAKER_CHANNELS = 2
POST_PLAYBACK_WAIT   = 15    # seconds to flush VAC/LocalAgreement buffer after playback
WER_PASS_THRESHOLD   = 0.30  # Never change this! Otherwise this is cheating and we can't trust the test!

# Speaker identification test settings
SEGMENT_POST_WAIT      = 6    # seconds to flush each clip after playback
MULTI_WER_THRESH       = 0.50 # looser — short clips through loopback
MIN_DISTINCT_SPEAKERS  = 2    # must identify at least this many distinct speakers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_wav(path: Path) -> None:
    """Skip test if audio file is missing."""
    if not path.exists():
        pytest.skip(f"Audio file not found: {path}. Place it in the tests/ directory.")


def _find_output_device(name_sub: str) -> int:
    """Return PyAudio device index for a named output device."""
    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if name_sub in info.get("name", "") and info.get("maxOutputChannels", 0) > 0:
                return i
    finally:
        pa.terminate()
    raise RuntimeError(f"Output device containing '{name_sub}' not found")


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


def _find_sd_output_device(name_sub: str) -> int:
    """Return sounddevice device index for a named output device."""
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        for i, d in enumerate(devs):
            if name_sub in d["name"] and d["max_output_channels"] > 0:
                return i
    except Exception as exc:
        raise RuntimeError(f"sounddevice not available: {exc}")
    raise RuntimeError(f"sounddevice output device containing '{name_sub}' not found")


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
        return item.get("speaker", "Speaker 1")
    return "Speaker 1"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_speaker_playback_only():
    """Smoke test: verify speaker plays male audio without error."""
    _require_wav(MALE_WAV)
    speaker_idx = _find_output_device(SPEAKER_NAME)
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

    print(f"\n[test] Played {len(pcm)/SPEAKER_CHANNELS/SPEAKER_RATE:.1f}s audio through {SPEAKER_NAME}")


@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_mic_capture_rms():
    """Verify ReSpeaker captures non-silent audio during playback."""
    _require_wav(MALE_WAV)
    speaker_idx = _find_output_device(SPEAKER_NAME)

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

    assert avg_rms > 0.001, (
        f"RMS too low ({avg_rms:.5f}) — mic may not be picking up speaker audio. "
        "Check volume and physical proximity."
    )


@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_acoustic_loopback_wer():
    """
    Play male audio through Dell AC511, capture via ReSpeaker, transcribe via
    StreamingEngine, and assert WER < threshold.
    """
    _require_wav(MALE_WAV)

    from audio.capture import MicCapture
    from transcribe.streaming_engine import StreamingEngine

    speaker_idx = _find_output_device(SPEAKER_NAME)
    print(f"\n[test] Speaker: {SPEAKER_NAME} at index {speaker_idx}")

    pcm = _decode_audio_to_pcm(MALE_WAV, SPEAKER_RATE, SPEAKER_CHANNELS)
    audio_duration = len(pcm) / SPEAKER_CHANNELS / SPEAKER_RATE
    print(f"[test] Audio duration: {audio_duration:.1f}s")

    mic = MicCapture(preferred_mic=MIC_NAME)
    engine = StreamingEngine(model_name="base.en", use_vac=True,
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
                spk = item.get("speaker", "") if isinstance(item, dict) else ""
                ang = item.get("angle") if isinstance(item, dict) else None
                badge = f"[{spk}|{ang}°] " if spk else ""
                print(f"[transcript] {badge}{text}")
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
# Multi-speaker identification test
# ---------------------------------------------------------------------------

# 4 clips: (label, wav_path, stereo_channel)
#   channel 0 = left speaker  → DOA should point roughly left  (~270°)
#   channel 1 = right speaker → DOA should point roughly right (~90°)
MULTI_SPEAKER_CLIPS = [
    ("male_left",    MALE_WAV,   0),
    ("female_left",  FEMALE_WAV, 0),
    ("male_right",   MALE_WAV,   1),
    ("female_right", FEMALE_WAV, 1),
]


@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_speaker_identification():
    """
    Play 4 audio clips (male/female × left/right channel) through Dell AC511.
    Verify StreamingEngine + SpeakerRegistry identifies >= 2 distinct speakers.

    Voice differentiation: male vs female voice embeddings.
    Spatial differentiation: left vs right stereo channel → different DOA angles.
    """
    try:
        import sounddevice as sd  # noqa: F401
    except ImportError:
        pytest.skip("sounddevice not installed: pip install sounddevice")

    _require_wav(MALE_WAV)
    _require_wav(FEMALE_WAV)

    from audio.capture import MicCapture
    from transcribe.streaming_engine import StreamingEngine

    sd_device_idx = _find_sd_output_device(SPEAKER_NAME)
    print(f"\n[test] sounddevice output: index={sd_device_idx}")

    mic = MicCapture(preferred_mic=MIC_NAME)
    engine = StreamingEngine(model_name="base.en", use_vac=True,
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
        for label, wav_path, channel in MULTI_SPEAKER_CLIPS:
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

            hyp_text = " ".join(_extract_text(it) for it in seg_items).strip()
            speakers  = list({_extract_speaker(it) for it in seg_items})
            angles    = [it.get("angle") for it in seg_items
                         if isinstance(it, dict) and it.get("angle") is not None]
            avg_angle = int(np.mean(angles)) if angles else None

            print(f"  Hypothesis: {hyp_text[:80]}")
            print(f"  Speakers  : {speakers}  DOA avg: {avg_angle}°")

            segment_results.append({
                "label":    label,
                "hyp_text": hyp_text,
                "speakers": speakers,
                "angle":    avg_angle,
            })

            # Brief pause between clips so LocalAgreement separates them
            time.sleep(1.5)

    finally:
        engine.finish_session()
        _drain_items()
        mic.stop()

    # --- Assertions ---
    print("\n--- Multi-speaker identification results ---")

    all_speakers: set = set()
    for r in segment_results:
        all_speakers.update(r["speakers"])
        wer = _word_error_rate(REFERENCE, r["hyp_text"])
        print(f"  {r['label']:15s}  speakers={r['speakers']}  DOA={r['angle']}°  WER={wer:.1%}")

    n_distinct = len(all_speakers)
    print(f"\n  Distinct speakers: {n_distinct}  {sorted(all_speakers)}")
    print(f"  Required         : >= {MIN_DISTINCT_SPEAKERS}")

    assert n_distinct >= MIN_DISTINCT_SPEAKERS, (
        f"Speaker registry identified only {n_distinct} distinct speaker(s) "
        f"(expected >= {MIN_DISTINCT_SPEAKERS}). "
        f"Speakers seen: {sorted(all_speakers)}"
    )
