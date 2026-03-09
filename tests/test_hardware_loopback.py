"""
tests/test_hardware_loopback.py — End-to-end acoustic loopback test.

Plays a TTS-generated audio clip through the Dell AC511 USB SoundBar, records
it via the ReSpeaker 4 Mic Array in the same room, transcribes with
StreamingEngine, and measures Word Error Rate against the reference transcript.

TTS is generated once via espeak-ng (install: sudo apt-get install -y espeak-ng)
and cached in tests/fixtures/reference.wav.

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

AUDIO_FILE = Path(__file__).parent / "fixtures" / "reference.wav"

# Clear natural sentences — easy for Whisper to transcribe from TTS
REFERENCE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Speech recognition works best with clear audio and natural speech. "
    "The microphone array captures sound from multiple directions. "
    "Real time transcription requires low latency processing."
)

REFERENCE = " ".join(
    re.sub(r"[^\w\s']", "", REFERENCE_TEXT.lower()).split()
)

SPEAKER_NAME = "Dell AC511"   # substring match for output device
MIC_NAME = "ReSpeaker"        # substring match for input device

SPEAKER_RATE = 44100
SPEAKER_CHANNELS = 2
POST_PLAYBACK_WAIT = 10      # extra seconds after playback ends to let VAC/LocalAgreement flush buffer
WER_PASS_THRESHOLD = 0.20     # 20% WER = 80% accuracy target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _decode_mp3_to_pcm(path: Path, rate: int, channels: int) -> np.ndarray:
    """Use ffmpeg to decode an MP3 to interleaved int16 PCM."""
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


def _word_error_rate(ref: str, hyp: str) -> float:
    """Compute WER: edit_distance(ref_words, hyp_words) / len(ref_words)."""
    def normalise(s):
        s = s.lower()
        s = re.sub(r"[^\w\s']", "", s)
        return s.split()

    r = normalise(ref)
    h = normalise(hyp)
    if not r:
        return 0.0 if not h else 1.0

    # Levenshtein distance (DP)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r):
        prev = d[:]
        d[0] = i + 1
        for j, hw in enumerate(h):
            d[j + 1] = min(prev[j] + (0 if rw == hw else 1),
                           d[j] + 1,
                           prev[j + 1] + 1)
    return d[len(h)] / len(r)


def _generate_audio_if_missing():
    """Generate TTS reference audio using gTTS (Google TTS, natural voice).

    Requires internet on first run. Output cached in tests/fixtures/reference.wav.
    Falls back to espeak-ng if gTTS is unavailable.
    """
    if AUDIO_FILE.exists():
        return
    AUDIO_FILE.parent.mkdir(parents=True, exist_ok=True)
    mp3_tmp = AUDIO_FILE.with_suffix(".mp3")

    try:
        from gtts import gTTS
        print(f"\n[setup] Generating TTS audio via gTTS → {AUDIO_FILE}")
        tts = gTTS(text=REFERENCE_TEXT, lang="en", slow=False)
        tts.save(str(mp3_tmp))
        # Convert MP3 → WAV (16 kHz mono) so PyAudio can read it directly
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", str(mp3_tmp),
             "-ar", "16000", "-ac", "1", str(AUDIO_FILE)],
            check=True,
        )
        mp3_tmp.unlink(missing_ok=True)
        print("[setup] TTS audio generated (Google TTS).")
    except Exception as e:
        print(f"\n[setup] gTTS failed ({e}), falling back to espeak-ng...")
        mp3_tmp.unlink(missing_ok=True)
        if subprocess.run(["which", "espeak-ng"], capture_output=True).returncode != 0:
            pytest.skip(
                "Neither gTTS nor espeak-ng available. "
                "Install gTTS: pip install gtts  OR  espeak-ng: sudo apt-get install -y espeak-ng"
            )
        subprocess.run(
            ["espeak-ng", "-v", "en-us", "-s", "145", "-a", "180",
             "-w", str(AUDIO_FILE), REFERENCE_TEXT],
            check=True,
        )
        print("[setup] TTS audio generated (espeak-ng fallback).")


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_acoustic_loopback_wer():
    """
    Play the Bane monologue through the Dell AC511 speaker, capture via
    ReSpeaker, transcribe via StreamingEngine, and assert WER < threshold.
    """
    _generate_audio_if_missing()

    from audio.capture import MicCapture
    from transcribe.streaming_engine import StreamingEngine

    speaker_idx = _find_output_device(SPEAKER_NAME)
    print(f"\n[test] Speaker: {SPEAKER_NAME} at index {speaker_idx}")

    # Decode audio for playback
    pcm = _decode_mp3_to_pcm(AUDIO_FILE, SPEAKER_RATE, SPEAKER_CHANNELS)
    audio_duration = len(pcm) / SPEAKER_CHANNELS / SPEAKER_RATE
    print(f"[test] Audio duration: {audio_duration:.1f}s")

    # Start mic capture and engine
    mic = MicCapture(preferred_mic=MIC_NAME)
    engine = StreamingEngine(model_name="base.en", use_vac=True)
    engine.init_session()
    mic.start()

    # --- Playback thread ---
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
    print("[test] Playback started — speak is going through the speaker now...")

    # --- Feed mic audio to engine while playing + post-playback flush ---
    transcripts = []

    def _drain_text():
        while not engine.text_queue.empty():
            text = engine.text_queue.get_nowait()
            if text.strip():
                print(f"[transcript] {text}")
                transcripts.append(text)

    try:
        # Phase 1: feed mic during playback
        while not playback_done.is_set():
            try:
                chunk = mic.raw_audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            engine.process_audio(chunk)
            _drain_text()

        print("[test] Playback complete — flushing remaining audio buffer...")

        # Phase 2: keep feeding mic for POST_PLAYBACK_WAIT to let LocalAgreement commit
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
        _drain_text()   # collect anything finish_session() put in text_queue
        mic.stop()

    playback_thread.join(timeout=5)

    # --- Results ---
    full_hypothesis = " ".join(transcripts)
    wer = _word_error_rate(REFERENCE, full_hypothesis)

    print("\n--- Loopback test results ---")
    print(f"  Reference : {REFERENCE}")
    print(f"  Hypothesis: {full_hypothesis}")
    print(f"  WER       : {wer:.1%}  (threshold: {WER_PASS_THRESHOLD:.0%})")

    assert transcripts, "No transcript produced — check speaker volume and mic position"
    assert wer < WER_PASS_THRESHOLD, (
        f"WER {wer:.1%} exceeds {WER_PASS_THRESHOLD:.0%} threshold.\n"
        f"  Ref: {REFERENCE}\n"
        f"  Hyp: {full_hypothesis}"
    )


@pytest.mark.hardware
@pytest.mark.usefixtures("suspend_service")
def test_speaker_playback_only():
    """Smoke test: verify speaker plays without error (no mic/transcription)."""
    _generate_audio_if_missing()
    speaker_idx = _find_output_device(SPEAKER_NAME)
    pcm = _decode_mp3_to_pcm(AUDIO_FILE, SPEAKER_RATE, SPEAKER_CHANNELS)

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
    _generate_audio_if_missing()
    speaker_idx = _find_output_device(SPEAKER_NAME)

    from audio.capture import MicCapture
    mic = MicCapture(preferred_mic=MIC_NAME)
    mic.start()

    pcm = _decode_mp3_to_pcm(AUDIO_FILE, SPEAKER_RATE, SPEAKER_CHANNELS)

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
