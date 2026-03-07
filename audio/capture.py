"""
audio/capture.py — Continuous mic capture for HIK 1080P USB camera.

Uses SpeechRecognition with PyAudio backend. Detects the HIK camera
by name so it works even if the ALSA card index shifts on reboot.
Puts raw PCM bytes into a thread-safe Queue consumed by the transcriber.

VAD: webrtcvad runs on every captured chunk as a second-stage filter.
SpeechRecognition's energy threshold gates the first stage (open/close mic).
webrtcvad then verifies the chunk actually contains speech frames before
enqueueing it, dropping false positives (loud bangs, clicks, fan noise).
"""

import ctypes
import queue
from typing import Optional
import numpy as np
import webrtcvad
import speech_recognition as sr

# Silence ALSA lib error messages globally (they spam stderr when PyAudio
# probes unavailable virtual PCM devices like hdmi, modem, rear, etc.)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(ctypes.c_void_p(None))
except OSError:
    pass


SAMPLE_RATE = 16000       # Hz — native rate of HIK mic, required by Whisper
RECORD_TIMEOUT = 3        # seconds: max duration of a single recording chunk
PHRASE_TIMEOUT = 3        # seconds: silence gap that marks end of a phrase
ENERGY_THRESHOLD = 1000   # RMS energy threshold for voice detection
MIC_NAME = "HIK 1080P Camera"

# webrtcvad settings
VAD_AGGRESSIVENESS = 3    # 0=least aggressive, 3=most aggressive (was 2)
VAD_FRAME_MS = 20         # webrtcvad supports 10, 20, or 30 ms frames
VAD_SPEECH_RATIO = 0.5    # minimum fraction of voiced frames to pass chunk (was 0.3)


def _bandpass_filter(audio: np.ndarray, low_hz: int = 300, high_hz: int = 3400) -> np.ndarray:
    """Zero out frequencies outside the speech band via FFT (no scipy needed)."""
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / SAMPLE_RATE)
    fft[(freqs < low_hz) | (freqs > high_hz)] = 0
    return np.fft.irfft(fft, n=len(audio))


def _normalize_rms(audio: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """Scale audio to a fixed RMS level so quiet speech is not drowned by noise."""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-6:
        return audio
    return audio * (target_rms / rms)


def find_mic_index() -> Optional[int]:
    """Return the PyAudio device index for the HIK camera mic, or None."""
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        if MIC_NAME in name:
            return i
    return None


def _vad_has_speech(pcm_bytes: bytes, aggressiveness: int = VAD_AGGRESSIVENESS) -> bool:
    """
    Return True if at least VAD_SPEECH_RATIO of 20ms frames are voiced.
    pcm_bytes must be S16LE at SAMPLE_RATE.
    """
    vad = webrtcvad.Vad(aggressiveness)
    frame_bytes = int(SAMPLE_RATE * VAD_FRAME_MS / 1000) * 2  # 2 bytes per S16 sample
    total = voiced = 0
    for i in range(0, len(pcm_bytes) - frame_bytes + 1, frame_bytes):
        frame = pcm_bytes[i:i + frame_bytes]
        total += 1
        if vad.is_speech(frame, SAMPLE_RATE):
            voiced += 1
    if total == 0:
        return False
    return (voiced / total) >= VAD_SPEECH_RATIO


class MicCapture:
    """
    Captures audio from the HIK USB mic in a background thread.

    Usage:
        cap = MicCapture()
        cap.start()
        while True:
            audio_bytes = cap.queue.get()   # raw S16LE PCM at 16kHz
            ...
        cap.stop()
    """

    def __init__(self):
        self.queue: queue.Queue[bytes] = queue.Queue()
        self._recognizer = sr.Recognizer()
        self._recognizer.energy_threshold = ENERGY_THRESHOLD
        self._recognizer.dynamic_energy_threshold = False
        self._stop_fn = None

        mic_index = find_mic_index()
        if mic_index is None:
            raise RuntimeError(
                f"Microphone '{MIC_NAME}' not found. "
                "Check that the USB camera is plugged in."
            )
        self._source = sr.Microphone(
            device_index=mic_index,
            sample_rate=SAMPLE_RATE,
        )

    def start(self):
        """Adjust for ambient noise, then start background recording."""
        with self._source:
            self._recognizer.adjust_for_ambient_noise(self._source)

        self._stop_fn = self._recognizer.listen_in_background(
            self._source,
            self._callback,
            phrase_time_limit=RECORD_TIMEOUT,
        )

    def stop(self):
        if self._stop_fn:
            self._stop_fn(wait_for_stop=False)

    def _callback(self, recognizer, audio: sr.AudioData):
        pcm = audio.get_raw_data()
        if not _vad_has_speech(pcm):   # VAD runs on raw S16LE (webrtcvad requirement)
            return
        # Bandpass filter + RMS normalization before Whisper sees the audio
        audio_f32 = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        audio_f32 = _bandpass_filter(audio_f32)
        audio_f32 = _normalize_rms(audio_f32)
        processed = (audio_f32 * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
        self.queue.put(processed)
