"""
audio/capture.py — Continuous raw audio capture (ReSpeaker or HIK USB mic).

Uses PyAudio directly (no SpeechRecognition, no VAD).  Emits small
numpy float32 chunks into raw_audio_queue for consumption by the
streaming transcription engine (which handles its own VAD via Silero).

ReSpeaker 4 Mic Array (UAC1.0): 6-channel, 16 kHz. Channel 0 is the
beamformed output — the other 5 channels are raw mics + playback ref.
"""

import ctypes
import logging
import queue
from typing import Optional

import numpy as np
import pyaudio

# Silence ALSA lib error messages (they spam stderr when PyAudio
# probes unavailable virtual PCM devices like hdmi, modem, rear, etc.)
try:
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(ctypes.c_void_p(None))
except OSError:
    pass

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000    # Hz — required by Whisper; native rate of both mics
_CHUNK_SAMPLES = 8000  # samples per callback = 0.5 s at 16 kHz

# Supported mic profiles: (name_substring, capture_channels, beamformed_channel)
_MIC_PROFILES = [
    ("ReSpeaker 4 Mic Array", 6, 0),   # ch0 = beamformed output
    ("HIK 1080P Camera",      1, 0),   # mono
]


def find_mic(preferred: Optional[str] = None):
    """Scan PyAudio devices and return (index, name, capture_channels, out_channel).

    Tries mics in _MIC_PROFILES order unless *preferred* name is given.
    Returns None if nothing matches.
    """
    pa = pyaudio.PyAudio()
    try:
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append((i, info["name"], info["maxInputChannels"]))
    finally:
        pa.terminate()

    profiles = _MIC_PROFILES
    if preferred:
        profiles = [(preferred, 1, 0)] + list(_MIC_PROFILES)

    for name_sub, cap_ch, out_ch in profiles:
        for idx, dev_name, max_ch in devices:
            if name_sub in dev_name and max_ch >= cap_ch:
                return idx, dev_name, cap_ch, out_ch

    return None


class MicCapture:
    """Captures audio from a USB mic via a PyAudio stream callback.

    Usage:
        cap = MicCapture()
        cap.start()
        while True:
            chunk = cap.raw_audio_queue.get()   # np.float32, mono, 16 kHz
            ...
        cap.stop()
    """

    def __init__(self, preferred_mic: Optional[str] = None):
        self.raw_audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

        result = find_mic(preferred_mic)
        if result is None:
            names = [n for n, _, _ in _MIC_PROFILES]
            raise RuntimeError(
                f"No supported microphone found (tried: {names}). "
                "Check that the USB device is plugged in."
            )
        self._device_index, self._device_name, self._cap_ch, self._out_ch = result
        logger.info("[MicCapture] Found '%s' at device index %d (capture ch=%d, use ch=%d)",
                    self._device_name, self._device_index, self._cap_ch, self._out_ch)

    def start(self):
        """Open the PyAudio stream and start background capture."""
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._cap_ch,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=_CHUNK_SAMPLES,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        logger.info("[MicCapture] Stream started (%.2f s chunks, %d-ch → ch%d)",
                    _CHUNK_SAMPLES / SAMPLE_RATE, self._cap_ch, self._out_ch)

    def stop(self):
        """Stop the stream and release resources."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
        logger.info("[MicCapture] Stopped")

    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback — runs in a dedicated audio thread."""
        raw = np.frombuffer(in_data, dtype=np.int16)
        if self._cap_ch > 1:
            # Interleaved multi-channel: extract the beamformed channel
            raw = raw[self._out_ch :: self._cap_ch]
        audio = raw.astype(np.float32) / 32768.0
        self.raw_audio_queue.put(audio)
        return (None, pyaudio.paContinue)
