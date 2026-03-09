"""
audio/capture.py — Continuous raw audio capture from HIK 1080P USB mic.

Uses PyAudio directly (no SpeechRecognition, no VAD).  Emits small
numpy float32 chunks into raw_audio_queue for consumption by the
streaming transcription engine (which handles its own VAD via Silero).
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

SAMPLE_RATE = 16000       # Hz — native rate of HIK mic, required by Whisper
_CHUNK_SAMPLES = 8000     # samples per callback = 0.5 s at 16 kHz
MIC_NAME = "HIK 1080P Camera"


def find_mic_index() -> Optional[int]:
    """Return the PyAudio device index for the HIK camera mic, or None."""
    pa = pyaudio.PyAudio()
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if MIC_NAME in info.get("name", "") and info.get("maxInputChannels", 0) > 0:
                return i
    finally:
        pa.terminate()
    return None


class MicCapture:
    """Captures audio from the HIK USB mic via a PyAudio stream callback.

    Usage:
        cap = MicCapture()
        cap.start()
        while True:
            chunk = cap.raw_audio_queue.get()   # np.float32 at 16 kHz
            ...
        cap.stop()
    """

    def __init__(self):
        self.raw_audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

        mic_index = find_mic_index()
        if mic_index is None:
            raise RuntimeError(
                f"Microphone '{MIC_NAME}' not found. "
                "Check that the USB camera is plugged in."
            )
        self._device_index = mic_index
        logger.info("[MicCapture] Found '%s' at device index %d", MIC_NAME, mic_index)

    def start(self):
        """Open the PyAudio stream and start the background capture."""
        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=_CHUNK_SAMPLES,
            stream_callback=self._callback,
        )
        self._stream.start_stream()
        logger.info("[MicCapture] Stream started (%.2f s chunks)", _CHUNK_SAMPLES / SAMPLE_RATE)

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
        audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.raw_audio_queue.put(audio)
        return (None, pyaudio.paContinue)
