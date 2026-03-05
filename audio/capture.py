"""
audio/capture.py — Continuous mic capture for HIK 1080P USB camera.

Uses SpeechRecognition with PyAudio backend. Detects the HIK camera
by name so it works even if the ALSA card index shifts on reboot.
Puts raw PCM bytes into a thread-safe Queue consumed by the transcriber.
"""

import queue
from typing import Optional
import speech_recognition as sr


SAMPLE_RATE = 16000       # Hz — native rate of HIK mic, required by Whisper
RECORD_TIMEOUT = 2        # seconds: max duration of a single recording chunk
PHRASE_TIMEOUT = 3        # seconds: silence gap that marks end of a phrase
ENERGY_THRESHOLD = 1000   # RMS energy threshold for voice detection
MIC_NAME = "HIK 1080P Camera"


def find_mic_index() -> Optional[int]:
    """Return the PyAudio device index for the HIK camera mic, or None."""
    for i, name in enumerate(sr.Microphone.list_microphone_names()):
        if MIC_NAME in name:
            return i
    return None


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
        self.queue.put(audio.get_raw_data())
