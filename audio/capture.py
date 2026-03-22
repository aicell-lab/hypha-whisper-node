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

from audio.led_control import led_off

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
        found = find_mic(preferred_mic)
        if found is None:
            raise RuntimeError(
                "No supported USB mic found. "
                "Supported: ReSpeaker 4 Mic Array, HIK 1080P Camera."
            )

        self._dev_index, self._dev_name, self._cap_ch, self._out_ch = found
        logger.info("[MicCapture] Using '%s' (channels=%d, output_ch=%d)",
                    self._dev_name, self._cap_ch, self._out_ch)

        # Turn off ReSpeaker LED ring (those lights are annoying!)
        if "ReSpeaker" in self._dev_name:
            led_off()

        self._pa: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None

        # Queue for ASR audio (ch0 - beamformed)
        self.raw_audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)

        self._running = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_channel(self, interleaved: np.ndarray, ch: int, total_ch: int):
        """Extract a single channel from interleaved multi-channel audio."""
        return interleaved[ch::total_ch].astype(np.float32) / 32768.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the PyAudio input stream (idempotent)."""
        if self._running:
            return

        self._pa = pyaudio.PyAudio()

        def _callback(in_data, frame_count, time_info, status):
            raw = np.frombuffer(in_data, dtype=np.int16)
            
            # Extract the beamformed channel (ch0 for ReSpeaker)
            ch_data = self._extract_channel(raw, self._out_ch, self._cap_ch)
            
            try:
                self.raw_audio_queue.put_nowait(ch_data)
            except queue.Full:
                pass  # Drop if consumer is slow
            
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self._cap_ch,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self._dev_index,
            frames_per_buffer=_CHUNK_SAMPLES,
            stream_callback=_callback,
        )

        self._stream.start_stream()
        self._running = True
        logger.info("[MicCapture] Stream started")

    def stop(self) -> None:
        """Stop and close the PyAudio stream.
        
        This method is idempotent and handles partial cleanup gracefully.
        Should be called on shutdown to release the USB audio device.
        """
        if not self._running and self._stream is None and self._pa is None:
            return
        
        self._running = False

        # Stop and close stream with error handling
        if self._stream is not None:
            try:
                if self._stream.is_active():
                    self._stream.stop_stream()
            except Exception as exc:
                logger.debug("[MicCapture] Error stopping stream: %s", exc)
            try:
                self._stream.close()
            except Exception as exc:
                logger.debug("[MicCapture] Error closing stream: %s", exc)
            finally:
                self._stream = None

        # Terminate PyAudio instance
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception as exc:
                logger.debug("[MicCapture] Error terminating PyAudio: %s", exc)
            finally:
                self._pa = None

        # Drain queue to help GC
        while not self.raw_audio_queue.empty():
            try:
                self.raw_audio_queue.get_nowait()
            except queue.Empty:
                break

        logger.info("[MicCapture] Stream stopped")
