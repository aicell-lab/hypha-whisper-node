"""
transcribe/whisper_engine.py — Whisper inference wrapper for Jetson Orin.

Loads openai-whisper onto the GPU and exposes a single transcribe() method
that accepts raw S16LE PCM bytes (16kHz mono) from audio/capture.py.
"""

import time
import numpy as np
import torch
import whisper

# fp16=False: Jetson Orin supports FP16 but Whisper's FP16 path can produce
# NaN on some inputs; FP32 is safe and performance difference is small for
# base.en on this hardware.
DEFAULT_MODEL = "base.en"
SAMPLE_RATE = 16000


class WhisperEngine:
    """
    Loads a Whisper model on GPU and transcribes PCM audio chunks.

    Usage:
        engine = WhisperEngine()          # loads model
        text = engine.transcribe(pcm_bytes)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[whisper] Loading model '{model_name}' on {self.device}...")
        t0 = time.time()
        self.model = whisper.load_model(model_name, device=self.device)
        print(f"[whisper] Model loaded in {time.time() - t0:.1f}s")

    def transcribe(self, pcm_bytes: bytes) -> str:
        """
        Transcribe raw S16LE PCM bytes captured at 16kHz mono.
        Returns the transcript string (empty string if nothing detected).
        """
        audio_np = (
            np.frombuffer(pcm_bytes, dtype=np.int16)
            .astype(np.float32) / 32768.0
        )

        result = self.model.transcribe(
            audio_np,
            fp16=False,
            temperature=0.0,
            language="en",
        )
        return result["text"].strip()
