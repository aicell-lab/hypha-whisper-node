"""
transcribe/whisper_engine.py — Whisper inference wrapper for Jetson Orin.

Loads openai-whisper onto the GPU and exposes a single transcribe() method
that accepts raw S16LE PCM bytes (16kHz mono) from audio/capture.py.
"""

import re
import time
import numpy as np
import torch
import whisper

# fp16=False: Jetson Orin supports FP16 but Whisper's FP16 path can produce
# NaN on some inputs; FP32 is safe and performance difference is small for
# base.en on this hardware.
DEFAULT_MODEL = "small.en"
SAMPLE_RATE = 16000

# Regex covering common Whisper hallucination patterns (bracketed tags,
# repetitive filler phrases observed in the wild and in reference implementations).
_HALLUCINATION_RE = re.compile(
    r'\[.*?\]'                        # [BLANK_AUDIO], [Music], [Silence], …
    r'|\(.*?\)'                       # (music), (silence), …
    r'|(?:thank you[\.,]?\s*){2,}'   # "Thank you. Thank you."
    r'|(?:okay[\.,]?\s*){2,}'        # "Okay. Okay. Okay"
    r"|(?:i'm sorry[\.,]?\s*){2,}"   # "I'm sorry. I'm sorry."
    r'|(?:a little bit of\s+){2,}'   # "a little bit of a little bit of"
    r'|!{3,}'                         # "!!!!!"
    r'|(?:\.\s*){3,}',               # ellipsis spam "... ..."
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Strip hallucination artifacts; return empty string if nothing real remains."""
    text = _HALLUCINATION_RE.sub("", text).strip()
    return text if len(text) >= 2 else ""


class WhisperEngine:
    """
    Loads a Whisper model on GPU and transcribes PCM audio chunks.

    Usage:
        engine = WhisperEngine()          # loads model
        text = engine.transcribe(pcm_bytes)
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
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
            condition_on_previous_text=False,  # prevents hallucination cascades
            no_speech_threshold=0.6,           # reject segments Whisper flags as non-speech
            logprob_threshold=-1.0,            # reject low-confidence segments
            compression_ratio_threshold=2.4,   # reject repetitive output
        )
        return _clean(result["text"].strip())
