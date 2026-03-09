"""
transcribe/streaming_engine.py — Streaming Whisper engine using LocalAgreement.

Wraps whisper_streaming's OnlineASRProcessor / VACOnlineASRProcessor.
Usage:
    engine = StreamingEngine(model_name="base.en")
    engine.init_session()          # called when SSE client connects
    text = engine.process_audio(chunk)   # np.float32, 16 kHz, any size
    engine.finish_session()        # called when SSE client disconnects
"""

import logging
import queue
import sys
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "base.en"


class _JetsonFasterWhisperASR:
    """faster-whisper backend subclass tuned for Jetson Orin Nano.

    Overrides load_model() to use device="auto" and compute_type="float32"
    so it auto-selects CUDA and avoids float16 precision issues on Tegra.
    Mirrors the FasterWhisperASR interface from whisper_online.py.
    """

    sep = ""

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.original_language = lan
        self.modelsize = modelsize
        self.cache_dir = cache_dir
        self.model_dir = model_dir
        self.logfile = logfile
        self.transcribe_kargs = {}
        self.model = None  # loaded lazily by load_model()

    def load_model(self):
        from faster_whisper import WhisperModel
        model_size_or_path = self.model_dir or self.modelsize
        if not model_size_or_path:
            raise ValueError("modelsize or model_dir must be set")
        logger.info("[_JetsonFasterWhisperASR] Loading '%s' on cpu (float32)...", model_size_or_path)
        self.model = WhisperModel(
            model_size_or_path,
            device="cpu",
            compute_type="float32",
            download_root=self.cache_dir,
        )
        return self.model

    def transcribe(self, audio, init_prompt=""):
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt if init_prompt else None,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments):
        out = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                out.append((word.start, word.end, word.word))
        return out

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class StreamingEngine:
    """Wraps OnlineASRProcessor and exposes a simple per-session API.

    Attributes:
        model_name: Whisper model name (used by /health endpoint).
        text_queue: Queue[str] — committed transcript segments placed here
                    by process_audio(). Consumed by HyphaClient's sse_gen().
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        language: str = "en",
        backend: str = "faster-whisper",
        use_vac: bool = False,
        chunk_seconds: float = 0.5,
    ):
        self.model_name = model_name
        self.text_queue: queue.Queue[str] = queue.Queue()
        self._finished = False

        # Add transcribe/ to sys.path so whisper_online.py can be imported
        _here = os.path.dirname(__file__)
        if _here not in sys.path:
            sys.path.insert(0, _here)
        # Project root also needed for silero_vad_iterator
        _root = os.path.dirname(_here)
        if _root not in sys.path:
            sys.path.insert(0, _root)

        from whisper_online import (
            WhisperTimestampedASR,
            FasterWhisperASR,
            OnlineASRProcessor,
            VACOnlineASRProcessor,
        )

        logger.info("[StreamingEngine] Loading model '%s' backend='%s'...", model_name, backend)
        if backend == "faster-whisper":
            # _JetsonFasterWhisperASR does NOT load in __init__, so call explicitly.
            asr = _JetsonFasterWhisperASR(lan=language, modelsize=model_name)
            asr.load_model()
        else:
            # WhisperTimestampedASR (ASRBase) loads the model inside __init__.
            asr = WhisperTimestampedASR(lan=language, modelsize=model_name)

        if use_vac:
            logger.info("[StreamingEngine] Using VACOnlineASRProcessor (Silero VAD)")
            self._online = VACOnlineASRProcessor(chunk_seconds, asr)
        else:
            logger.info("[StreamingEngine] Using OnlineASRProcessor (no VAD)")
            self._online = OnlineASRProcessor(asr)

        logger.info("[StreamingEngine] Ready")

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def init_session(self, offset: Optional[float] = None) -> None:
        """Reset streaming state for a new client connection.

        Drains any stale text from the queue and resets LocalAgreement
        hypothesis buffer and audio accumulation.
        """
        self._finished = False
        # Drain stale text from previous session
        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break
        self._online.init()
        logger.info("[StreamingEngine] Session initialised")

    def finish_session(self) -> Optional[str]:
        """Flush remaining audio context at end of client session.

        Safe to call multiple times — subsequent calls are no-ops.
        Returns committed text from the final flush, or None.
        """
        if self._finished:
            return None
        self._finished = True
        try:
            begin, end, text = self._online.finish()
        except Exception as exc:
            logger.warning("[StreamingEngine] finish() raised: %s", exc)
            return None
        text = text.strip() if text else ""
        if text:
            self.text_queue.put(text)
            logger.info("[StreamingEngine] Flushed final: %r", text)
            return text
        return None

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def process_audio(self, chunk: np.ndarray) -> Optional[str]:
        """Feed an audio chunk and return committed text if available.

        Args:
            chunk: np.float32 array at 16 kHz, range ~[-1.0, 1.0].
        Returns:
            Committed text string, or None if nothing committed yet.
        """
        self._online.insert_audio_chunk(chunk)
        try:
            begin, end, text = self._online.process_iter()
        except Exception as exc:
            logger.warning("[StreamingEngine] process_iter raised: %s", exc)
            return None
        text = text.strip() if text else ""
        if text:
            self.text_queue.put(text)
            logger.debug("[StreamingEngine] Committed [%.2f–%.2f]: %r", begin or 0, end or 0, text)
            return text
        return None
