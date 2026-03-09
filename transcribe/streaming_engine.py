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



class _OptimizedWhisperTimestampedASR:
    """Optimized whisper-timestamped backend for Jetson Orin Nano GPU.

    Uses fp16 on CUDA, beam_size=3, and condition_on_previous_text=False
    for faster streaming transcription with LocalAgreement.
    """

    sep = " "

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.original_language = lan
        self.transcribe_kargs = {}
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        logger.info("[_OptimizedWhisperTimestampedASR] Loading '%s'...", modelsize or model_dir)
        self.model = whisper.load_model(modelsize or model_dir, download_root=cache_dir)
        logger.info("[_OptimizedWhisperTimestampedASR] Model loaded")

    def transcribe(self, audio, init_prompt=""):
        import torch
        use_fp16 = torch.cuda.is_available()
        result = self.transcribe_timestamped(
            self.model,
            audio,
            language=self.original_language,
            initial_prompt=init_prompt if init_prompt else None,
            verbose=None,
            condition_on_previous_text=False,
            fp16=use_fp16,
            beam_size=3,
            **self.transcribe_kargs,
        )
        return result

    def ts_words(self, r):
        o = []
        for s in r["segments"]:
            for w in s["words"]:
                t = (w["start"], w["end"], w["text"])
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s["end"] for s in res["segments"]]

    def use_vad(self):
        self.transcribe_kargs["vad"] = True

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
        use_vac: bool = True,
        chunk_seconds: float = 0.5,
        buffer_trimming_sec: float = 8.0,
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

        from whisper_online import OnlineASRProcessor, VACOnlineASRProcessor

        logger.info("[StreamingEngine] Loading model '%s'...", model_name)
        asr = _OptimizedWhisperTimestampedASR(lan=language, modelsize=model_name)

        buffer_trimming = ("segment", buffer_trimming_sec)

        if use_vac:
            logger.info("[StreamingEngine] Using VACOnlineASRProcessor (Silero VAD)")
            # Patch torch.hub.load so silero-vad JIT loads directly,
            # avoiding the torchaudio dependency in the hub's utils_vad.py.
            import torch
            _orig_hub_load = torch.hub.load

            def _hub_load_patch(repo_or_dir, model, *a, **kw):
                if "silero-vad" in str(repo_or_dir) and model == "silero_vad":
                    jit_path = os.path.expanduser(
                        "~/.cache/torch/hub/snakers4_silero-vad_master"
                        "/src/silero_vad/data/silero_vad.jit"
                    )
                    if os.path.exists(jit_path):
                        logger.info("[StreamingEngine] Loading silero-vad JIT from cache: %s", jit_path)
                        m = torch.jit.load(jit_path, map_location="cpu")
                        m.eval()
                        return m, None
                return _orig_hub_load(repo_or_dir, model, *a, **kw)

            torch.hub.load = _hub_load_patch
            try:
                self._online = VACOnlineASRProcessor(
                    chunk_seconds, asr, buffer_trimming=buffer_trimming
                )
            finally:
                torch.hub.load = _orig_hub_load
        else:
            logger.info("[StreamingEngine] Using OnlineASRProcessor (no VAD)")
            self._online = OnlineASRProcessor(asr, buffer_trimming=buffer_trimming)

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

        For VACOnlineASRProcessor: VAC's is_currently_final path skips
        online.process_iter() and jumps straight to online.finish(), so the
        last speech segment audio is in online.audio_buffer but
        transcript_buffer.buffer is stale.  We fix this by running one extra
        process_iter() pass on the underlying OnlineASRProcessor before the
        final finish() so LocalAgreement sees the last segment.

        Safe to call multiple times — subsequent calls are no-ops.
        Returns committed text from the final flush, or None.
        """
        if self._finished:
            return None
        self._finished = True

        # Extra pass for VACOnlineASRProcessor: transcribe audio accumulated
        # in the underlying OnlineASRProcessor that wasn't run through
        # process_iter() before VAC's finish() path was triggered.
        try:
            from whisper_online import VACOnlineASRProcessor
            if isinstance(self._online, VACOnlineASRProcessor):
                begin, end, text = self._online.online.process_iter()
                text = text.strip() if text else ""
                if text:
                    self.text_queue.put(text)
                    logger.info("[StreamingEngine] Pre-flush committed: %r", text)
        except Exception as exc:
            logger.warning("[StreamingEngine] pre-flush process_iter raised: %s", exc)

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
