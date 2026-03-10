"""
transcribe/streaming_engine.py — Streaming Whisper engine using LocalAgreement.

Wraps whisper_streaming's OnlineASRProcessor / VACOnlineASRProcessor.
Usage:
    engine = StreamingEngine(model_name="small.en")
    engine.init_session()          # called when SSE client connects
    text = engine.process_audio(chunk)   # np.float32, 16 kHz, any size
    engine.finish_session()        # called when SSE client disconnects

text_queue items are dicts: {"text": str, "speaker": str, "angle": int|None}
when DOAReader / SpeakerRegistry are available, otherwise plain str for
backwards compatibility.
"""

import logging
import queue
import sys
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "small.en"


def _try_import_doa():
    """Lazy import of DOAReader — returns class or None."""
    try:
        # Ensure project root is in sys.path for 'audio' package
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from audio.doa_reader import DOAReader
        return DOAReader
    except Exception:
        return None


def _try_import_speaker_registry():
    """Lazy import of SpeakerRegistry — returns class or None."""
    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from transcribe.speaker_registry import SpeakerRegistry
        return SpeakerRegistry
    except Exception:
        return None



class _DistilWhisperASR:
    """Distil-Whisper backend using HuggingFace transformers.

    Uses distil-whisper/distil-small.en — ~2x faster than small.en with near-identical WER.
    Runs on PyTorch CUDA via transformers pipeline (no CTranslate2 required).
    Word-level timestamps are produced via return_timestamps="word".
    """

    sep = " "

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        hf_model = model_dir or modelsize or "distil-whisper/distil-small.en"
        # English-only models (*.en) don't accept task/language in generate_kwargs
        self._is_english_only = hf_model.endswith(".en")
        self.original_language = None if (self._is_english_only or lan == "auto") else lan
        self.transcribe_kargs = {}
        import torch
        from transformers import pipeline as hf_pipeline
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[_DistilWhisperASR] CUDA not available — "
                "cannot load model (check LD_LIBRARY_PATH and GPU state)"
            )
        logger.info("[_DistilWhisperASR] Loading '%s' on cuda...", hf_model)
        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=hf_model,
            torch_dtype=torch.float16,
            device="cuda",
        )
        logger.info("[_DistilWhisperASR] Model loaded on cuda")

    def transcribe(self, audio, init_prompt=""):
        generate_kwargs = {}
        if self.original_language:
            generate_kwargs["language"] = self.original_language
        generate_kwargs.update(self.transcribe_kargs)
        # distil-whisper crashes with return_timestamps="word" (cross-attention
        # alignment heads mismatch in distilled layers). Chunk-level is fine.
        result = self._pipe(
            audio.copy(),
            return_timestamps=True,
            generate_kwargs=generate_kwargs,
        )
        return result

    def ts_words(self, r):
        """Convert HuggingFace chunks [{"text":..,"timestamp":(s,e)}] to [(start,end,text)]."""
        o = []
        for chunk in r.get("chunks", []):
            ts = chunk.get("timestamp") or (None, None)
            start, end = ts
            if start is None or end is None:
                continue
            o.append((start, end, chunk["text"]))
        return o

    def segments_end_ts(self, res):
        """Group words into pseudo-segments by silence gaps > 0.5s; return segment end times."""
        words = self.ts_words(res)
        if not words:
            return []
        segment_ends = []
        prev_end = words[0][1]
        for start, end, _ in words[1:]:
            if start - prev_end > 0.5:
                segment_ends.append(prev_end)
            prev_end = end
        segment_ends.append(prev_end)
        return segment_ends

    def use_vad(self):
        pass  # VAC handles VAD externally

    def set_translate_task(self):
        if not self._is_english_only:
            self.transcribe_kargs["task"] = "translate"


class _OptimizedWhisperTimestampedASR:
    """Optimized whisper-timestamped backend for Jetson Orin Nano GPU.

    Uses fp16 on CUDA, beam_size=3, and condition_on_previous_text=False
    for faster streaming transcription with LocalAgreement.
    """

    sep = " "

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.original_language = lan
        self.transcribe_kargs = {}
        import torch
        import whisper
        import whisper_timestamped
        from whisper_timestamped import transcribe_timestamped
        self.transcribe_timestamped = transcribe_timestamped
        if not torch.cuda.is_available():
            raise RuntimeError(
                "[_OptimizedWhisperTimestampedASR] CUDA not available — "
                "cannot load Whisper model (check LD_LIBRARY_PATH and GPU state)"
            )
        logger.info("[_OptimizedWhisperTimestampedASR] Loading '%s' on cuda...",
                    modelsize or model_dir)
        self.model = whisper.load_model(modelsize or model_dir, device="cuda",
                                        download_root=cache_dir)
        logger.info("[_OptimizedWhisperTimestampedASR] Model loaded on cuda")

    def transcribe(self, audio, init_prompt=""):
        result = self.transcribe_timestamped(
            self.model,
            audio,
            language=self.original_language,
            initial_prompt=init_prompt if init_prompt else None,
            verbose=None,
            condition_on_previous_text=False,
            fp16=True,
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
        text_queue: Queue of committed transcript items. Each item is either:
            - dict: {"text": str, "speaker": str, "angle": int|None}  (when speaker ID is active)
            - str:  plain committed text  (legacy/fallback)
          Consumed by HyphaClient's sse_gen().
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        language: str = "en",
        backend: str = "whisper-timestamped",
        use_vac: bool = True,
        chunk_seconds: float = 0.5,
        buffer_trimming_sec: float = 8.0,
        enable_doa: bool = True,
        enable_speaker_id: bool = True,
    ):
        self.model_name = model_name
        self.text_queue: queue.Queue = queue.Queue()
        self._finished = False

        # DOA reader (optional — USB hardware DOA via ReSpeaker ctrl_transfer)
        self._doa: Optional[object] = None
        if enable_doa:
            _DOAReader = _try_import_doa()
            if _DOAReader is not None:
                try:
                    self._doa = _DOAReader()
                    self._doa.start()
                except Exception as exc:
                    logger.warning("[StreamingEngine] DOAReader init failed: %s", exc)
                    self._doa = None

        # Speaker registry (optional)
        self._speaker_registry: Optional[object] = None
        if enable_speaker_id:
            _SpeakerRegistry = _try_import_speaker_registry()
            if _SpeakerRegistry is not None:
                try:
                    self._speaker_registry = _SpeakerRegistry()
                except Exception as exc:
                    logger.warning("[StreamingEngine] SpeakerRegistry init failed: %s", exc)
                    self._speaker_registry = None

        self._last_commit_time: float = 0.0

        # Add transcribe/ to sys.path so whisper_online.py can be imported
        _here = os.path.dirname(__file__)
        if _here not in sys.path:
            sys.path.insert(0, _here)
        # Project root also needed for silero_vad_iterator
        _root = os.path.dirname(_here)
        if _root not in sys.path:
            sys.path.insert(0, _root)

        from whisper_online import OnlineASRProcessor, VACOnlineASRProcessor

        logger.info("[StreamingEngine] Loading model '%s' (backend: %s)...", model_name, backend)
        if backend == "distil-whisper":
            asr = _DistilWhisperASR(lan=language, modelsize=model_name)
        else:
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
        self._last_commit_time = time.monotonic()
        if self._speaker_registry is not None:
            self._speaker_registry.reset()
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
                    self._emit_item(text, self._last_commit_time)
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
            self._emit_item(text, self._last_commit_time)
            logger.info("[StreamingEngine] Flushed final: %r", text)

        return text if text else None

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
            commit_time = self._last_commit_time
            self._last_commit_time = time.monotonic()
            self._emit_item(text, commit_time)
            logger.debug("[StreamingEngine] Committed [%.2f–%.2f]: %r", begin or 0, end or 0, text)
            return text
        return None

    # ------------------------------------------------------------------
    # Speaker identification helpers
    # ------------------------------------------------------------------

    def _emit_item(self, text: str, commit_time: float) -> None:
        """Build and enqueue a text_queue item with DOA-based speaker ID."""
        angle: Optional[int] = None
        speaker = "Speaker 1"

        if self._doa is not None and self._doa.enabled:
            angle = self._doa.median_angle_since(commit_time)

        if self._speaker_registry is not None:
            try:
                speaker = self._speaker_registry.identify(doa_angle=angle)
            except Exception as exc:
                logger.warning("[StreamingEngine] Speaker identify error: %s", exc)

        self.text_queue.put({"text": text, "speaker": speaker, "angle": angle})
