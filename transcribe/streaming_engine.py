"""
transcribe/streaming_engine.py — Streaming Whisper engine using LocalAgreement.

Wraps whisper_streaming's OnlineASRProcessor / VACOnlineASRProcessor.
Usage:
    engine = StreamingEngine(model_name="base.en")
    engine.init_session()          # called when SSE client connects
    text = engine.process_audio(chunk)   # np.float32, 16 kHz, any size
    engine.finish_session()        # called when SSE client disconnects

text_queue items are dicts: {"text": str, "speaker": str, "angle": int|None}
when DOAReader / SpeakerRegistry are available, otherwise plain str for
backwards compatibility.
"""

import concurrent.futures
import logging
import queue
import sys
import os
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "base.en"


def _try_import_doa():
    """Lazy import of DOAReader and DOAIntervalBuffer — returns classes or None."""
    try:
        # Ensure project root is in sys.path for 'audio' package
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from audio.doa_reader import DOAReader, DOAIntervalBuffer
        return DOAReader, DOAIntervalBuffer
    except Exception as e:
        logger.debug("[StreamingEngine] DOAReader not available: %s", e)
        return None, None


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
        text_queue: Queue of committed transcript items. Each item is either:
            - dict: {"text": str, "speaker": str, "angle": int|None}  (when speaker ID is active)
            - str:  plain committed text  (legacy/fallback)
          Consumed by HyphaClient's sse_gen().
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        language: str = "en",
        use_vac: bool = True,
        chunk_seconds: float = 0.5,
        buffer_trimming_sec: float = 8.0,
        enable_doa: bool = True,
        enable_speaker_id: bool = True,
    ):
        self.model_name = model_name
        self.text_queue: queue.Queue = queue.Queue()
        self._finished = False
        
        # Session timing for DOA timestamp conversion
        # process_iter() returns begin/end relative to this time
        self._session_start_time = time.monotonic()

        # DOA: Read from ReSpeaker firmware via USB (XMOS XVF-3000 built-in DOA)
        self._doa: Optional[object] = None  # DOAReader instance
        self._doa_buffer: Optional[object] = None  # DOAIntervalBuffer for duration-weighted lookup
        
        if enable_doa:
            _DOAReader, _DOAIntervalBuffer = _try_import_doa()
            if _DOAReader is not None and _DOAIntervalBuffer is not None:
                try:
                    # DOAReader polls DOA from firmware via USB
                    poll_interval = 0.05  # 50ms polling
                    self._doa = _DOAReader(poll_interval=poll_interval)
                    self._doa.start()
                    # DOAIntervalBuffer stores DOA as intervals for duration-weighted lookup
                    # This is the key fix: we calculate which angle had longest overlap
                    self._doa_buffer = _DOAIntervalBuffer(
                        maxlen=200,  # ~10s of history
                        poll_interval=poll_interval
                    )
                    logger.info("[StreamingEngine] ReSpeaker firmware DOA initialized (duration-weighted)")
                except Exception as exc:
                    logger.warning("[StreamingEngine] DOA init failed: %s", exc)
                    self._doa = None
                    self._doa_buffer = None

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

        # Audio accumulator for speaker embedding (collects audio between commits)
        self._audio_since_last_commit: list = []
        self._last_commit_time: float = 0.0

        # Thread pool for async speaker identification (1 worker to avoid concurrent CUDA calls)
        self._speaker_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
        if self._speaker_registry is not None:
            self._speaker_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="speaker-id"
            )

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
        self._audio_since_last_commit = []
        self._last_commit_time = time.monotonic()
        # Track session start time for DOA timestamp conversion
        # process_iter() returns begin/end relative to this time
        self._session_start_time = time.monotonic()
        # Clear DOA buffer for new session
        if self._doa_buffer is not None:
            self._doa_buffer.clear()
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
                    audio_snapshot = list(self._audio_since_last_commit)
                    self._audio_since_last_commit = []
                    commit_time = self._last_commit_time
                    self._emit_item_async(text, audio_snapshot, commit_time)
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
            audio_snapshot = list(self._audio_since_last_commit)
            self._audio_since_last_commit = []
            commit_time = self._last_commit_time
            self._emit_item_async(text, audio_snapshot, commit_time)
            logger.info("[StreamingEngine] Flushed final: %r", text)

        # Wait for any in-flight speaker ID tasks to complete and push their items
        if self._speaker_executor is not None:
            self._speaker_executor.shutdown(wait=True, cancel_futures=False)
            # Re-create executor for next session
            self._speaker_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="speaker-id"
            )

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
        # Read DOA from firmware at capture time (not emit time) to avoid 
        # misattribution when speakers change during the buffering period.
        # The ReSpeaker firmware's built-in DOA algorithm processes raw mics on-chip.
        if self._doa is not None and self._doa.enabled and self._doa_buffer is not None:
            try:
                # Read current DOA from firmware via USB
                angle = self._doa.read()
                if angle is not None:
                    self._doa_buffer.add(angle)
            except Exception as exc:
                logger.debug("[StreamingEngine] DOA read error: %s", exc)
        
        # Accumulate audio for speaker embedding
        self._audio_since_last_commit.append(chunk)

        self._online.insert_audio_chunk(chunk)
        try:
            begin, end, text = self._online.process_iter()
        except Exception as exc:
            logger.warning("[StreamingEngine] process_iter raised: %s", exc)
            return None
        text = text.strip() if text else ""
        if text:
            # Snapshot audio buffer and reset before async embedding so subsequent
            # audio is not included in this segment's embedding.
            audio_snapshot = list(self._audio_since_last_commit)
            self._audio_since_last_commit = []
            
            # Convert relative timestamps (begin, end) to absolute monotonic time
            # This allows us to query DOA for the exact time period of this audio segment
            segment_start = self._session_start_time + (begin or 0)
            segment_end = self._session_start_time + (end or 0)
            
            self._emit_item_async(text, audio_snapshot, segment_start, segment_end)
            logger.debug("[StreamingEngine] Committed [%.2f–%.2f] (text not logged)", begin or 0, end or 0)
            return text
        return None

    def _emit_item_async(self, text: str, audio_snapshot: list, 
                         segment_start: float, segment_end: float) -> None:
        """Identify speaker asynchronously and push item to text_queue when done.
        
        Args:
            text: Transcribed text
            audio_snapshot: Audio chunks accumulated since last commit
            segment_start: Absolute monotonic time when this segment started
            segment_end: Absolute monotonic time when this segment ended
        """
        if self._speaker_executor is None:
            # No speaker ID — emit immediately
            self.text_queue.put(self._build_item_sync(text, audio_snapshot, segment_start, segment_end))
            return

        def _task():
            item = self._build_item_sync(text, audio_snapshot, segment_start, segment_end)
            self.text_queue.put(item)

        self._speaker_executor.submit(_task)

    # ------------------------------------------------------------------
    # Speaker identification helpers
    # ------------------------------------------------------------------

    def _build_item_sync(self, text: str, audio_snapshot: list, 
                         segment_start: float, segment_end: float) -> dict:
        """Build a text_queue item with optional speaker ID and DOA angle.

        This may block for embedding computation — runs in the speaker-id
        thread pool when speaker identification is enabled.
        
        Uses duration-weighted DOA lookup: the angle with the longest overlap
        with the transcript's time range is assigned (learned from WhisperX).
        
        Args:
            text: Transcribed text
            audio_snapshot: Audio chunks since last commit
            segment_start: Absolute monotonic time when this segment started
            segment_end: Absolute monotonic time when this segment ended
        """
        speaker = "Speaker 1"
        angle: Optional[int] = None

        # Query DOA using duration-weighted overlap (the fundamental fix).
        # Instead of median/mode, we find which angle had the longest total
        # overlap with the time range of this transcript segment.
        if self._doa_buffer is not None:
            angle = self._doa_buffer.dominant_angle(segment_start, segment_end)
        
        # Fallback: if no overlap found, try most recent DOA reading
        if angle is None and self._doa is not None and self._doa.enabled:
            angle = self._doa.read()

        if self._speaker_registry is not None:
            audio_buf = np.concatenate(audio_snapshot) if audio_snapshot else np.zeros(0, dtype=np.float32)
            try:
                speaker = self._speaker_registry.identify(audio_buf, sample_rate=16000, doa_angle=angle)
            except Exception as exc:
                logger.warning("[StreamingEngine] Speaker identify error: %s", exc)

        return {"text": text, "speaker": speaker, "angle": angle}
