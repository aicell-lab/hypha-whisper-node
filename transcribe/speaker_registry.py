"""
transcribe/speaker_registry.py — Per-session dynamic speaker identification.

Uses voice embeddings from resemblyzer + optional DOA angle hint to identify
and label individual speakers in a session as "Speaker 1", "Speaker 2", etc.

Requires resemblyzer: pip install --user resemblyzer --no-deps
(--no-deps prevents numpy upgrade from 1.26.4)

If resemblyzer is not installed, all calls to identify() return "Speaker 1"
without crashing — the system degrades gracefully to single-speaker mode.

Algorithm:
  1. Extract 256-dim L2-normalised voice embedding via resemblyzer GE2E model.
  2. Compute cosine similarity to all known speaker embeddings.
  3. If best match >= HIGH_THRESH (0.75): same speaker.
  4. If best match in [LOW_THRESH, HIGH_THRESH) (0.65–0.75): ambiguous —
     use DOA angle difference as tiebreaker (> ANGLE_DIFF_DEG → new speaker).
  5. Otherwise: register as new speaker.

DOA hint is optional — if angle is None, the angle tiebreaker is skipped and
the embedding decision alone determines the speaker label.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

HIGH_THRESH     = 0.75   # cosine similarity → same speaker (confident)
LOW_THRESH      = 0.65   # below this → new speaker
ANGLE_DIFF_DEG  = 60     # degrees difference → treat as different speaker in ambiguous zone
_DOA_HISTORY    = 10     # rolling window for per-speaker DOA median

try:
    from resemblyzer import VoiceEncoder
    _ENCODER_AVAILABLE = True
except ImportError:
    _ENCODER_AVAILABLE = False
    logger.warning("[SpeakerRegistry] resemblyzer not installed — speaker ID disabled (all speech → 'Speaker 1')")


class SpeakerRegistry:
    """Track and identify speakers dynamically within a session.

    Usage:
        registry = SpeakerRegistry()
        speaker_id = registry.identify(audio_chunk, sample_rate=16000, doa_angle=42)
        registry.reset()   # call on new session
    """

    def __init__(self):
        self._encoder: Optional["VoiceEncoder"] = None
        self._enabled = _ENCODER_AVAILABLE
        if self._enabled:
            try:
                # Force CPU to avoid GPU contention with Whisper which uses CUDA.
                # CPU inference for the GE2E encoder is fast enough (~20-50 ms per
                # utterance on Jetson CPU) and doesn't interfere with Whisper's
                # CUDA operations running concurrently.
                self._encoder = VoiceEncoder(device="cpu")
                logger.info("[SpeakerRegistry] VoiceEncoder loaded (CPU)")
                # Pre-warm: first embed_utterance call may be slower due to
                # internal state initialisation. Run it now.
                _dummy = np.zeros(16000 * 3, dtype=np.float32)
                self._encoder.embed_utterance(_dummy)
                logger.info("[SpeakerRegistry] VoiceEncoder warmup complete")
            except Exception as exc:
                logger.warning("[SpeakerRegistry] VoiceEncoder failed to load (%s) — disabled", exc)
                self._enabled = False

        # List of dicts: {label, embedding, doa_angles}
        self._speakers: list = []

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear registry for a new session."""
        self._speakers.clear()
        logger.info("[SpeakerRegistry] Session reset")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def identify(
        self,
        audio_chunk: np.ndarray,
        sample_rate: int = 16000,
        doa_angle: Optional[int] = None,
    ) -> str:
        """Identify the speaker in *audio_chunk*.

        Args:
            audio_chunk: float32 mono array at *sample_rate* Hz.
            sample_rate: audio sample rate (must be 16000 for resemblyzer).
            doa_angle:   DOA angle in degrees [0, 359] or None.

        Returns:
            Speaker label string, e.g. "Speaker 1".
        """
        if not self._enabled or self._encoder is None:
            return "Speaker 1"

        # resemblyzer needs at least ~1.6 s of audio; short chunks get skipped
        min_samples = int(sample_rate * 1.6)
        if len(audio_chunk) < min_samples:
            logger.debug("[SpeakerRegistry] Audio too short (%d samples), skipping embedding", len(audio_chunk))
            if self._speakers:
                return self._speakers[-1]["label"]
            return self._register_new(None, doa_angle)

        try:
            embedding = self._encoder.embed_utterance(audio_chunk)
        except Exception as exc:
            logger.warning("[SpeakerRegistry] Embedding failed: %s", exc)
            return "Speaker 1"

        return self._match_or_register(embedding, doa_angle)

    # ------------------------------------------------------------------
    # Internal matching
    # ------------------------------------------------------------------

    def _match_or_register(self, embedding: np.ndarray, doa_angle: Optional[int]) -> str:
        if not self._speakers:
            return self._register_new(embedding, doa_angle)

        # Compute cosine similarities to all known speakers
        sims = [_cosine_sim(embedding, sp["embedding"]) for sp in self._speakers]
        best_idx = int(np.argmax(sims))
        best_sim = sims[best_idx]

        if best_sim >= HIGH_THRESH:
            # Confident match
            self._update_doa(best_idx, doa_angle)
            return self._speakers[best_idx]["label"]

        if best_sim >= LOW_THRESH:
            # Ambiguous — use DOA as tiebreaker if available
            if doa_angle is not None:
                best_doa = self._speakers[best_idx].get("doa_median")
                if best_doa is not None:
                    diff = _angle_diff(doa_angle, best_doa)
                    if diff > ANGLE_DIFF_DEG:
                        logger.debug(
                            "[SpeakerRegistry] Ambiguous sim=%.3f but DOA diff=%.0f° > %d° → new speaker",
                            best_sim, diff, ANGLE_DIFF_DEG,
                        )
                        return self._register_new(embedding, doa_angle)
            # No DOA or angle close → accept best match
            self._update_doa(best_idx, doa_angle)
            return self._speakers[best_idx]["label"]

        # Below low threshold → new speaker
        return self._register_new(embedding, doa_angle)

    def _register_new(self, embedding: Optional[np.ndarray], doa_angle: Optional[int]) -> str:
        n = len(self._speakers) + 1
        label = f"Speaker {n}"
        entry = {
            "label": label,
            "embedding": embedding,
            "doa_angles": [doa_angle] if doa_angle is not None else [],
            "doa_median": doa_angle,
        }
        self._speakers.append(entry)
        logger.info(
            "[SpeakerRegistry] Registered %s (DOA=%s, total=%d)",
            label, doa_angle, len(self._speakers),
        )
        return label

    def _update_doa(self, idx: int, doa_angle: Optional[int]) -> None:
        if doa_angle is None:
            return
        entry = self._speakers[idx]
        entry["doa_angles"].append(doa_angle)
        # Keep rolling window
        if len(entry["doa_angles"]) > _DOA_HISTORY:
            entry["doa_angles"] = entry["doa_angles"][-_DOA_HISTORY:]
        entry["doa_median"] = int(np.median(entry["doa_angles"]))

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def speaker_count(self) -> int:
        return len(self._speakers)

    def speaker_labels(self) -> list:
        return [sp["label"] for sp in self._speakers]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    if a is None or b is None:
        return 0.0
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _angle_diff(a: int, b: int) -> float:
    """Smallest angular difference between two DOA angles (degrees, circular)."""
    diff = abs(a - b) % 360
    return float(min(diff, 360 - diff))
