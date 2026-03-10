"""
transcribe/speaker_registry.py — DOA-only speaker identification.

Uses the ReSpeaker USB DOA angle to track speakers within a session.
No voice embeddings — purely directional.

Algorithm:
  1. First utterance → register Speaker 1.
  2. Find the existing speaker with the smallest angular distance to the
     current DOA angle.
  3. If closest diff <= DOA_MATCH_DEG: assign to that speaker.
  4. If closest diff > DOA_EXCL_DEG AND all known speakers have trusted DOA
     (>= DOA_MIN_READINGS): register a new speaker from a distinct direction.
  5. Otherwise (ambiguous / no trusted DOA yet): assign to closest.

If DOA is unavailable (angle=None): always assign to the last active speaker.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DOA_MATCH_DEG    = 30    # diff <= this → same speaker
DOA_EXCL_DEG     = 50    # diff > this (with trusted DOA) → new speaker
DOA_MIN_READINGS = 1     # readings needed before a speaker's DOA is trusted
_DOA_HISTORY     = 10    # rolling window for per-speaker DOA median


class SpeakerRegistry:
    """Track speakers by DOA angle within a session.

    Usage:
        registry = SpeakerRegistry()
        label = registry.identify(doa_angle=42)
        registry.reset()
    """

    def __init__(self):
        # List of dicts: {label, doa_angles, doa_median}
        self._speakers: list = []
        self._last_label: Optional[str] = None

    def reset(self) -> None:
        """Clear registry for a new session."""
        self._speakers.clear()
        self._last_label = None
        logger.info("[SpeakerRegistry] Session reset")

    def identify(
        self,
        audio_chunk=None,    # accepted but ignored (kept for call-site compatibility)
        sample_rate: int = 16000,
        doa_angle: Optional[int] = None,
    ) -> str:
        """Identify the speaker direction.

        Args:
            doa_angle: DOA angle in degrees [0, 359] or None.

        Returns:
            Speaker label string, e.g. "Speaker 1".
        """
        if doa_angle is None:
            # No DOA — return last active speaker or default
            return self._last_label or self._register_new(None)

        if not self._speakers:
            label = self._register_new(doa_angle)
            self._last_label = label
            return label

        label = self._match_or_register(doa_angle)
        self._last_label = label
        return label

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match_or_register(self, doa_angle: int) -> str:
        # Find closest speaker by angular distance.
        distances = [
            (idx, _angle_diff(doa_angle, sp["doa_median"]))
            for idx, sp in enumerate(self._speakers)
        ]
        distances.sort(key=lambda x: x[1])
        closest_idx, closest_diff = distances[0]

        if closest_diff <= DOA_MATCH_DEG:
            self._update_doa(closest_idx, doa_angle)
            sp = self._speakers[closest_idx]
            logger.debug(
                "[SpeakerRegistry] DOA match: %s (diff=%.0f°)",
                sp["label"], closest_diff,
            )
            return sp["label"]

        # Check if the closest speaker has a trusted DOA median.
        closest_trusted = len(self._speakers[closest_idx]["doa_angles"]) >= DOA_MIN_READINGS
        if closest_trusted and closest_diff > DOA_EXCL_DEG:
            logger.debug(
                "[SpeakerRegistry] DOA exclusion: closest speaker far (%.0f° > %d°) → new speaker",
                closest_diff, DOA_EXCL_DEG,
            )
            return self._register_new(doa_angle)

        # Ambiguous zone or not enough history — assign to closest.
        self._update_doa(closest_idx, doa_angle)
        sp = self._speakers[closest_idx]
        logger.debug(
            "[SpeakerRegistry] Ambiguous DOA (diff=%.0f°) — assigned to closest %s",
            closest_diff, sp["label"],
        )
        return sp["label"]

    def _register_new(self, doa_angle: Optional[int]) -> str:
        n = len(self._speakers) + 1
        label = f"Speaker {n}"
        entry = {
            "label": label,
            "doa_angles": [doa_angle] if doa_angle is not None else [],
            "doa_median": doa_angle if doa_angle is not None else 0,
        }
        self._speakers.append(entry)
        logger.info(
            "[SpeakerRegistry] Registered %s (DOA=%s, total=%d)",
            label, doa_angle, len(self._speakers),
        )
        return label

    def _update_doa(self, idx: int, doa_angle: int) -> None:
        entry = self._speakers[idx]
        entry["doa_angles"].append(doa_angle)
        if len(entry["doa_angles"]) > _DOA_HISTORY:
            entry["doa_angles"] = entry["doa_angles"][-_DOA_HISTORY:]
        entry["doa_median"] = int(np.median(entry["doa_angles"]))

    def speaker_count(self) -> int:
        return len(self._speakers)

    def speaker_labels(self) -> list:
        return [sp["label"] for sp in self._speakers]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _angle_diff(a: int, b: int) -> float:
    """Smallest angular difference between two DOA angles (degrees, circular)."""
    diff = abs(a - b) % 360
    return float(min(diff, 360 - diff))
