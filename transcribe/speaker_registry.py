"""
transcribe/speaker_registry.py — Direction-based speaker identification.

Uses the ReSpeaker USB DOA angle to track speakers within a session.
No voice embeddings — purely directional.

Algorithm:
  1. First utterance → register first speaker at that angle.
  2. Find the existing speaker with the smallest angular distance.
  3. If diff <= DOA_MATCH_DEG: same speaker; continue.
  4. If diff > DOA_MATCH_DEG: new speaker at the new angle.

Labels are the direction angle, e.g. "45°".

If DOA is unavailable (angle=None): assign to last active speaker.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

DOA_MATCH_DEG = 30  # angular diff <= this → same speaker


class SpeakerRegistry:
    """Track speakers by DOA angle within a session.

    Usage:
        registry = SpeakerRegistry()
        label = registry.identify(doa_angle=42)
        registry.reset()
    """

    def __init__(self):
        # List of dicts: {label, angle}
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
            Speaker label string, e.g. "45°".
        """
        if doa_angle is None:
            return self._last_label or "?"

        if not self._speakers:
            label = self._register_new(doa_angle)
            self._last_label = label
            return label

        # Find existing speaker with smallest angular distance.
        closest_idx, closest_diff = min(
            ((i, _angle_diff(doa_angle, sp["angle"])) for i, sp in enumerate(self._speakers)),
            key=lambda x: x[1],
        )

        if closest_diff <= DOA_MATCH_DEG:
            label = self._speakers[closest_idx]["label"]
            logger.debug("[SpeakerRegistry] match: %s (diff=%.0f°)", label, closest_diff)
        else:
            label = self._register_new(doa_angle)
            logger.debug(
                "[SpeakerRegistry] new speaker at %d° (closest diff=%.0f° > %d°)",
                doa_angle, closest_diff, DOA_MATCH_DEG,
            )

        self._last_label = label
        return label

    def _register_new(self, doa_angle: int) -> str:
        label = f"{doa_angle}°"
        self._speakers.append({"label": label, "angle": doa_angle})
        logger.info(
            "[SpeakerRegistry] Registered %s (total=%d)",
            label, len(self._speakers),
        )
        return label

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
