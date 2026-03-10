"""
tests/test_speaker_registry.py — Unit tests for the simplified
direction-based SpeakerRegistry.

Labels are angle strings like "45°". Only DOA_MATCH_DEG threshold is used.
"""

import pytest
from transcribe.speaker_registry import SpeakerRegistry, DOA_MATCH_DEG, _angle_diff


# ---------------------------------------------------------------------------
# _angle_diff helper
# ---------------------------------------------------------------------------

def test_angle_diff_zero():
    assert _angle_diff(0, 0) == 0.0


def test_angle_diff_straight():
    assert _angle_diff(0, 180) == 180.0


def test_angle_diff_wrap_around():
    # 350° and 10° are 20° apart (wrapping)
    assert _angle_diff(350, 10) == pytest.approx(20.0)


def test_angle_diff_symmetric():
    assert _angle_diff(30, 90) == _angle_diff(90, 30)


# ---------------------------------------------------------------------------
# SpeakerRegistry — core logic
# ---------------------------------------------------------------------------

def test_first_utterance_registers_angle():
    """First call creates a speaker labelled with the angle."""
    reg = SpeakerRegistry()
    label = reg.identify(doa_angle=45)
    assert label == "45°"
    assert reg.speaker_count() == 1


def test_same_angle_same_speaker():
    """Exact same angle → same label."""
    reg = SpeakerRegistry()
    l1 = reg.identify(doa_angle=90)
    l2 = reg.identify(doa_angle=90)
    assert l1 == l2
    assert reg.speaker_count() == 1


def test_within_threshold_same_speaker():
    """Angle within DOA_MATCH_DEG → same speaker."""
    reg = SpeakerRegistry()
    l1 = reg.identify(doa_angle=100)
    # Move by DOA_MATCH_DEG exactly — still same speaker
    l2 = reg.identify(doa_angle=100 + DOA_MATCH_DEG)
    assert l1 == l2
    assert reg.speaker_count() == 1


def test_beyond_threshold_new_speaker():
    """Angle beyond DOA_MATCH_DEG → new speaker registered."""
    reg = SpeakerRegistry()
    l1 = reg.identify(doa_angle=0)
    l2 = reg.identify(doa_angle=DOA_MATCH_DEG + 1)
    assert l1 != l2
    assert reg.speaker_count() == 2


def test_two_distinct_directions():
    """Opposite sides (0° and 180°) always register as separate speakers."""
    reg = SpeakerRegistry()
    l1 = reg.identify(doa_angle=0)
    l2 = reg.identify(doa_angle=180)
    assert l1 != l2
    assert reg.speaker_count() == 2


def test_label_format():
    """Speaker label is the angle followed by the degree symbol."""
    reg = SpeakerRegistry()
    label = reg.identify(doa_angle=270)
    assert label == "270°"


def test_no_doa_returns_last_label():
    """When doa_angle is None, last active label is returned."""
    reg = SpeakerRegistry()
    reg.identify(doa_angle=45)
    label = reg.identify(doa_angle=None)
    assert label == "45°"


def test_no_doa_no_history_returns_question_mark():
    """When doa_angle is None and no speakers registered yet, return '?'."""
    reg = SpeakerRegistry()
    label = reg.identify(doa_angle=None)
    assert label == "?"


def test_reset_clears_registry():
    """reset() wipes all registered speakers."""
    reg = SpeakerRegistry()
    reg.identify(doa_angle=30)
    reg.identify(doa_angle=210)
    assert reg.speaker_count() == 2
    reg.reset()
    assert reg.speaker_count() == 0
    assert reg.speaker_labels() == []


def test_reset_then_new_registration():
    """After reset(), first utterance re-registers from scratch."""
    reg = SpeakerRegistry()
    reg.identify(doa_angle=90)
    reg.reset()
    label = reg.identify(doa_angle=45)
    assert label == "45°"
    assert reg.speaker_count() == 1


def test_three_speakers():
    """Three well-separated angles register as three distinct speakers."""
    reg = SpeakerRegistry()
    l1 = reg.identify(doa_angle=0)
    l2 = reg.identify(doa_angle=120)
    l3 = reg.identify(doa_angle=240)
    assert len({l1, l2, l3}) == 3
    assert reg.speaker_count() == 3


def test_closest_speaker_wins():
    """When two speakers exist, the one with smallest angular diff is matched."""
    reg = SpeakerRegistry()
    reg.identify(doa_angle=0)    # speaker "0°"
    reg.identify(doa_angle=180)  # speaker "180°"
    # Angle 170° is closer to 180° than to 0° (diff=10° vs 170°)
    label = reg.identify(doa_angle=170)
    assert label == "180°"


def test_wrap_around_matching():
    """350° should match a speaker registered at 10° (diff=20°, within threshold)."""
    reg = SpeakerRegistry()
    reg.identify(doa_angle=10)
    label = reg.identify(doa_angle=350)
    assert label == "10°"


def test_speaker_labels_list():
    """speaker_labels() returns all registered labels in registration order."""
    reg = SpeakerRegistry()
    reg.identify(doa_angle=45)
    reg.identify(doa_angle=200)
    reg.identify(doa_angle=300)
    assert reg.speaker_labels() == ["45°", "200°", "300°"]


def test_audio_chunk_arg_ignored():
    """audio_chunk parameter is accepted but ignored (call-site compatibility)."""
    import numpy as np
    reg = SpeakerRegistry()
    label = reg.identify(audio_chunk=np.zeros(1600, dtype=np.float32), doa_angle=90)
    assert label == "90°"
