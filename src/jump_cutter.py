"""P1 Jump-Cut Engine: dead-air removal from word timestamps.

Firecrawl research (2026-10-05): removing dead air and filler pauses is the
industry-standard "feels edited" upgrade (TimeBolt, SavvyCut, Jump Cutter
ULTRA all sell exactly this), and 50-60% of short-form viewers drop off in the
first three seconds (KometMedia pacing guide), so pacing is retention. Clips
that keep every natural pause read as "raw excerpt", not "edited".

This module is the pure engine: it turns word timestamps into a keep-segment
list and a compacted timeline (old clip time -> new timeline). The renderer
consumes the plan; the caption layer remaps word times through it. No media
access, so every rule is unit-tested with constructed word lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

# Gaps shorter than this are natural speech rhythm, not dead air.
MIN_DEAD_AIR_S = 0.65
# Breath kept on each side of a speech span so cuts do not clip onsets.
KEEP_PADDING_S = 0.08
# The pixel verdict's contract floor (30-55s). A cut plan that would push the
# clip below the floor is refused outright rather than shipping a short clip.
MIN_RESULT_DURATION_S = 30.0


@dataclass
class CutPlan:
    """Keep-segments in original clip time, plus the removed total."""

    keep_segments: List[Tuple[float, float]] = field(default_factory=list)
    original_duration_s: float = 0.0
    removed_s: float = 0.0
    applied: bool = False

    @property
    def duration_s(self) -> float:
        return sum(end - start for start, end in self.keep_segments)

    def remap(self, time_s: float) -> float:
        """Original clip time -> compacted timeline, snapping removed gaps.

        Times inside a removed gap snap to the following segment's start;
        times past the last segment clamp to the compacted end.
        """
        kept = 0.0
        for start, end in self.keep_segments:
            if time_s < start:
                return kept
            if time_s <= end:
                return kept + (time_s - start)
            kept += end - start
        return kept


def _speech_spans(
    words: Sequence[Tuple[float, float]], min_gap_s: float = MIN_DEAD_AIR_S
) -> List[Tuple[float, float]]:
    """Merge word (start, end) pairs into contiguous speech spans.

    A gap shorter than ``min_gap_s`` is speech rhythm and never cut, so it is
    also the merge threshold.
    """
    spans: List[Tuple[float, float]] = []
    for start, end in sorted(words):
        if end <= start:
            continue
        if spans and start - spans[-1][1] < min_gap_s:
            spans[-1] = (spans[-1][0], max(spans[-1][1], end))
        else:
            spans.append((float(start), float(end)))
    return spans


def build_cut_plan(
    words: Sequence[Tuple[float, float]],
    clip_duration_s: float,
    min_gap_s: float = MIN_DEAD_AIR_S,
    padding_s: float = KEEP_PADDING_S,
    min_duration_s: float = MIN_RESULT_DURATION_S,
) -> CutPlan:
    """Plan the dead-air cuts for one rendered clip's word timings.

    ``words`` are ``(start_s, end_s)`` pairs in clip-relative time. The plan is
    an identity (one keep-segment, ``applied=False``) whenever there is nothing
    worth cutting or a guard refuses: an empty word list, no gap at least
    ``min_gap_s``, or a result that would fall below ``min_duration_s``.
    """
    identity = CutPlan(
        keep_segments=[(0.0, float(clip_duration_s))],
        original_duration_s=float(clip_duration_s),
        removed_s=0.0,
        applied=False,
    )
    if not words or clip_duration_s <= 0.0:
        return identity

    spans = _speech_spans(words, min_gap_s)
    if not spans:
        return identity
    padded = [
        (max(0.0, start - padding_s), min(float(clip_duration_s), end + padding_s))
        for start, end in spans
    ]
    merged: List[Tuple[float, float]] = []
    for start, end in padded:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Keep the head/tail when the first/last span essentially starts/ends with
    # the clip: only a real dead-air run at the edge is worth removing.
    if merged[0][0] <= min_gap_s:
        merged[0] = (0.0, merged[0][1])
    if float(clip_duration_s) - merged[-1][1] <= min_gap_s:
        merged[-1] = (merged[-1][0], float(clip_duration_s))

    removed = float(clip_duration_s) - sum(end - start for start, end in merged)
    if removed <= 0.0:
        return identity
    if float(clip_duration_s) - removed < min_duration_s:
        return identity
    return CutPlan(
        keep_segments=merged,
        original_duration_s=float(clip_duration_s),
        removed_s=removed,
        applied=True,
    )
