"""P4 Edit cues: b-roll windows and transition sound, planned from the edit.

Firecrawl research (2026-10-05): b-roll, transitions and sound design are the
standard short-form kit (OpusClip/Vizard/Submagic feature comparisons all lead
with them). The renderer has accepted ``broll_cues`` and ``sfx_cues`` for a
long time - but nothing ever generated them: ``broll_count`` was 0 in every
run and whooshes did not follow the cuts.

This module is the planner. It consumes the clip's speech spans and the P1
cut plan and returns cue times; asset resolution (Pexels b-roll, whoosh.wav)
stays with the renderer's existing plumbing. Pure functions, unit-tested.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from src.jump_cutter import CutPlan

# The hook must never be covered: retention research puts the entire opening
# decision in the first three seconds.
HOOK_GUARD_S = 3.0
BROLL_MIN_S = 2.5
BROLL_MAX_S = 4.0
BROLL_MAX_CUES = 2
# Two b-roll windows on top of the speaker within seconds of each other read
# as a montage; space them so the speaker remains the primary image.
BROLL_MIN_GAP_S = 8.0
WHOOSH_MAX_CUES = 6


def plan_broll_cues(
    speech_spans: Sequence[Tuple[float, float]],
    duration_s: float,
    max_cues: int = BROLL_MAX_CUES,
    min_start_s: float = HOOK_GUARD_S,
    min_gap_s: float = BROLL_MIN_GAP_S,
) -> List[Tuple[float, float]]:
    """B-roll windows ``(start, end)``: longest eligible spans, spaced out.

    Only spans at least ``BROLL_MIN_S`` long (after the hook guard) qualify;
    each window is capped at ``BROLL_MAX_S`` so b-roll accents rather than
    replaces the speaker. Returns at most ``max_cues`` windows, in time order.
    """
    candidates: List[Tuple[float, float]] = []
    for start, end in speech_spans:
        clipped_start = max(float(start), min_start_s)
        clipped_end = min(float(end), float(duration_s))
        if clipped_end - clipped_start >= BROLL_MIN_S:
            candidates.append((clipped_start, min(clipped_end, clipped_start + BROLL_MAX_S)))
    chosen: List[Tuple[float, float]] = []
    for start, end in sorted(candidates, key=lambda span: -(span[1] - span[0])):
        if len(chosen) >= max_cues:
            break
        if all(abs(start - kept_start) >= min_gap_s for kept_start, _ in chosen):
            chosen.append((start, end))
    return sorted(chosen)


def plan_cut_whooshes(
    cut_plan: CutPlan, max_cues: int = WHOOSH_MAX_CUES
) -> List[float]:
    """Whoosh cue times at internal jump-cut boundaries, compacted timeline.

    Boundary ``i`` sits at the sum of the kept durations before it - the same
    arithmetic the concat graph performs, so a whoosh lands exactly on the
    frame where the cut happens. An identity plan produces no cues.
    """
    if not cut_plan.applied:
        return []
    times: List[float] = []
    kept = 0.0
    for index, (start, end) in enumerate(cut_plan.keep_segments):
        if index > 0:
            times.append(round(kept, 3))
        kept += end - start
    return times[:max_cues]
