"""Bounded render-repair policy.

Editorial QA judges a finished render, so every issue it raises used to be fatal:
the clip was discarded and, when it was the only clip, the whole run aborted.
This module turns a QA veto into an *actionable* instruction for the next render
attempt, so the renderer can fix what QA just complained about instead of the run
dying on first contact.

The policy is deliberately small and deterministic:

* Only issues with a known render-side remedy are repaired. A fatal probe failure
  or a missing audio stream will never be fixed by re-rendering, so retrying it
  would just burn runner minutes.
* Repairs are cumulative and strictly escalate, so attempt N+1 is always a
  strictly stronger version of attempt N rather than a random retry.
* The attempt count is hard-capped, so a clip that cannot be satisfied still
  terminates and the run still fails loudly rather than looping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# One initial render plus this many repairs. Three renders of a ~60s clip is
# still comfortably inside the GitHub Actions 6 hour job ceiling.
MAX_REPAIR_ATTEMPTS = 2

# motion_gain is a multiplier on the static-shot drift ratio. The cap keeps the
# sweep at a plausible camera move rather than a visible wobble.
MIN_MOTION_GAIN = 1.0
MAX_MOTION_GAIN = 3.6
MOTION_GAIN_STEP = 1.8
# Never let the drift exceed this fraction of the crop width, whatever the gain.
MAX_DRIFT_RATIO = 0.20

# code -> the render-side lever that can plausibly address it
#
# `freeze_interval` is deliberately absent. The renderer's static-shot motion
# guarantee (video_editor._drift_axis_expr) already prevents frozen output, and
# was verified against the real CI artifact plus a sweep of every crop position,
# so escalating gain on a freeze verdict only masked a regression in that
# guarantee. If a freeze ever reaches QA again it should fail the run loudly
# rather than be papered over by a second motion mechanism.
REPAIRABLE_CODES: Dict[str, str] = {
    "static_hold_limit": "motion",
    "black_interval": "motion",
    "caption_source_collision": "captions",
}

# Codes that indicate a broken render rather than a fixable editorial decision.
UNREPAIRABLE_CODES = frozenset({
    "output_probe",
    "video_stream_missing",
    "audio_stream_missing",
    "video_codec",
    "audio_stream",
    "composition_missing",
    "output_duration",
    # A freeze is unrepairable by design: the renderer's motion guarantee is the
    # real fix, so a freeze verdict must surface as a regression.
    "freeze_interval",
})


@dataclass
class RenderAdjustment:
    """Levers the renderer accepts to satisfy a QA complaint."""

    motion_gain: float = MIN_MOTION_GAIN
    avoid_caption_collisions: bool = False

    def drift_ratio(self, base_ratio: float) -> float:
        return min(MAX_DRIFT_RATIO, base_ratio * self.motion_gain)

    def to_dict(self) -> Dict[str, object]:
        return {
            "motion_gain": round(self.motion_gain, 3),
            "avoid_caption_collisions": self.avoid_caption_collisions,
        }


@dataclass
class RepairOutcome:
    succeeded: bool
    attempts: int
    # Whatever attempt_fn returned for the final attempt (normally the output path).
    result: object = None
    final_codes: List[str] = field(default_factory=list)
    adjustment: RenderAdjustment = field(default_factory=RenderAdjustment)
    # One entry per repair: the codes that triggered it and what changed.
    history: List[Dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "succeeded": self.succeeded,
            "attempts": self.attempts,
            "final_codes": self.final_codes,
            "adjustment": self.adjustment.to_dict(),
            "history": self.history,
        }


def plan_repair(
    issue_codes: Sequence[str],
    current: RenderAdjustment,
) -> Tuple[RenderAdjustment, List[str]]:
    """
    Return the next adjustment plus the list of levers actually moved.

    An empty action list means nothing here is repairable and the caller should
    stop rather than spend another render.
    """
    motion_gain = current.motion_gain
    avoid_captions = current.avoid_caption_collisions
    actions: List[str] = []

    for code in issue_codes:
        lever = REPAIRABLE_CODES.get(code)
        if lever is None:
            continue
        if lever == "motion":
            if motion_gain < MAX_MOTION_GAIN:
                motion_gain = min(MAX_MOTION_GAIN, max(motion_gain, MIN_MOTION_GAIN) * MOTION_GAIN_STEP)
                actions.append(f"motion_gain->{motion_gain:.2f}")
        elif lever == "captions":
            if not avoid_captions:
                avoid_captions = True
                actions.append("avoid_caption_collisions->True")

    if not actions:
        return current, []
    return RenderAdjustment(
        motion_gain=round(motion_gain, 3),
        avoid_caption_collisions=avoid_captions,
    ), actions


def is_repairable(issue_codes: Sequence[str]) -> bool:
    return any(code in REPAIRABLE_CODES for code in issue_codes)


def render_with_repair(
    attempt_fn: Callable[[RenderAdjustment], object],
    qa_fn: Callable[[object], Tuple[bool, Sequence[str]]],
    max_attempts: int = MAX_REPAIR_ATTEMPTS,
    log: Optional[Callable[[str], None]] = print,
) -> RepairOutcome:
    """
    Render, judge, and escalate on failure until the clip passes or attempts run out.

    attempt_fn receives the current RenderAdjustment and returns the rendered path.
    qa_fn receives that path and returns (passed, issue_codes).
    Exceptions raised by attempt_fn are intentionally *not* caught: an ffmpeg
    failure is deterministic, and silently retrying it would hide a real bug.
    """
    if max_attempts < 0:
        raise ValueError("max_attempts must be >= 0")

    adjustment = RenderAdjustment()
    history: List[Dict[str, object]] = []
    attempts = 0
    codes: List[str] = []
    result: object = None

    for attempt in range(max_attempts + 1):
        attempts = attempt + 1
        result = attempt_fn(adjustment)
        passed, raw_codes = qa_fn(result)
        codes = list(raw_codes or [])
        if passed:
            return RepairOutcome(
                succeeded=True,
                attempts=attempts,
                result=result,
                final_codes=[],
                adjustment=adjustment,
                history=history,
            )

        if attempt >= max_attempts:
            break

        next_adjustment, actions = plan_repair(codes, adjustment)
        history.append({
            "attempt": attempts,
            "issue_codes": codes,
            "actions": actions,
        })
        if not actions:
            if log:
                blocked = [c for c in codes if c in UNREPAIRABLE_CODES]
                log(
                    f"[!] QA issues are not repairable by re-rendering: {codes}"
                    + (f" (fatal: {blocked})" if blocked else "")
                )
            break
        if log:
            log(f"[!] QA veto on attempt {attempts} ({codes}); repairing via {actions}")
        adjustment = next_adjustment

    return RepairOutcome(
        succeeded=False,
        attempts=attempts,
        result=result,
        final_codes=codes,
        adjustment=adjustment,
        history=history,
    )
