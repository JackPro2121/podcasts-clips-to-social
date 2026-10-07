"""P2 Hook Engineering: the opening 3 seconds, measured and deliberate.

Firecrawl research (2026-10-05): "50-60% of viewers drop off within the first
three seconds - your hook is the most important edit" (KometMedia pacing
guide); every viral-podcast workflow leads with the hook. Our clips currently
start at the technical moment boundary, so the opening can be a mid-breath
cut with no context.

This engine picks where the clip should actually start: a short lead-in to the
beginning of the spoken sentence that precedes the moment, snapped to a word
onset (never mid-breath, never in silence), and extracts the first caption
line from the same words. Pure functions, unit-tested with constructed words;
the render path adopts the returned start through the existing moment
start/end plumbing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

# Do not reach further back than this: a longer lead-in spends watch time
# before the hook. Between 0.5 and 1.6s is the researched sweet spot for a
# context line ("so I told my wife...").
HOOK_MAX_LEAD_IN_S = 1.6
# The first caption line should read as one breath: cap it and prefer to end
# on punctuation when one exists inside the cap.
HOOK_LINE_MAX_CHARS = 42


# The pixel verdict's duration contract is 30-55s. A hook lead-in extends the
# clip backwards, so keep headroom below the ceiling rather than discover the
# overage at the verdict.
HOOK_MAX_CLIP_S = 53.5


def can_extend_with_lead_in(
    clip_duration_s: float, lead_in_s: float, max_clip_s: float = HOOK_MAX_CLIP_S
) -> bool:
    """True when the lead-in keeps the clip inside the duration contract.

    The call-site refuses the lead-in otherwise - a hook is never worth a
    duration block (the render window, caption base and shadow window all move
    together by exactly ``lead_in_s``).
    """
    if lead_in_s <= 0.0:
        return True
    return float(clip_duration_s) + float(lead_in_s) <= float(max_clip_s)


@dataclass
class HookPlan:
    start_s: float = 0.0
    lead_in_s: float = 0.0
    first_line: str = ""

    def as_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 3),
            "lead_in_s": round(self.lead_in_s, 3),
            "first_line": self.first_line,
        }


def hook_start(
    preceding_words: Sequence[Tuple[str, float, float]],
    moment_start_s: float,
    max_lead_in_s: float = HOOK_MAX_LEAD_IN_S,
) -> float:
    """The clip start: earliest word onset within the lead-in window.

    ``preceding_words`` are ``(word, start_s, end_s)`` in clip-relative time,
    sorted; only words before the moment matter. A word onset is always
    chosen, so the first frame is a spoken word - never silence, never a
    mid-breath cut. When no word starts inside the window (a pause precedes
    the moment), the moment's own start is kept.
    """
    best = float(moment_start_s)
    window = float(moment_start_s) - float(max_lead_in_s)
    for _word, start, end in preceding_words:
        if end <= window or start >= moment_start_s:
            continue
        if start < best:
            best = float(start)
    return best


def first_line(
    words: Sequence[Tuple[str, float, float]],
    start_s: float,
    max_chars: int = HOOK_LINE_MAX_CHARS,
) -> str:
    """The first caption line: words from ``start_s`` until the char cap.

    Ends on punctuation (., !, ? ,) when one falls inside the cap, because a
    hook line that clips mid-thought reads as broken; otherwise the last
    whole word that fits.
    """
    selected: List[str] = []
    length = 0
    for word, word_start, _end in words:
        if word_start < start_s:
            continue
        addition = len(word) + (1 if selected else 0)
        if selected and length + addition > max_chars:
            break
        selected.append(word)
        length += addition
    if not selected:
        return ""
    line = " ".join(selected)
    for index in range(len(selected) - 1, 0, -1):
        if selected[index][-1] in ".!?,":
            return " ".join(selected[: index + 1])
    return line


def build_hook_plan(
    preceding_words: Sequence[Tuple[str, float, float]],
    moment_words: Sequence[Tuple[str, float, float]],
    moment_start_s: float,
    max_lead_in_s: float = HOOK_MAX_LEAD_IN_S,
) -> HookPlan:
    """Start + lead-in + first line for one moment."""
    start = hook_start(preceding_words, moment_start_s, max_lead_in_s)
    combined = list(preceding_words) + list(moment_words)
    combined.sort(key=lambda item: item[1])
    return HookPlan(
        start_s=start,
        lead_in_s=max(0.0, float(moment_start_s) - start),
        first_line=first_line(combined, start),
    )
