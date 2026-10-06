"""P5 Emphasis Choreography: emphasis words drive sound and motion together.

Firecrawl research (2026-10-05): sound design and zoom accents are retention
tools only when they punctuate meaning - a ding on every word is noise. P3
decides which words deserve emphasis; this module schedules the punch: impact
SFX at the top emphasis moments (numbers first) and push-in zooms on the shots
that contain them, so the ear and the eye hit the same word.

The renderer already accepts ``sfx_cues`` and per-shot ``ShotPlan.motion``
(push_in/pull_out/drift), so the choreography is adoptable without new render
capabilities. Pure functions, unit-tested.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from src.caption_emphasis import is_emphasis_word

_NUMERIC = re.compile(r"^\$?\d[\d,.]*%?$")

# Punctuating sound: a handful per clip, spaced, never in the first breath.
SFX_MAX_CUES = 4
SFX_MIN_GAP_S = 4.0
SFX_MIN_START_S = 1.0
PUSH_IN_MAX_SHOTS = 2


@dataclass
class Choreography:
    sfx_cues: List[Tuple[float, str]] = field(default_factory=list)
    push_in_shots: List[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sfx_cues": [(round(t, 3), kind) for t, kind in self.sfx_cues],
            "push_in_shots": list(self.push_in_shots),
        }


def _numeric(word: str) -> bool:
    return bool(_NUMERIC.match(word.strip().strip("$,.%!?,")))


def plan_choreography(
    words: Sequence[str],
    word_times: Sequence[Tuple[float, float]],
    shot_spans: Sequence[Tuple[float, float]],
    max_sfx: int = SFX_MAX_CUES,
    min_gap_s: float = SFX_MIN_GAP_S,
    min_start_s: float = SFX_MIN_START_S,
    max_push_in_shots: int = PUSH_IN_MAX_SHOTS,
) -> Choreography:
    """Impact cues + punch shots for one clip.

    Candidates are the emphasised words, ranked numbers-first and then by
    time; accepted cues respect a minimum spacing and never fire inside the
    first second. Each accepted cue pulls its shot into a push-in (unique
    shots, capped), because a ding on a static wide shot reads as noise.
    """
    candidates: List[Tuple[int, int, float]] = []
    for index, word in enumerate(words):
        if index >= len(word_times) or not is_emphasis_word(word):
            continue
        start = float(word_times[index][0])
        rank = 0 if _numeric(word) else 1
        candidates.append((rank, index, start))
    candidates.sort(key=lambda item: (item[0], item[2]))

    choreography = Choreography()
    for _rank, _index, start in candidates:
        if len(choreography.sfx_cues) >= max_sfx:
            break
        if start < min_start_s:
            continue
        if any(abs(start - kept) < min_gap_s for kept, _kind in choreography.sfx_cues):
            continue
        choreography.sfx_cues.append((start, "ding"))
        for shot_index, (shot_start, shot_end) in enumerate(shot_spans):
            if shot_start <= start < shot_end:
                if (
                    shot_index not in choreography.push_in_shots
                    and len(choreography.push_in_shots) < max_push_in_shots
                ):
                    choreography.push_in_shots.append(shot_index)
                break
    choreography.sfx_cues.sort(key=lambda cue: cue[0])
    choreography.push_in_shots.sort()
    return choreography
