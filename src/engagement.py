"""Per-window speech-energy analysis for editorial shot selection.

The director previously scored a candidate window from hook text, a coarse visual
score, duration proximity, payoff words and filler ratio. None of those describe
*pacing* -- a real editor also asks how fast the person is talking, how much dead
air is inside the window, whether they pause for effect or lose the thread, and
where the vocal peaks land.

Everything here is derived from data the pipeline already produces (word-level
transcript timestamps) plus stdlib only, so it adds no dependency and no model
download to the constrained GitHub Actions runner.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from src.transcriber import TranscriptSegment

# Disfluencies that signal a speaker losing their thread rather than delivering a
# line. Kept local to this module so it stays independent of edit_director.
FILLER_TOKENS = frozenset({
    "um", "uh", "erm", "hmm", "mhm", "uhh", "umm", "er", "ah", "eh",
})

# A pause at least this long reads as a deliberate beat rather than a breath.
DELIBERATE_PAUSE_SECONDS = 0.7
# Beyond this a pause is dead air and should cost the window score.
DEAD_AIR_SECONDS = 1.6

_WORD_RE = re.compile(r"[A-Za-z']+")
# Strips everything a word-level token may pick up from ASR output (commas,
# hyphens, digits) while keeping the alphabetic core for filler matching.
_NOISE_RE = re.compile(r"[^a-z']+")


def _tokenize(text: str) -> List[str]:
    return [match.group(0).lower() for match in _WORD_RE.finditer(text or "")]


def _clean_token(word: str) -> str:
    return _NOISE_RE.sub("", (word or "").lower())


def _words_in_window(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
) -> List[Tuple[float, str]]:
    """(timestamp, lowercase token) for every word spoken inside the window."""
    words: List[Tuple[float, str]] = []
    for segment in segments:
        if segment.end <= start or segment.start >= end:
            continue
        if segment.words:
            for word in segment.words:
                if word.end > start and word.start < end:
                    words.append((word.start, _clean_token(word.word)))
        else:
            text = segment.text or ""
            tokens = _tokenize(text)
            if not tokens:
                continue
            span = max(1e-6, segment.end - segment.start)
            step = span / len(tokens)
            for index, token in enumerate(tokens):
                position = segment.start + index * step
                if start <= position < end:
                    words.append((position, token))
    words.sort(key=lambda item: item[0])
    return [(position, token) for position, token in words if token]


@dataclass
class EngagementSample:
    """Editorial pacing descriptors for one window of the timeline."""

    start: float
    end: float
    word_count: int = 0
    duration: float = 0.0
    words_per_second: float = 0.0
    speech_ratio: float = 0.0
    longest_pause: float = 0.0
    dead_air: float = 0.0
    filler_ratio: float = 0.0
    emphasis_rate: float = 0.0
    question_rate: float = 0.0
    score: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _emphasis_and_questions(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
) -> Tuple[int, int]:
    """Count emphasised and interrogative utterances overlapping the window."""
    emphasis = 0
    questions = 0
    for segment in segments:
        if segment.end <= start or segment.start >= end:
            continue
        text = segment.text or ""
        if not text.strip():
            continue
        if "!" in text or re.search(r"\b[A-Z]{2,}\b", text):
            emphasis += 1
        if "?" in text:
            questions += 1
    return emphasis, questions


def analyze_window(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
) -> EngagementSample:
    """Measure the pacing of the window [start, end)."""
    duration = max(0.0, end - start)
    sample = EngagementSample(start=round(start, 2), end=round(end, 2), duration=round(duration, 2))
    if duration <= 0.0:
        sample.warnings.append("empty_window")
        return sample

    words = _words_in_window(segments, start, end)
    sample.word_count = len(words)
    if not words:
        sample.warnings.append("no_speech")
        # A window with no speech at all cannot be a hook.
        sample.score = 0.0
        return sample

    sample.words_per_second = round(len(words) / duration, 4)

    # Coverage: how much of the window is actually covered by a spoken word.
    # Word timestamps are the best available proxy for VAD speech spans, which
    # faster-whisper already filters on internally (vad_filter=True).
    covered = 0.0
    longest_pause = 0.0
    previous_end: Optional[float] = None
    fillers = 0
    for position, token in words:
        if previous_end is not None:
            gap = max(0.0, position - previous_end)
            longest_pause = max(longest_pause, gap)
        # A word occupies a nominal slice derived from the local speaking rate.
        estimated_slice = min(0.6, max(0.08, 1.0 / max(0.2, sample.words_per_second)))
        covered += estimated_slice
        previous_end = position + estimated_slice
        if token in FILLER_TOKENS:
            fillers += 1

    sample.speech_ratio = round(min(1.0, covered / duration), 4)
    sample.longest_pause = round(min(longest_pause, duration), 3)
    sample.dead_air = round(max(0.0, longest_pause - DEAD_AIR_SECONDS), 3)
    sample.filler_ratio = round(fillers / max(1, len(words)), 4)

    emphasis, questions = _emphasis_and_questions(segments, start, end)
    sample.emphasis_rate = round(emphasis / duration, 4)
    sample.question_rate = round(questions / duration, 4)

    if longest_pause >= DEAD_AIR_SECONDS:
        sample.warnings.append("dead_air")
    if sample.filler_ratio > 0.08:
        sample.warnings.append("filler_heavy")
    if sample.words_per_second < 0.9:
        sample.warnings.append("slow_delivery")

    sample.score = round(_score_sample(sample), 4)
    return sample


def _score_sample(sample: EngagementSample) -> float:
    """Blend pacing descriptors into a 0..1 editorial energy score."""
    if sample.word_count == 0:
        return 0.0

    # Conversational podcast delivery sits around 2.0-3.2 words/second; both
    # very slow and very fast delivery lose points.
    rate = sample.words_per_second
    if rate <= 0.0:
        rate_score = 0.0
    elif rate < 2.0:
        rate_score = max(0.0, (rate - 0.9) / 1.1)
    elif rate <= 3.2:
        rate_score = 1.0
    else:
        rate_score = max(0.0, 1.0 - (rate - 3.2) / 3.0)

    coverage_score = min(1.0, sample.speech_ratio / 0.75)
    dead_air_penalty = min(0.45, sample.dead_air * 0.18)
    # A deliberate pause inside the window is a bonus, not a defect.
    pause_bonus = 0.05 if DELIBERATE_PAUSE_SECONDS <= sample.longest_pause < DEAD_AIR_SECONDS else 0.0
    filler_penalty = min(0.2, sample.filler_ratio * 2.0)
    emphasis_score = min(1.0, sample.emphasis_rate / 0.25)
    question_score = min(1.0, sample.question_rate / 0.2)

    blended = (
        rate_score * 0.28
        + coverage_score * 0.24
        + emphasis_score * 0.16
        + question_score * 0.12
        + pause_bonus
        - dead_air_penalty
        - filler_penalty
    )
    return max(0.0, min(1.0, blended))


def engagement_curve(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
    bucket_seconds: float = 1.0,
) -> List[EngagementSample]:
    """Time series of engagement across [start, end), for locating energy peaks."""
    if bucket_seconds <= 0.0 or end <= start:
        return []
    samples: List[EngagementSample] = []
    cursor = start
    while cursor < end - 1e-9:
        bucket_end = min(end, cursor + bucket_seconds)
        samples.append(analyze_window(segments, cursor, bucket_end))
        cursor = bucket_end
    return samples


@dataclass
class WindowCandidate:
    """A candidate edit window paired with its measured energy."""

    start: float
    end: float
    label: str = ""
    base_score: float = 0.0
    engagement: EngagementSample = field(default_factory=lambda: EngagementSample(0.0, 0.0))

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def combined_score(self) -> float:
        """
        Blend the caller's own score with measured pacing energy.

        The blend is unconditional so the weighting stays predictable: a caller
        with no independent signal passes base_score=0.0 and simply receives the
        energy term at its 35% weight rather than having it silently promoted.
        """
        return round(self.base_score * 0.65 + self.engagement.score * 0.35, 4)


def rank_candidates(
    candidates: Sequence[Tuple[float, float, str, float]],
    segments: Sequence[TranscriptSegment],
    min_duration: float = 30.0,
    max_duration: float = 60.0,
) -> List[WindowCandidate]:
    """
    Rank (start, end, label, base_score) windows by base score plus pacing energy.

    Windows outside the platform duration window are still measured and returned
    so callers can report *why* a candidate was rejected, but they sort last.
    """
    ranked: List[WindowCandidate] = []
    for start, end, label, base_score in candidates:
        if end <= start:
            continue
        engagement = analyze_window(segments, start, end)
        duration = end - start
        if duration < min_duration:
            engagement.warnings.append("too_short")
        elif duration > max_duration:
            engagement.warnings.append("too_long")
        ranked.append(WindowCandidate(
            start=round(start, 2),
            end=round(end, 2),
            label=label,
            base_score=round(base_score, 4),
            engagement=engagement,
        ))

    def sort_key(item: WindowCandidate) -> Tuple[int, float, float]:
        out_of_range = 1 if any(
            code in item.engagement.warnings for code in ("too_short", "too_long")
        ) else 0
        return (out_of_range, -item.combined_score, -item.engagement.score)

    ranked.sort(key=sort_key)
    return ranked


def best_energy_peak(
    segments: Sequence[TranscriptSegment],
    start: float,
    end: float,
    min_duration: float = 30.0,
    max_duration: float = 60.0,
    bucket_seconds: float = 1.0,
) -> Optional[Tuple[float, float, float]]:
    """
    Slide a legal-duration window across [start, end) and return the busiest.

    Returns (window_start, window_end, score) or None when the range is too short
    to contain a legal window.
    """
    if end - start < min_duration:
        return None

    best: Optional[Tuple[float, float, float]] = None
    cursor = start
    while cursor + min_duration <= end + 1e-9:
        for length in (min_duration, max_duration, (min_duration + max_duration) / 2.0):
            window_end = min(end, cursor + length)
            if window_end - cursor < min_duration - 1e-9:
                continue
            sample = analyze_window(segments, cursor, window_end)
            if best is None or sample.score > best[2]:
                best = (round(cursor, 2), round(window_end, 2), sample.score)
        if cursor + min_duration >= end:
            break
        cursor = min(cursor + bucket_seconds, end - min_duration)
    return best


def summarise(samples: Sequence[EngagementSample]) -> Dict[str, float]:
    """Aggregate a curve for logging and artifact serialisation."""
    if not samples:
        return {}
    return {
        "buckets": float(len(samples)),
        "mean_score": round(sum(s.score for s in samples) / len(samples), 4),
        "max_score": round(max(s.score for s in samples), 4),
        "mean_words_per_second": round(
            sum(s.words_per_second for s in samples) / len(samples), 4),
        "mean_speech_ratio": round(
            sum(s.speech_ratio for s in samples) / len(samples), 4),
        "dead_air_windows": float(sum(1 for s in samples if "dead_air" in s.warnings)),
    }
