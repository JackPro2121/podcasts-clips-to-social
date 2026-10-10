"""High-Energy Streamer Viral Detector (V2).

Specialized AI model prompting and heuristic fallback for high-energy streamer content:
- Screaming, rage, desk slams, chair falls, jumping
- Hysterical laughter, comedy fails, unhinged reactions
- Chat donation roasts, troll moments, beef, drama
- High-stakes clutches, jumpscares, chaotic energy

Runs Gemini 3.8 Flash -> Groq -> Ollama -> Streamer heuristic detector.
"""

import json
import re
from typing import List, Tuple
from pydantic import BaseModel, Field

from src.transcriber import TranscriptSegment
from src.viral_detector import (
    query_gemini_models, query_groq_free_models, query_ollama_cloud_models,
    strip_emojis, _normalize_clip_window, _safe_int
)
from src.config import GEMINI_API_KEY, GROQ_API_KEY, OLLAMA_API_KEY


class StreamerClipCandidate(BaseModel):
    title: str = Field(description="Punchy all-caps viral hook title (under 45 chars)")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds")
    duration: float = Field(description="Duration in seconds (15 to 45s)")
    viral_score: int = Field(description="Virality score 1 to 100")
    hook_reason: str = Field(description="Why this streamer moment goes viral")
    social_caption: str = Field(description="TikTok/Shorts engaging caption")
    hashtags: List[str] = Field(description="Relevant hashtags like #streamer #funny #twitch #shorts")
    streamer_energy: str = Field(default="hype", description="rage | funny | clutch | drama | hype")
    sfx_cues: List[Tuple[float, str]] = Field(
        default_factory=list,
        description="Sound effect triggers relative to clip start: [[timestamp_sec, 'whoosh'|'pop'|'ding'], ...]"
    )


STREAMER_KEYWORDS = {
    "rage": ["bro", "what", "no way", "stop", "swear", "deadass", "shut up", "hell no", "cap"],
    "funny": ["laugh", "bruh", "nah", "cooked", "chat", "gg", "lmaooo", "wild", "tripping"],
    "hype": ["let's go", "clutch", "insane", "unbelievable", "w", "huge", "holy", "god"],
}


def build_streamer_detection_prompt(
    transcript_text: str,
    creator_name: str,
    num_clips: int = 3
) -> str:
    return f"""You are the world's #1 viral short-form video editor for top US Twitch and YouTube streamers (Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc).
Your goal is to extract EXACTLY {num_clips} of the MOST EXPLOSIVE, FUNNY, OR UNHINGED VIRAL MOMENTS from this stream by {creator_name}.

### STREAMER VIRALITY FORMULA:
1. **Immediate Chaos / Hook (0-2s)**: The clip MUST start right before a massive spike in energy or right as the funny/shocking situation unfolds.
   - PURE RAGE: Screaming, table slams, rage quitting, unhinged disbelief.
   - COMEDY / FAILS: Getting roasted by chat/donations, failing a game in a ridiculous way, laughing uncontrollably.
   - DRAMA / BEEF: Confrontations, calling someone out, unfiltered hot takes.
   - HYPE / CLUTCH: Insane gaming clutch, massive celebration, unexpected twist.
2. **Optimal Pacing (20s - 45s Sweet Spot)**: Keep it tight and fast-paced! Never drag past 45 seconds.
3. **Standalone Cohesive Scene**: The moment must make complete sense on TikTok/Reels without context.
4. **Clean Complete Boundaries**: Do not cut off in the middle of a scream or punchline.
5. **Hook Title in ALL CAPS**: Under 45 characters, curiosity-driven (e.g. "KAI CANNOT BELIEVE THIS HAPPENED", "SPEED RAGES OVER NEW UPDATE", "CASEOH GETS ROASTED BY DONATION").
6. **NO EMOJIS**: Strictly NO emojis in titles, captions, or hashtags.

### OUTPUT JSON FORMAT:
{{
  "clips": [
    {{
      "title": "CATCHY VIRAL HOOK TITLE",
      "start_time": 120.5,
      "end_time": 155.0,
      "duration": 34.5,
      "viral_score": 95,
      "hook_reason": "High emotional spike and loud streamer reaction",
      "social_caption": "Wait until you see how this ended. Bro was not ready.",
      "hashtags": ["#streamer", "#twitch", "#funny", "#shorts", "#viral"],
      "streamer_energy": "rage",
      "sfx_cues": [[0.1, "whoosh"], [12.0, "pop"], [32.0, "ding"]]
    }}
  ]
}}

STREAM TRANSCRIPT:
{transcript_text}
"""


def parse_streamer_clips_json(
    raw_text: str,
    segments: List[TranscriptSegment],
    num_clips: int
) -> List[StreamerClipCandidate]:
    clean_text = raw_text.strip()
    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]

    parsed = json.loads(clean_text.strip())
    clips_data = parsed.get("clips", [])
    candidates = []

    for c in clips_data:
        start = float(c.get("start_time", 0.0))
        end = float(c.get("end_time", start + 30.0))
        normalized = _normalize_clip_window(start, end, segments)
        if normalized is None:
            continue
        start, end = normalized
        dur = end - start

        title = strip_emojis(re.sub(r'[^\w\s\-\'\,\.\?]', '', str(c.get("title", "STREAMER VIRAL MOMENT"))).strip().upper())
        caption = strip_emojis(str(c.get("social_caption", "Bro was not ready for this. #streamer #shorts")))
        raw_hashtags = c.get("hashtags", ["#streamer", "#twitch", "#funny", "#shorts"])
        clean_hashtags = [strip_emojis(str(h)).strip() for h in raw_hashtags if strip_emojis(str(h)).strip()]

        raw_sfx = c.get("sfx_cues", [])
        clean_sfx = []
        for s in raw_sfx:
            if isinstance(s, (list, tuple)) and len(s) == 2:
                try:
                    s_t = float(s[0])
                    s_type = str(s[1]).lower().strip()
                    if s_type in ("whoosh", "pop", "ding"):
                        clean_sfx.append((s_t, s_type))
                except Exception:
                    pass

        if not clean_sfx:
            clean_sfx = [(0.1, "whoosh"), (max(1.0, dur - 1.5), "ding")]

        candidates.append(StreamerClipCandidate(
            title=title or "UNBELIEVABLE STREAM MOMENT",
            start_time=start,
            end_time=end,
            duration=dur,
            viral_score=_safe_int(c.get("viral_score", 90), default=90),
            hook_reason=strip_emojis(str(c.get("hook_reason", "High energy stream reaction"))),
            social_caption=caption,
            hashtags=clean_hashtags or ["#streamer", "#funny", "#twitch", "#shorts"],
            streamer_energy=str(c.get("streamer_energy", "hype")),
            sfx_cues=clean_sfx
        ))

    candidates.sort(key=lambda x: x.viral_score, reverse=True)
    return candidates[:num_clips]


def detect_streamer_viral_moments(
    segments: List[TranscriptSegment],
    creator_name: str = "Streamer",
    num_clips: int = 3
) -> List[StreamerClipCandidate]:
    """Detects viral moments using multi-tier fallback tuned for streamer excitement."""
    lines = []
    total_len = 0
    for seg in segments:
        line = f"[{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text}"
        total_len += len(line)
        if total_len > 40000:
            lines.append("... [transcript truncated] ...")
            break
        lines.append(line)
    transcript_text = "\n".join(lines)

    prompt = build_streamer_detection_prompt(transcript_text, creator_name, num_clips)

    # 1. Primary: Gemini 3.8 Flash
    if GEMINI_API_KEY:
        try:
            print("[*] V2 Streamer Detector: Querying Google Gemini Flash...")
            res = query_gemini_models(prompt, GEMINI_API_KEY)
            if res:
                parsed = parse_streamer_clips_json(res, segments, num_clips)
                if parsed:
                    print(f"[+] V2 Streamer Detector: Found {len(parsed)} viral moments via Gemini!")
                    return parsed
        except Exception as e:
            print(f"[-] Gemini streamer detection failed: {e}")

    # 2. Fallback: Groq
    if GROQ_API_KEY:
        try:
            print("[*] V2 Streamer Detector: Falling back to Groq...")
            res = query_groq_free_models(prompt, GROQ_API_KEY)
            if res:
                parsed = parse_streamer_clips_json(res, segments, num_clips)
                if parsed:
                    return parsed
        except Exception as e:
            print(f"[-] Groq streamer detection failed: {e}")

    # 3. Fallback: Ollama
    if OLLAMA_API_KEY:
        try:
            print("[*] V2 Streamer Detector: Falling back to Ollama Cloud...")
            res = query_ollama_cloud_models(prompt, OLLAMA_API_KEY)
            if res:
                parsed = parse_streamer_clips_json(res, segments, num_clips)
                if parsed:
                    return parsed
        except Exception:
            pass

    # 4. Fallback: Heuristic streamer density
    print("[*] V2 Streamer Detector: Running streamer excitement heuristic fallback...")
    return _heuristic_streamer_detector(segments, num_clips)


def _heuristic_streamer_detector(
    segments: List[TranscriptSegment],
    num_clips: int = 3
) -> List[StreamerClipCandidate]:
    """Fast rule-based detection for exclamation marks, caps, and streamer slang."""
    scored_windows = []
    window_s = 35.0

    for i, seg in enumerate(segments):
        start = seg.start
        end = start + window_s
        window_segs = [s for s in segments if s.start >= start and s.end <= end]
        if not window_segs:
            continue
        text = " ".join(s.text for s in window_segs)
        score = 60
        # Exclamation marks & caps indicate shouting/rage
        score += text.count("!") * 4
        score += sum(1 for w in text.split() if w.isupper() and len(w) > 2) * 3
        # Keywords
        lower = text.lower()
        for kw_list in STREAMER_KEYWORDS.values():
            for kw in kw_list:
                if kw in lower:
                    score += 5

        scored_windows.append((score, start, end, text[:60]))

    scored_windows.sort(key=lambda x: x[0], reverse=True)
    results = []
    used_times: List[Tuple[float, float]] = []

    for score, start, end, sample in scored_windows:
        overlap = any(not (end <= u_s or start >= u_e) for u_s, u_e in used_times)
        if overlap:
            continue
        used_times.append((start, end))
        results.append(StreamerClipCandidate(
            title="INSANE STREAM REACTION",
            start_time=start,
            end_time=end,
            duration=end - start,
            viral_score=min(95, score),
            hook_reason="High density of vocal excitement and reactions",
            social_caption="Bro really thought he could get away with this. #streamer #funny #shorts",
            hashtags=["#streamer", "#twitch", "#funny", "#shorts"],
            streamer_energy="hype",
            sfx_cues=[(0.1, "whoosh"), (end - start - 1.5, "ding")]
        ))
        if len(results) >= num_clips:
            break

    return results
