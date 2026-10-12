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
import unicodedata
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


def _build_streamer_detection_prompt(
    transcript_text: str,
    creator_name: str,
    num_clips: int = 3
) -> str:
    return f"""You are the world's #1 viral short-form video editor for top US Twitch and YouTube streamers (Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc).
Your goal is to extract EXACTLY {num_clips} of the MOST EXPLOSIVE, FUNNY, OR UNHINGED VIRAL MOMENTS from this stream by {creator_name}.

### STREAMER TIKTOK VIRALITY FORMULA:
1. **Immediate Chaos / Hook (0-2s)**: The clip MUST start right before a massive spike in energy or right as the funny/shocking situation unfolds.
   - PURE RAGE: Screaming, table slams, rage quitting, unhinged disbelief.
   - COMEDY / FAILS: Getting roasted by chat/donations, failing a game in a ridiculous way, laughing uncontrollably.
   - DRAMA / BEEF: Confrontations, calling someone out, unfiltered hot takes.
   - HYPE / CLUTCH: Insane gaming clutch, massive celebration, unexpected twist.
2. **Optimal Pacing (20s - 45s Sweet Spot)**: Keep it tight and fast-paced! Never drag past 45 seconds.
3. **Standalone Cohesive Scene**: The moment must make complete sense on TikTok/Reels without context.
4. **Clean Complete Boundaries**: Do not cut off in the middle of a scream or punchline.
5. **TikTok Meme Hook Title**: Under 45 characters. Write like top TikTok clip channels and ALWAYS include 1 or 2 viral emojis at the end (e.g. "Nah Kai really started crashing out 💀", "Kai's cooked 😭", "No way Kai responded to this 💀😭", "Bro had Kevin Hart in tears 😂", "Kai Cenat NEVER beating these ALLEGATIONS 💀").
6. **Social Caption**: 1 short punchy sentence matching authentic TikTok viewer tone (e.g. "Bro was not ready for this at all.").
7. **HASHTAGS (STRICT REQUIREMENT)**: EXACTLY 2 to 4 hashtags ONLY! Never more than 4, never less than 2.
   Examples: ["#kaicenat", "#streamer", "#funny", "#fyp"]

### OUTPUT JSON FORMAT:
{{
  "clips": [
    {{
      "title": "Nah Kai really started crashing out 💀",
      "start_time": 120.5,
      "end_time": 155.0,
      "duration": 34.5,
      "viral_score": 95,
      "hook_reason": "High emotional spike and loud streamer reaction",
      "social_caption": "Wait until you see how this ended. Bro was not ready.",
      "hashtags": ["#kaicenat", "#streamer", "#funny", "#fyp"],
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
        # HARD CAP: Streamer clips for TikTok/Shorts must be strictly under 45s
        if end - start > 45.0:
            end = start + 45.0
        normalized = _normalize_clip_window(start, end, segments)
        if normalized is not None:
            start, end = normalized
            if end - start > 45.0:
                end = start + 45.0
        else:
            # Resilient fallback for streamer clips (bounds 20s to 45s)
            transcript_end = max((s.end for s in segments), default=9999.0)
            start = max(0.0, min(start, max(0.0, transcript_end - 25.0)))
            end = min(transcript_end, max(start + 20.0, min(end, start + 45.0)))
        dur = end - start

        # Title formatting: Keep natural TikTok casing or uppercase with emojis
        raw_title = str(c.get("title", "STREAMER VIRAL MOMENT 💀")).strip()
        clean_title = "".join(
            ch for ch in unicodedata.normalize("NFC", raw_title)
            if not unicodedata.category(ch).startswith("C") and ch not in "\r\n\t"
        ).strip()
        if not clean_title:
            clean_title = "STREAMER VIRAL MOMENT 💀"

        # If title doesn't contain an emoji, append an authentic viral emoji based on energy
        energy_emojis = {
            "rage": "💀",
            "funny": "😭",
            "clutch": "🔥",
            "hype": "🔥",
            "drama": "💀😭",
        }
        has_emoji = any(
            (0x1F000 <= ord(ch) <= 0x1FAFF) or (0x2600 <= ord(ch) <= 0x27BF)
            for ch in clean_title
        )
        if not has_emoji:
            energy = str(c.get("streamer_energy", "funny")).lower().strip()
            clean_title = f"{clean_title} {energy_emojis.get(energy, '💀')}"

        caption = strip_emojis(str(c.get("social_caption", "Bro was not ready for this.")))

        # Enforce strictly 2 to 4 hashtags (user requirement)
        raw_hashtags = c.get("hashtags", [])
        if not isinstance(raw_hashtags, list):
            raw_hashtags = [str(raw_hashtags)]
        formatted_hashtags: List[str] = []
        for h in raw_hashtags:
            clean_h = strip_emojis(str(h)).strip()
            if not clean_h:
                continue
            tag = clean_h if clean_h.startswith("#") else f"#{clean_h}"
            tag = re.sub(r"[^\w#]", "", tag)
            if tag and tag not in formatted_hashtags:
                formatted_hashtags.append(tag)

        # Strictly 2 to 4 hashtags
        if len(formatted_hashtags) > 4:
            formatted_hashtags = formatted_hashtags[:4]
        elif len(formatted_hashtags) < 2:
            default_tags = ["#streamer", "#funny", "#fyp", "#viral"]
            for dt in default_tags:
                if dt not in formatted_hashtags and len(formatted_hashtags) < 4:
                    formatted_hashtags.append(dt)
        clean_hashtags = formatted_hashtags[:4]

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
            title=clean_title,
            start_time=start,
            end_time=end,
            duration=dur,
            viral_score=_safe_int(c.get("viral_score", 90), default=90),
            hook_reason=strip_emojis(str(c.get("hook_reason", "High energy stream reaction"))),
            social_caption=caption,
            hashtags=clean_hashtags,
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

    prompt = _build_streamer_detection_prompt(transcript_text, creator_name, num_clips)

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
    return _heuristic_streamer_detector(segments, creator_name=creator_name, num_clips=num_clips)


def _generate_dynamic_streamer_title(creator_name: str, sample_text: str, energy: str) -> str:
    """Generates an authentic high-CTR TikTok hook title with viral emojis based on text context."""
    clean_creator = creator_name.title() if creator_name and creator_name.lower() not in ("streamer", "streamers") else "Kai"
    lower = sample_text.lower()

    if any(k in lower for k in ["beast", "mrbeast", "jimmy"]):
        options = [
            f"Nah {clean_creator} really thought MrBeast was normal 💀",
            f"MrBeast had {clean_creator} completely terrified 😭💀",
            f"{clean_creator} and MrBeast had the wildest stream 💀🔥",
        ]
    elif any(k in lower for k in ["firework", "fireworks", "boom", "explosion", "bomb"]):
        options = [
            f"Nah {clean_creator} really lit fireworks in his room 💀😭",
            f"{clean_creator} almost burned his house down 💀🔥",
            f"Bro had the whole room shaking 😭💀",
        ]
    elif any(k in lower for k in ["speed", "ishowspeed"]):
        options = [
            f"Speed and {clean_creator} are completely unhinged 💀😭",
            f"No way Speed did this on {clean_creator}'s stream 💀",
        ]
    elif energy == "rage" or any(k in lower for k in ["shut up", "hell no", "swear", "stop", "deadass"]):
        options = [
            f"Nah {clean_creator} really started crashing out 💀",
            f"{clean_creator} was completely fed up with this 💀😭",
            f"Bro {clean_creator} is actually so mad right now 😭",
            f"{clean_creator} caught screaming at chat 💀",
        ]
    elif energy == "funny" or any(k in lower for k in ["laugh", "bruh", "nah", "cooked", "chat"]):
        options = [
            f"Bro {clean_creator} is actually cooked 😭💀",
            f"Chat had {clean_creator} in tears laughing 😭",
            f"No way {clean_creator} actually did this 💀😭",
            f"{clean_creator} could NOT hold his laugh 😭💀",
        ]
    else:
        options = [
            f"Nah {clean_creator} is NOT real for this 💀😭",
            f"{clean_creator} caught completely off guard 💀",
            f"Wait till you see {clean_creator}'s reaction 😭🔥",
            f"{clean_creator} NEVER beating the allegations 💀",
            f"Bro was NOT ready for what happened next 💀",
        ]
    import random
    return random.choice(options)


def _heuristic_streamer_detector(
    segments: List[TranscriptSegment],
    creator_name: str = "Streamer",
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
        energy = "hype"
        for e_key, kw_list in STREAMER_KEYWORDS.items():
            for kw in kw_list:
                if kw in lower:
                    score += 5
                    energy = e_key

        scored_windows.append((score, start, end, text[:60], energy))

    scored_windows.sort(key=lambda x: x[0], reverse=True)
    results = []
    used_times: List[Tuple[float, float]] = []

    clean_creator_tag = f"#{creator_name.lower().replace(' ', '')}" if creator_name and creator_name.lower() != "streamer" else "#kaicenat"

    for score, start, end, sample, energy in scored_windows:
        overlap = any(not (end <= u_s or start >= u_e) for u_s, u_e in used_times)
        if overlap:
            continue
        used_times.append((start, end))

        title = _generate_dynamic_streamer_title(creator_name, sample, energy)

        results.append(StreamerClipCandidate(
            title=title,
            start_time=start,
            end_time=end,
            duration=end - start,
            viral_score=min(95, score),
            hook_reason="High density of vocal excitement and reactions",
            social_caption="Bro was not ready for this. Wait until you see how it ended.",
            hashtags=[clean_creator_tag, "#streamer", "#funny", "#fyp"],
            streamer_energy=energy,
            sfx_cues=[(0.1, "whoosh"), (end - start - 1.5, "ding")]
        ))
        if len(results) >= num_clips:
            break

    return results
