import json
import warnings
from typing import List, Optional
from pydantic import BaseModel, Field

# Support both google.genai (new official SDK) and google.generativeai
try:
    from google import genai
    from google.genai import types as genai_types
    HAS_NEW_GENAI = True
except ImportError:
    HAS_NEW_GENAI = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        try:
            import google.generativeai as legacy_genai
        except ImportError:
            legacy_genai = None

from src.config import GEMINI_API_KEY
from src.transcriber import TranscriptSegment

class ViralClipCandidate(BaseModel):
    title: str = Field(description="Catchy viral hook title (under 50 chars)")
    start_time: float = Field(description="Start time in seconds")
    end_time: float = Field(description="End time in seconds (must be 30-60s after start_time)")
    duration: float = Field(description="Duration in seconds (30.0 to 60.0)")
    viral_score: int = Field(description="Predicted virality score from 1 to 100")
    hook_reason: str = Field(description="Why this moment grabs immediate viewer attention")
    social_caption: str = Field(description="Ready-to-post engaging caption for TikTok/Reels/Shorts")
    hashtags: List[str] = Field(description="High-traffic relevant hashtags (e.g. ['#podcast', '#viral', '#mindset'])")

class ViralDetectionResponse(BaseModel):
    clips: List[ViralClipCandidate]

def format_transcript_with_timestamps(segments: List[TranscriptSegment], max_chars: int = 80000) -> str:
    """Formats transcript segments with start and end timestamps for LLM analysis."""
    lines = []
    total_len = 0
    for seg in segments:
        line = f"[{seg.start:.1f}s - {seg.end:.1f}s]: {seg.text}"
        total_len += len(line)
        if total_len > max_chars:
            lines.append("... [transcript truncated for token budget] ...")
            break
        lines.append(line)
    return "\n".join(lines)

def detect_viral_moments(
    segments: List[TranscriptSegment],
    num_clips: int = 3,
    api_key: Optional[str] = None
) -> List[ViralClipCandidate]:
    """
    Uses Google Gemini Flash (Free Tier) to identify high-retention viral moments.
    Returns sorted list of top viral clip candidates.
    """
    key = api_key or GEMINI_API_KEY
    if not key:
        print("[!] GEMINI_API_KEY is not set. Falling back to heuristic/rule-based moment detector.")
        return fallback_rule_based_detector(segments, num_clips)

    models_to_try = ["gemini-2.5-flash", "gemini-flash-latest", "gemini-1.5-flash"]
    transcript_text = format_transcript_with_timestamps(segments)

    prompt = f"""
You are the world's top viral short-form video editor and content strategist (specializing in TikTok, Instagram Reels, and YouTube Shorts).
Your goal is to analyze the following podcast transcript and extract the top {num_clips} most VIRAL standalone moments.

### VIRALITY CRITERIA:
1. **Immediate Hook (0-5s)**: Must start with a bold statement, intriguing question, shock value, or strong emotion that prevents scrolling.
2. **High Emotional Intensity or Insight**: Debates, counter-intuitive advice, mind-blowing facts, deep vulnerability, or high humor.
3. **Standalone Cohesion**: The clip must make complete sense on its own without needing the rest of the 2-hour podcast.
4. **Optimal Duration**: Each clip MUST be strictly between 30 and 60 seconds (target: 35-50s).
5. **Exact Timestamps**: Use the provided transcript timestamps to specify precise start_time and end_time.

### PODCAST TRANSCRIPT:
{transcript_text}

### OUTPUT FORMAT:
Output MUST be valid JSON only matching this schema:
{{
  "clips": [
    {{
      "title": "Short punchy title",
      "start_time": 124.5,
      "end_time": 172.0,
      "duration": 47.5,
      "viral_score": 95,
      "hook_reason": "Opens with a shocking contrarian statement about wealth.",
      "social_caption": "This perspective changes everything. Drop your thoughts below 👇",
      "hashtags": ["#mindset", "#podcast", "#success", "#reels"]
    }}
  ]
}}
Do not include markdown backticks or commentary outside the JSON.
"""

    raw_text = None
    last_err = None
    for model_name in models_to_try:
        print(f"[*] Sending transcript to Gemini Flash ({model_name}) for viral moment hunting...")
        try:
            if HAS_NEW_GENAI:
                client = genai.Client(api_key=key)
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.4,
                        response_mime_type="application/json"
                    )
                )
                raw_text = response.text.strip()
            elif legacy_genai is not None:
                legacy_genai.configure(api_key=key)
                model = legacy_genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=legacy_genai.GenerationConfig(
                        temperature=0.4,
                        response_mime_type="application/json"
                    )
                )
                raw_text = response.text.strip()
            else:
                raise RuntimeError("No Google GenAI library installed.")
            
            if raw_text:
                break
        except Exception as e:
            last_err = e
            print(f"[-] Model {model_name} failed: {e}. Trying fallback model...")

    if not raw_text:
        print(f"[-] All Gemini models failed ({last_err}). Falling back to heuristic detector.")
        return fallback_rule_based_detector(segments, num_clips)

    try:
        # Clean any accidental wrapping
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        parsed = json.loads(raw_text.strip())
        clips_data = parsed.get("clips", [])
        
        candidates = []
        for c in clips_data:
            start = float(c.get("start_time", 0.0))
            end = float(c.get("end_time", start + 40.0))
            if end <= start:
                end = start + 40.0
            
            # Enforce 30-60s constraints
            dur = end - start
            if dur > 65.0:
                end = start + 55.0
                dur = 55.0
            elif dur < 25.0:
                end = min(start + 40.0, segments[-1].end if segments else start + 40.0)
                dur = end - start

            candidate = ViralClipCandidate(
                title=c.get("title", "Viral Moment"),
                start_time=start,
                end_time=end,
                duration=dur,
                viral_score=int(c.get("viral_score", 85)),
                hook_reason=c.get("hook_reason", "High engagement segment"),
                social_caption=c.get("social_caption", "Wait until the end... #shorts"),
                hashtags=c.get("hashtags", ["#podcast", "#viral", "#shorts"])
            )
            candidates.append(candidate)

        # Sort by viral_score descending
        candidates.sort(key=lambda x: x.viral_score, reverse=True)
        print(f"[+] Successfully detected {len(candidates)} viral candidates via Gemini!")
        return candidates[:num_clips]

    except Exception as e:
        print(f"[-] Parsing Gemini response failed ({e}). Falling back to heuristic detector.")
        return fallback_rule_based_detector(segments, num_clips)

def fallback_rule_based_detector(segments: List[TranscriptSegment], num_clips: int = 3) -> List[ViralClipCandidate]:
    """Heuristic fallback if AI API is unavailable or offline."""
    if not segments:
        return []

    total_duration = segments[-1].end - segments[0].start
    step = total_duration / (num_clips + 1)
    
    candidates = []
    for i in range(num_clips):
        target_start = segments[0].start + step * (i + 0.5)
        # Find closest segment
        closest_seg = min(segments, key=lambda s: abs(s.start - target_start))
        start_t = closest_seg.start
        end_t = min(start_t + 45.0, segments[-1].end)
        
        candidates.append(ViralClipCandidate(
            title=f"Viral Highlight #{i+1}",
            start_time=round(start_t, 1),
            end_time=round(end_t, 1),
            duration=round(end_t - start_t, 1),
            viral_score=80 - (i * 5),
            hook_reason="Engaging dialogue section with continuous speech",
            social_caption=f"Check out this moment from the podcast! Let us know what you think below. 👇",
            hashtags=["#podcast", "#clips", "#reels", "#tiktok", "#shorts"]
        ))
    return candidates
