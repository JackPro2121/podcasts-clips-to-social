import json
import re
import time
import requests
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

from src.config import GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY
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

def query_gemini_models(prompt: str, key: str) -> Optional[str]:
    """Queries Google Gemini Flash free tier with exponential retry backoff."""
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash-8b",
        "gemini-2.5-flash"
    ]
    for model_name in models_to_try:
        print(f"[*] Trying Gemini Flash ({model_name})...")
        for attempt in range(3):
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
                    raw = response.text.strip()
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
                    raw = response.text.strip()
                else:
                    return None

                if raw and len(raw) > 20:
                    return raw
            except Exception as e:
                err_str = str(e)
                if ("503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str) and attempt < 2:
                    wait_sec = 2 ** (attempt + 1)
                    print(f"[-] Model {model_name} busy ({e}). Retrying in {wait_sec}s...")
                    time.sleep(wait_sec)
                else:
                    print(f"[-] Model {model_name} failed: {e}")
                    break
    return None

def query_groq_free_models(prompt: str, key: str) -> Optional[str]:
    """Queries Groq free tier models (ultra-fast inference, $0 budget)."""
    models = ["groq/compound-mini", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for m in models:
        print(f"[*] Trying Groq Free Model ({m})...")
        try:
            payload = {
                "model": m,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            r = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=25)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                if content and len(content) > 20:
                    print(f"[+] Groq free model ({m}) returned viral moments successfully!")
                    return content
            else:
                print(f"[-] Groq {m} returned status {r.status_code}")
        except Exception as e:
            print(f"[-] Groq {m} error: {e}")
    return None

def query_openrouter_free_models(prompt: str, key: str) -> Optional[str]:
    """Queries OpenRouter verified 100% free models (:free tier)."""
    models = ["minimax/minimax-m3:free", "minimax/minimax-m2.7:free", "liquid/lfm-2.5-2.6b:free"]
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/JackPro2121/podcasts-clips-to-social",
        "X-Title": "Podcast Clipper"
    }
    for m in models:
        print(f"[*] Trying OpenRouter Free Model ({m})...")
        try:
            payload = {
                "model": m,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "response_format": {"type": "json_object"}
            }
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=25)
            if r.status_code == 200:
                content = r.json()["choices"][0]["message"]["content"]
                if content and len(content) > 20:
                    print(f"[+] OpenRouter free model ({m}) returned viral moments successfully!")
                    return content
            else:
                print(f"[-] OpenRouter {m} returned status {r.status_code}")
        except Exception as e:
            print(f"[-] OpenRouter {m} error: {e}")
    return None

def parse_clips_json(raw_text: str, segments: List[TranscriptSegment], num_clips: int) -> List[ViralClipCandidate]:
    """Parses raw LLM JSON into validated ViralClipCandidate objects."""
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
        end = float(c.get("end_time", start + 40.0))
        if end <= start:
            end = start + 40.0
        
        dur = end - start
        if dur > 65.0:
            end = start + 55.0
            dur = 55.0
        elif dur < 25.0:
            end = min(start + 40.0, segments[-1].end if segments else start + 40.0)
            dur = end - start

        candidate_title = re.sub(r'[^\w\s\-\'\,\.\?]', '', str(c.get("title", "Viral Moment"))).strip().upper()
        candidate = ViralClipCandidate(
            title=candidate_title or "VIRAL MOMENT",
            start_time=start,
            end_time=end,
            duration=dur,
            viral_score=int(c.get("viral_score", 85)),
            hook_reason=c.get("hook_reason", "High engagement segment"),
            social_caption=c.get("social_caption", "Wait until the end... #shorts"),
            hashtags=c.get("hashtags", ["#podcast", "#viral", "#shorts"])
        )
        candidates.append(candidate)

    candidates.sort(key=lambda x: x.viral_score, reverse=True)
    return candidates[:num_clips]

def detect_viral_moments(
    segments: List[TranscriptSegment],
    num_clips: int = 3,
    api_key: Optional[str] = None
) -> List[ViralClipCandidate]:
    """
    Multi-Tier Zero-Cost Autonomous AI Viral Detection:
    1. Primary: Google Gemini Flash Free Tier
    2. Fallback 1: Groq Free Tier (groq/compound-mini, openai/gpt-oss-120b)
    3. Fallback 2: OpenRouter Free Tier (minimax/minimax-m3:free)
    4. Fallback 3: Semantic topic extraction from spoken dialogue
    """
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
6. **Punchy Viral Title**: Give each clip an engaging, click-worthy hook title in ALL CAPS (e.g., "THE SECRET TO BETTER SLEEP", "HOW CORTISOL PEAKS", "DO THIS EVERY MORNING"). Max 5-7 words. Strictly DO NOT include any emojis or special symbols.

### PODCAST TRANSCRIPT:
{transcript_text}

### OUTPUT FORMAT:
Output MUST be valid JSON only matching this schema:
{{
  "clips": [
    {{
      "title": "PUNCHY VIRAL TITLE",
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

    gemini_key = api_key or GEMINI_API_KEY
    raw_text = None

    # Tier 1: Gemini Flash Free Tier
    if gemini_key:
        print("[*] Tier 1: Sending transcript to Google Gemini Flash...")
        raw_text = query_gemini_models(prompt, gemini_key)

    # Tier 2: Groq Free Tier
    if not raw_text and GROQ_API_KEY:
        print("[*] Tier 2: Falling back to Groq free models...")
        raw_text = query_groq_free_models(prompt, GROQ_API_KEY)

    # Tier 3: OpenRouter Free Tier
    if not raw_text and OPENROUTER_API_KEY:
        print("[*] Tier 3: Falling back to OpenRouter free models...")
        raw_text = query_openrouter_free_models(prompt, OPENROUTER_API_KEY)

    # Parse JSON if any LLM responded
    if raw_text:
        try:
            candidates = parse_clips_json(raw_text, segments, num_clips)
            if candidates:
                print(f"[+] Successfully detected {len(candidates)} viral candidates via AI!")
                return candidates
        except Exception as e:
            print(f"[-] Parsing AI response failed ({e}). Falling back to semantic topic detector.")

    # Tier 4: Semantic dialogue topic extraction
    print("[*] Tier 4: Using intelligent semantic topic detector.")
    return fallback_rule_based_detector(segments, num_clips)

def fallback_rule_based_detector(segments: List[TranscriptSegment], num_clips: int = 3) -> List[ViralClipCandidate]:
    """Intelligent semantic fallback that extracts meaningful topic titles from transcript speech."""
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
        
        # Extract speech text within this window to form a relevant semantic title
        chunk_words = []
        for s in segments:
            if s.end >= start_t and s.start <= end_t:
                chunk_words.extend(s.text.split())
        
        # Derive a punchy 4-7 word title from the opening statement
        clean_words = [re.sub(r'[^\w\s]', '', w) for w in chunk_words[:12] if len(w) > 1]
        if clean_words:
            # Pick first 4-6 words as capitalized headline
            title_words = clean_words[:5]
            derived_title = " ".join(title_words).upper()
        else:
            derived_title = f"POWERFUL PODCAST INSIGHT #{i+1}"
        
        candidates.append(ViralClipCandidate(
            title=derived_title,
            start_time=round(start_t, 1),
            end_time=round(end_t, 1),
            duration=round(end_t - start_t, 1),
            viral_score=80 - (i * 5),
            hook_reason="Engaging dialogue section with high-retention speech",
            social_caption=f"{derived_title}\n\nWhat are your thoughts on this? Let us know below! 👇",
            hashtags=["#podcast", "#mindset", "#shorts", "#reels", "#viral"]
        ))
    return candidates
