import re
from pathlib import Path
from typing import List, Optional, Any, Dict
from src.config import (
    SUBTITLE_THEMES, OUTPUT_WIDTH, OUTPUT_HEIGHT, CHANNEL_WATERMARK,
    ENABLE_TOP_HOOK_BADGE, HOOK_BADGE_DURATION, HOOK_BADGE_MARGIN_V
)
from src.transcriber import TranscriptSegment, WordTimestamp

# Default lower-third safe-zone margins per layout (must match generate_ass_header).
_SAFE_MARGIN_V = {
    "split_screen": 880,  # \an2 (bottom-center): MarginV=880 → text baseline at Y≈1000 (center divider)
    "blur_stack": 400,    # Just below the centered 16:9 panel
    "single_smooth": 460, # Golden sweet spot: Y≈1460 (safely above TikTok drawer, below speaker)
}
_DEFAULT_MARGIN_V = 460

MONEY_KEYWORDS = {
    "money", "cash", "dollar", "dollars", "wealth", "invest", "investing", "saving", "savings",
    "profit", "net", "worth", "assets", "million", "millions", "billion", "billions", "income",
    "tax", "taxes", "crypto", "salary", "rich", "fund", "budget", "bank", "credit", "paycheck",
    "paid", "earn", "earning"
}
DANGER_KEYWORDS = {
    "broke", "debt", "lie", "lied", "hate", "hater", "scam", "lost", "lose", "losing",
    "risk", "danger", "stupid", "mistake", "zero", "fail", "failed", "crash", "boredom",
    "worst", "unemployment", "seduction", "fool", "stop", "bad", "cut", "terrible"
}
POWER_KEYWORDS = {
    "rules", "rule", "game", "secret", "never", "always", "truth", "master", "power",
    "double", "boss", "timing", "how", "much", "left", "win", "winning"
}


def default_margin_v(layout_mode: str) -> int:
    """Safe-zone MarginV for a layout when no per-shot override applies."""
    return _SAFE_MARGIN_V.get(layout_mode, _DEFAULT_MARGIN_V)


def format_ass_timestamp(seconds: float) -> str:
    """Converts seconds into ASS timestamp format: H:MM:SS.cs (centiseconds).

    Uses a single centisecond accumulator so a rounding carry propagates
    correctly through seconds -> minutes -> hours (the old code could emit an
    invalid ``SS=60`` on values like 59.999)."""
    if seconds < 0:
        seconds = 0
    total_cs = int(round(seconds * 100))
    hours = total_cs // 360000
    minutes = (total_cs % 360000) // 6000
    secs = (total_cs % 6000) // 100
    centis = total_cs % 100
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"

def generate_ass_header(
    theme_key: str = "hormozi",
    layout_mode: str = "single_smooth",
    watermark_margin_v: Optional[int] = None,
    font_size_override: Optional[int] = None
) -> str:
    """Generates ASS header with custom high-end styling and safe-zone margin."""
    theme = SUBTITLE_THEMES.get(theme_key, SUBTITLE_THEMES["hormozi"])

    font_name = theme["font_name"]
    font_size = font_size_override or theme["font_size"]
    primary_col = theme["primary_color"]
    outline_col = theme["outline_color"]
    outline_w = theme["outline_width"]
    shadow_col = theme["shadow_color"]
    shadow_d = theme["shadow_depth"]

    # Safe Zone Placement:
    # In split_screen: captions placed at the center divider — \an2 + MarginV=880 puts the
    #   text baseline at Y≈1000px (just below the 960px midpoint), clearly over both panes
    #   and well above TikTok's bottom 20% UI zone.
    # In single_smooth, dynamic_cut, or blur_stack: placed strictly in the lower-third safe zone
    if layout_mode == "split_screen":
        alignment = 2  # Bottom Center
        margin_v = 880  # baseline at Y≈1000, straddles the center divider
    elif layout_mode == "blur_stack":
        alignment = 2  # Bottom Center
        margin_v = 400  # Perfectly below the 16:9 centered diagram (which ends at Y=1264)
    else:
        alignment = 2  # Bottom Center
        margin_v = 460  # Y ≈ 1460px (sweet spot: clear of bottom drawer, below mouth)

    # Watermark MarginV: dynamically position @allinonepodcastsss right below the purple capsule
    header = f"""[Script Info]
Title: Viral Social Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {OUTPUT_WIDTH}
PlayResY: {OUTPUT_HEIGHT}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_col},&H000000FF,{outline_col},{shadow_col},-1,0,0,0,100,100,1.5,0,1,{outline_w},{shadow_d},{alignment},100,100,{margin_v},1
Style: TopHeader,Montserrat Black,42,&H00FFFFFF,&H000000FF,&H00B86B62,&H00000000,-1,0,0,0,100,100,1.2,0,3,11,0,8,100,100,{HOOK_BADGE_MARGIN_V},1
Style: Watermark,Montserrat Black,24,&H90FFFFFF,&H000000FF,&H90000000,&H00000000,-1,0,0,0,100,100,1.2,0,1,1.5,0.0,7,60,60,90,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    return header

def create_styled_ass_subtitles(
    segments: List[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    output_ass_path: Path,
    theme_key: str = "hormozi",
    layout_mode: str = "single_smooth",
    header_title: Optional[str] = None,
    watermark: Optional[str] = None,
    shots: Optional[List[Any]] = None,
    keyword_emojis: Optional[Dict[str, str]] = None
) -> Path:
    """
    Generates word-level animated karaoke-style ASS subtitles for a specific clip window.
    Dynamically positions subtitles based on active shot layout (e.g. margin_v=420 during slides),
    with kinetic bounce pops and contextual emoji injection.
    """
    theme = SUBTITLE_THEMES.get(theme_key, SUBTITLE_THEMES["hormozi"])
    max_words = theme.get("max_words_per_line", 3)
    uppercase = theme.get("uppercase", True)
    highlight_color = theme.get("highlight_color", "&H0000E6FF")
    primary_color = theme.get("primary_color", "&H00FFFFFF")

    # Filter words strictly within the clip duration and offset timestamps to 0.0s
    clip_words: List[WordTimestamp] = []
    for seg in segments:
        for w in seg.words:
            if w.end >= clip_start and w.start <= clip_end:
                rel_start = max(0.0, w.start - clip_start)
                rel_end = max(rel_start + 0.1, min(clip_end - clip_start, w.end - clip_start))
                clean_word = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', w.word.strip())
                # Strictly filter out empty or non-alphanumeric ghost artifacts (e.g. '__', '--', '—')
                if not clean_word or not any(c.isalnum() for c in clean_word):
                    continue
                clean_text = clean_word.upper() if uppercase else clean_word
                clip_words.append(WordTimestamp(
                    word=clean_text,
                    start=rel_start,
                    end=rel_end
                ))

    # Sort words by start time and enforce strictly monotonic, non-overlapping timestamps
    clip_words.sort(key=lambda x: x.start)
    monotonic_words: List[WordTimestamp] = []
    last_end = 0.0
    for w in clip_words:
        w_start = max(last_end, w.start)
        w_end = max(w_start + 0.12, w.end)
        monotonic_words.append(WordTimestamp(word=w.word, start=w_start, end=w_end))
        last_end = w_end
    clip_words = monotonic_words

    # Adaptive Cadence: If speaker is talking at ultra-fast pace (> 190 WPM),
    # switch to punchy 1-word flash chunks to match speed. Otherwise use theme default.
    clip_dur = max(1.0, clip_end - clip_start)
    wpm = (len(clip_words) / clip_dur) * 60.0
    effective_max_words = 1 if (wpm > 190 and len(clip_words) > 10) else max_words
    single_word_font_size = 78 if effective_max_words == 1 else None

    base_margin_v = default_margin_v(layout_mode)

    def get_shot_margin_v(t: float) -> int:
        # Per-shot override wins; otherwise fall back to the layout's safe-zone
        # default (NOT 0 -- a 0 here overrode the style and pinned captions to the
        # very bottom of the frame, under the TikTok/Reels UI).
        if shots:
            for s in shots:
                s_start = getattr(s, "start", 0.0)
                s_end = getattr(s, "end", 9999.0)
                if s_start <= t <= s_end:
                    return getattr(s, "margin_v", None) or base_margin_v
        return base_margin_v

    # Group words into short punchy batches of 1-3 words
    lines: List[str] = []
    i = 0

    while i < len(clip_words):
        chunk = clip_words[i:i + effective_max_words]
        i += effective_max_words
        if not chunk:
            continue

        for active_idx, target_word in enumerate(chunk):
            w_start = target_word.start
            if active_idx + 1 < len(chunk):
                w_end = chunk[active_idx + 1].start
            else:
                w_end = target_word.end

            word_elements = []
            for idx, w in enumerate(chunk):
                if idx == active_idx:
                    # Semantic word coloring: Money/Numbers = Lime Green, Danger = Red, Power = Gold
                    clean_lower = re.sub(r'[^a-z0-9]', '', w.word.lower())
                    if clean_lower in MONEY_KEYWORDS or any(c.isdigit() for c in w.word) or '$' in w.word:
                        active_color = "&H0033FF22"  # Neon Lime Green
                    elif clean_lower in DANGER_KEYWORDS:
                        active_color = "&H003333FF"  # Fire Red
                    elif clean_lower in POWER_KEYWORDS:
                        active_color = "&H0000D7FF"  # Warm Gold
                    else:
                        active_color = highlight_color

                    # High-energy kinetic bounce pop: 118% scale punch settling to 100%
                    # Clean bold typography without unrenderable emoji tofu boxes
                    word_elements.append(
                        f"{{\\c{active_color}\\t(0,70,\\fscx118\\fscy118)\\t(70,140,\\fscx100\\fscy100)}}{w.word}{{\\c{primary_color}\\fscx100\\fscy100}}"
                    )
                else:
                    word_elements.append(w.word)

            dialogue_text = " ".join(word_elements)
            w_mid = (w_start + w_end) / 2
            active_margin_v = get_shot_margin_v(w_mid)
            ass_line = f"Dialogue: 0,{format_ass_timestamp(w_start)},{format_ass_timestamp(w_end)},Default,,0,0,{active_margin_v},,{dialogue_text}"
            lines.append(ass_line)

    has_badge = bool(header_title and ENABLE_TOP_HOOK_BADGE)
    if has_badge:
        clean_words = re.sub(r'[^\w\s\-\'\,\.\?]', '', (header_title or "").strip().upper()).split()
        num_lines = 2 if len(clean_words) >= 2 else 1
        w_margin_v = HOOK_BADGE_MARGIN_V + (95 if num_lines == 2 else 55)
    else:
        w_margin_v = HOOK_BADGE_MARGIN_V

    header = generate_ass_header(
        theme_key=theme_key,
        layout_mode=layout_mode,
        watermark_margin_v=w_margin_v,
        font_size_override=single_word_font_size
    )
    dur_str = format_ass_timestamp(clip_end - clip_start)

    # Watermark (if configured)
    active_watermark = watermark if watermark is not None else CHANNEL_WATERMARK
    if active_watermark:
        lines.insert(0, f"Dialogue: 2,0:00:00.00,{dur_str},Watermark,,0,0,0,,{active_watermark.strip()}")

    # Hook title capsule badge: initial hook retention, smooth fade out, clean typography
    if header_title and ENABLE_TOP_HOOK_BADGE:
        clean_title = header_title.strip().upper()
        clean_title = re.sub(r'[^\w\s\-\'\,\.\?]', '', clean_title).strip()
        clean_title = re.sub(r'\s+', ' ', clean_title)

        words = clean_title.split()
        if len(words) >= 2:
            mid = (len(words) + 1) // 2
            clean_title = " ".join(words[:mid]) + "\\N" + " ".join(words[mid:])
        
        hook_dur = format_ass_timestamp(min(HOOK_BADGE_DURATION, clip_end - clip_start))
        lines.insert(0, f"Dialogue: 1,0:00:00.00,{hook_dur},TopHeader,,0,0,0,,{{\\fad(150,350)}}{clean_title}")

    full_content = header + "\n".join(lines) + "\n"

    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"[+] Styled ASS subtitle generated: {output_ass_path.name} (Theme: {theme['name']}, Words: {len(clip_words)})")
    return output_ass_path
