import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple
from src.config import (
    SUBTITLE_THEMES, OUTPUT_WIDTH, OUTPUT_HEIGHT, CHANNEL_WATERMARK,
    ENABLE_TOP_HOOK_BADGE, HOOK_BADGE_DURATION, HOOK_BADGE_MARGIN_V,
    get_subtitle_margin_v
)
from src.transcriber import TranscriptSegment, WordTimestamp

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


def normalize_caption_text(text: Any) -> str:
    normalized = unicodedata.normalize("NFC", str(text))
    cleaned: List[str] = []
    for char in normalized:
        codepoint = ord(char)
        if char in "\r\n\t":
            cleaned.append(" ")
            continue
        if codepoint in (0x200D, 0xFE0F, 0xFE0E):
            continue
        if 0x1F000 <= codepoint <= 0x1FAFF or 0x2600 <= codepoint <= 0x27BF:
            continue
        if unicodedata.category(char).startswith("C"):
            continue
        cleaned.append(char)
    return " ".join("".join(cleaned).split())


def escape_ass_text(text: Any) -> str:
    normalized = normalize_caption_text(text)
    return normalized.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def sanitize_watermark(text: Any) -> str:
    return escape_ass_text(text).strip()


def default_margin_v(layout_mode: str) -> int:
    """Safe-zone MarginV for a layout when no per-shot override applies."""
    return get_subtitle_margin_v(layout_mode)


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
    outline_w = min(float(theme["outline_width"]), 4.0)
    shadow_col = theme["shadow_color"]
    shadow_d = min(float(theme["shadow_depth"]), 2.5)

    alignment = 2
    margin_v = get_subtitle_margin_v(layout_mode)

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
    max_words = max(2, min(4, int(theme.get("max_words_per_line", 3))))
    uppercase = theme.get("uppercase", True)
    highlight_color = theme.get("highlight_color", "&H0000E6FF")

    # Filter words strictly within the clip duration and offset timestamps to 0.0s
    clip_duration = max(0.0, clip_end - clip_start)
    clip_words: List[WordTimestamp] = []
    has_word_timestamps = any(segment.words for segment in segments)
    for seg in segments:
        for w in seg.words:
            if w.end <= clip_start or w.start >= clip_end or w.end > clip_end + 0.05:
                continue
            rel_start = max(0.0, min(clip_duration, w.start - clip_start))
            rel_end = max(rel_start, min(clip_duration, w.end - clip_start))
            if rel_end <= rel_start:
                continue
            if rel_end - rel_start < 0.1:
                rel_end = min(clip_duration, rel_start + 0.1)
            if rel_end <= rel_start:
                continue
            clean_word = normalize_caption_text(w.word).strip(" \t.,!?;:'\"()[]{}-")
            if not clean_word or not any(c.isalnum() for c in clean_word):
                continue
            clean_text = clean_word.upper() if uppercase else clean_word
            clip_words.append(WordTimestamp(
                word=clean_text,
                start=rel_start,
                end=rel_end,
                is_estimated=bool(getattr(w, "is_estimated", False)),
            ))

    if not clip_words and not has_word_timestamps:
        for seg in segments:
            rel_start = max(0.0, min(clip_duration, seg.start - clip_start))
            rel_end = max(rel_start, min(clip_duration, seg.end - clip_start))
            text = normalize_caption_text(seg.text)
            if text and rel_end > rel_start:
                clip_words.append(WordTimestamp(
                    word=text.upper() if uppercase else text,
                    start=rel_start,
                    end=rel_end,
                    is_estimated=True,
                ))

    # Sort words by start time and enforce strictly monotonic, non-overlapping timestamps
    clip_words.sort(key=lambda x: x.start)
    monotonic_words: List[WordTimestamp] = []
    last_end = 0.0
    for w in clip_words:
        w_start = max(last_end, min(clip_duration, w.start))
        if w_start >= clip_duration:
            break
        w_end = min(clip_duration, max(w_start, w.end))
        if w_end - w_start < 0.12:
            w_end = min(clip_duration, w_start + 0.12)
        if w_end <= w_start:
            continue
        monotonic_words.append(WordTimestamp(
            word=w.word,
            start=w_start,
            end=w_end,
            is_estimated=bool(getattr(w, "is_estimated", False)),
        ))
        last_end = w_end
    clip_words = monotonic_words

    clip_dur = max(1.0, clip_end - clip_start)
    wpm = (len(clip_words) / clip_dur) * 60.0
    effective_max_words = max_words
    if wpm > 190 and len(clip_words) > 10:
        effective_max_words = min(4, effective_max_words + 1)

    base_margin_v = default_margin_v(layout_mode)

    def get_shot_style(t: float) -> Tuple[int, int, str]:
        if shots:
            for s in shots:
                s_start = getattr(s, "start", 0.0)
                s_end = getattr(s, "end", 9999.0)
                if s_start <= t <= s_end:
                    placement = getattr(s, "subtitle_placement", "lower_third")
                    if placement == "divider":
                        return 0, int(getattr(s, "subtitle_alignment", 5)), placement
                    margin = get_subtitle_margin_v(layout_mode, getattr(s, "margin_v", None))
                    return margin, int(getattr(s, "subtitle_alignment", 2)), placement
        return base_margin_v, 2, "lower_third"

    def word_color(word: WordTimestamp) -> str:
        clean_lower = re.sub(r'[^a-z0-9]', '', word.word.lower())
        if clean_lower in MONEY_KEYWORDS or any(c.isdigit() for c in word.word) or '$' in word.word:
            return "&H0033FF22"
        if clean_lower in DANGER_KEYWORDS:
            return "&H003333FF"
        if clean_lower in POWER_KEYWORDS:
            return "&H0000D7FF"
        return highlight_color

    lines: List[str] = []
    i = 0
    while i < len(clip_words):
        chunk: List[WordTimestamp] = []
        remaining_words = len(clip_words) - i
        chunk_limit = 2 if remaining_words == effective_max_words + 1 else effective_max_words
        while i < len(clip_words) and len(chunk) < chunk_limit:
            candidate = clip_words[i]
            if chunk and candidate.start - chunk[-1].end >= 1.2:
                break
            chunk.append(candidate)
            i += 1
        if not chunk:
            continue

        w_start = chunk[0].start
        w_end = chunk[-1].end
        animation_end_ms = max(70, min(140, int(round((w_end - w_start) * 1000))))
        animation_mid_ms = max(35, animation_end_ms // 2)
        word_elements = [
            f"{{\\c{word_color(word)}}}{escape_ass_text(word.word)}"
            for word in chunk
        ]
        dialogue_text = " ".join(word_elements)
        animation_prefix = (
            f"{{\\fad(0,100)\\bord3\\shad2"
            f"\\t(0,{animation_mid_ms},\\fscx118\\fscy118)"
            f"\\t({animation_mid_ms},{animation_end_ms},\\fscx100\\fscy100)}}"
        )
        w_mid = (w_start + w_end) / 2
        active_margin_v, active_alignment, placement = get_shot_style(w_mid)
        placement_prefix = f"{{\\an{active_alignment}}}" if placement == "divider" else ""
        event_name = "estimated" if any(getattr(word, "is_estimated", False) for word in chunk) else ""
        ass_line = f"Dialogue: 0,{format_ass_timestamp(w_start)},{format_ass_timestamp(w_end)},Default,{event_name},0,0,{active_margin_v},,{placement_prefix}{animation_prefix}{dialogue_text}"
        lines.append(ass_line)

    has_badge = bool(header_title and ENABLE_TOP_HOOK_BADGE)
    if has_badge:
        clean_words = normalize_caption_text(header_title or "").upper().split()
        num_lines = 2 if len(clean_words) >= 2 else 1
        w_margin_v = HOOK_BADGE_MARGIN_V + (95 if num_lines == 2 else 55)
    else:
        w_margin_v = HOOK_BADGE_MARGIN_V

    header = generate_ass_header(
        theme_key=theme_key,
        layout_mode=layout_mode,
        watermark_margin_v=w_margin_v,
        font_size_override=None
    )
    dur_str = format_ass_timestamp(clip_end - clip_start)

    # Watermark (if configured)
    active_watermark = watermark if watermark is not None else CHANNEL_WATERMARK
    if active_watermark:
        lines.insert(0, f"Dialogue: 2,0:00:00.00,{dur_str},Watermark,,0,0,0,,{sanitize_watermark(active_watermark)}")

    # Hook title capsule badge: initial hook retention, smooth fade out, clean typography
    if header_title and ENABLE_TOP_HOOK_BADGE:
        clean_title = normalize_caption_text(header_title).upper()
        words = clean_title.split()
        if len(words) >= 2:
            mid = (len(words) + 1) // 2
            clean_title = escape_ass_text(" ".join(words[:mid])) + "\\N" + escape_ass_text(" ".join(words[mid:]))
        else:
            clean_title = escape_ass_text(clean_title)

        hook_dur = format_ass_timestamp(min(HOOK_BADGE_DURATION, clip_end - clip_start))
        lines.insert(0, f"Dialogue: 1,0:00:00.00,{hook_dur},TopHeader,,0,0,0,,{{\\fad(150,350)}}{clean_title}")

    full_content = header + "\n".join(lines) + "\n"

    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"[+] Styled ASS subtitle generated: {output_ass_path.name} (Theme: {theme['name']}, Words: {len(clip_words)})")
    return output_ass_path
