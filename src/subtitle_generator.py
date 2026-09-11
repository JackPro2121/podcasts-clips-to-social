import math
import re
from pathlib import Path
from typing import List, Optional, Any, Dict, Tuple
from src.config import SUBTITLE_THEMES, SUBTITLES_DIR, OUTPUT_WIDTH, OUTPUT_HEIGHT, CHANNEL_WATERMARK
from src.transcriber import TranscriptSegment, WordTimestamp

# Default lower-third safe-zone margins per layout (must match generate_ass_header).
_SAFE_MARGIN_V = {
    "split_screen": 0,    # centered on the divider (Alignment 5)
    "blur_stack": 400,    # just below the centered 16:9 panel
    "single_smooth": 460, # lower-third, clear of the bottom UI overlay
}
_DEFAULT_MARGIN_V = 460


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
    layout_mode: str = "single_smooth"
) -> str:
    """Generates ASS header with custom high-end styling and safe-zone margin."""
    theme = SUBTITLE_THEMES.get(theme_key, SUBTITLE_THEMES["hormozi"])

    font_name = theme["font_name"]
    font_size = theme["font_size"]
    primary_col = theme["primary_color"]
    outline_col = theme["outline_color"]
    outline_w = theme["outline_width"]
    shadow_col = theme["shadow_color"]
    shadow_d = theme["shadow_depth"]

    # Safe Zone Placement:
    # In split_screen: captions placed right at the middle horizontal divider (center alignment 5)
    # In single_smooth, dynamic_cut, or blur_stack: placed strictly in the lower-third safe zone (margin_v: 460)
    # This prevents any overlap with TikTok/Reels/Shorts bottom description, sound title, or comment button.
    if layout_mode == "split_screen":
        alignment = 5  # Middle Center
        margin_v = 0
    elif layout_mode == "blur_stack":
        alignment = 2  # Bottom Center
        margin_v = 400  # Perfectly below the 16:9 centered diagram (which ends at Y=1264)
    else:
        alignment = 2  # Bottom Center
        margin_v = 460  # Y = 1460px (76% height, perfectly above bottom 22% UI overlay)

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
Style: TopHeader,Montserrat Black,46,&H00FFFFFF,&H000000FF,&H00B86B62,&H00000000,-1,0,0,0,100,100,1.2,0,3,18,0,8,120,120,210,1
Style: Watermark,Arial,28,&H99FFFFFF,&H000000FF,&H99000000,&H00000000,-1,0,0,0,100,100,1.2,0,1,1.5,0.0,8,60,60,335,1

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
    shots: Optional[List[Any]] = None
) -> Path:
    """
    Generates word-level animated karaoke-style ASS subtitles for a specific clip window.
    Dynamically positions subtitles based on active shot layout (e.g. margin_v=420 during slides).
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
                clean_word = re.sub(r'^[^\w]+|[^\w]+$', '', w.word.strip())
                if not clean_word:
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

    # Group words into short punchy batches of 2-4 words
    lines: List[str] = []
    i = 0
    while i < len(clip_words):
        chunk = clip_words[i:i + max_words]
        i += max_words
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
                    word_elements.append(f"{{\\c{highlight_color}\\t(0,80,\\fscx112\\fscy112)\\t(80,160,\\fscx100\\fscy100)}}{w.word}{{\\c{primary_color}\\fscx100\\fscy100}}")
                else:
                    word_elements.append(w.word)

            dialogue_text = " ".join(word_elements)
            w_mid = (w_start + w_end) / 2
            active_margin_v = get_shot_margin_v(w_mid)
            ass_line = f"Dialogue: 0,{format_ass_timestamp(w_start)},{format_ass_timestamp(w_end)},Default,,0,0,{active_margin_v},,{dialogue_text}"
            lines.append(ass_line)

    header = generate_ass_header(theme_key=theme_key, layout_mode=layout_mode)
    dur_str = format_ass_timestamp(clip_end - clip_start)

    # Watermark (if configured)
    active_watermark = watermark if watermark is not None else CHANNEL_WATERMARK
    if active_watermark:
        lines.insert(0, f"Dialogue: 2,0:00:00.00,{dur_str},Watermark,,0,0,0,,{active_watermark.strip()}")

    # Hook title capsule badge: first 3.5 seconds, clean fade out, 0 emojis
    if header_title:
        clean_title = header_title.strip().upper()
        clean_title = re.sub(r'[^\w\s\-\'\,\.\?]', '', clean_title).strip()
        clean_title = re.sub(r'\s+', ' ', clean_title)

        words = clean_title.split()
        if len(words) >= 2:
            mid = (len(words) + 1) // 2
            clean_title = " ".join(words[:mid]) + "\\N" + " ".join(words[mid:])
        
        hook_dur = format_ass_timestamp(min(3.5, clip_end - clip_start))
        lines.insert(0, f"Dialogue: 1,0:00:00.00,{hook_dur},TopHeader,,0,0,0,,{{\\fad(200,400)}}{clean_title}")

    full_content = header + "\n".join(lines) + "\n"

    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"[+] Styled ASS subtitle generated: {output_ass_path.name} (Theme: {theme['name']}, Words: {len(clip_words)})")
    return output_ass_path
