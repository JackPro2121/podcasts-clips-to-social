import math
import re
from pathlib import Path
from typing import List, Optional
from src.config import SUBTITLE_THEMES, SUBTITLES_DIR, OUTPUT_WIDTH, OUTPUT_HEIGHT, CHANNEL_WATERMARK
from src.transcriber import TranscriptSegment, WordTimestamp

def format_ass_timestamp(seconds: float) -> str:
    """Converts seconds into ASS timestamp format: H:MM:SS.cs (centiseconds)."""
    if seconds < 0:
        seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        secs += 1
        centis = 0
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

    # Safe Zone MarginV:
    # In split_screen, place captions right along the middle divider (center alignment)
    # In single crop or blur stack, place in the lower-third safe zone (above TikTok bottom UI)
    if layout_mode == "split_screen":
        alignment = 5  # Middle Center
        margin_v = 0
    else:
        alignment = 2  # Bottom Center
        margin_v = 420  # Safe above TikTok/Reels bottom UI bar (380px)

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
Style: TopHeader,Trebuchet MS,46,&H00FFFFFF,&H000000FF,&H0084323B,&H00000000,-1,0,0,0,100,100,1.2,0,3,18,0,8,120,120,210,1
Style: Watermark,Arial,28,&H80FFFFFF,&H000000FF,&H80000000,&H00000000,-1,0,0,0,100,100,1.2,0,1,1.5,0.0,8,60,60,330,1

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
    watermark: Optional[str] = None
) -> Path:
    """
    Generates word-level animated karaoke-style ASS subtitles for a specific clip window.
    Only displays 2-4 words at a time for maximum retention and viewer attention.
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
                # Relative timestamp relative to the clip start
                rel_start = max(0.0, w.start - clip_start)
                rel_end = max(rel_start + 0.1, min(clip_end - clip_start, w.end - clip_start))
                # Strip leading and trailing punctuation (. , ! ? ; : " ' - _ ~ etc.)
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

    # Group words into short punchy batches of 2-4 words
    lines: List[str] = []
    i = 0
    while i < len(clip_words):
        chunk = clip_words[i:i + max_words]
        i += max_words
        if not chunk:
            continue

        # For each word in this chunk, generate an active highlight state
        for active_idx, target_word in enumerate(chunk):
            w_start = target_word.start
            if active_idx + 1 < len(chunk):
                w_end = chunk[active_idx + 1].start
            else:
                w_end = target_word.end

            # Build line text where active word is rendered in highlight_color
            word_elements = []
            for idx, w in enumerate(chunk):
                if idx == active_idx:
                    # Highlighted active spoken word with kinetic spring pop
                    word_elements.append(f"{{\\c{highlight_color}\\t(0,80,\\fscx112\\fscy112)\\t(80,160,\\fscx100\\fscy100)}}{w.word}{{\\c{primary_color}\\fscx100\\fscy100}}")
                else:
                    word_elements.append(w.word)

            dialogue_text = " ".join(word_elements)
            ass_line = f"Dialogue: 0,{format_ass_timestamp(w_start)},{format_ass_timestamp(w_end)},Default,,0,0,0,,{dialogue_text}"
            lines.append(ass_line)

    header = generate_ass_header(theme_key=theme_key, layout_mode=layout_mode)
    
    dur_str = format_ass_timestamp(clip_end - clip_start)

    # If channel watermark is provided or configured, burn it at 50% opacity
    active_watermark = watermark if watermark is not None else CHANNEL_WATERMARK
    if active_watermark:
        lines.insert(0, f"Dialogue: 2,0:00:00.00,{dur_str},Watermark,,0,0,0,,{active_watermark.strip()}")

    # If a viral hook title is provided, burn it persistently at the top safe zone
    if header_title:
        clean_title = header_title.strip().upper()
        # If title is long, wrap it at a balanced word boundary to fit cleanly inside 1080px portrait
        if len(clean_title) > 30 and " " in clean_title:
            words = clean_title.split()
            mid = len(words) // 2
            clean_title = " ".join(words[:mid]) + "\\N" + " ".join(words[mid:])
        
        # Clean title without star emojis
        lines.insert(0, f"Dialogue: 1,0:00:00.00,{dur_str},TopHeader,,0,0,0,,{clean_title}")

    full_content = header + "\n".join(lines) + "\n"

    output_ass_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(full_content)

    print(f"[+] Styled ASS subtitle generated: {output_ass_path.name} (Theme: {theme['name']}, Words: {len(clip_words)})")
    return output_ass_path
