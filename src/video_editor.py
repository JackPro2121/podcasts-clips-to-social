import os
import subprocess
import re
from pathlib import Path
from typing import Optional, Dict, Any
from src.config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT, FPS, VIDEO_CRF, AUDIO_BITRATE,
    TARGET_LUFS, TARGET_TRUE_PEAK, HIGHPASS_FREQ, VOCAL_PRESENCE_FREQ, CLIPS_DIR
)
from src.face_tracker import FramingDecision

def sanitize_ffmpeg_path(path: Path) -> str:
    """Escapes path for FFmpeg filter arguments across Windows and Unix."""
    p_str = str(path.resolve()).replace("\\", "/")
    # Escape colon for drive letter on Windows (e.g. C\: -> C\\:)
    p_str = re.sub(r"^([A-Za-z]):", r"\1\\:", p_str)
    # Escape single quotes and brackets
    p_str = p_str.replace("'", "\\'").replace("[", "\\[").replace("]", "\\]")
    return p_str

def build_video_filtergraph(
    framing: FramingDecision,
    ass_subtitle_path: Optional[Path] = None,
    burn_subtitles: bool = True
) -> str:
    """
    Constructs the complete FFmpeg video filtergraph based on the framing decision:
    - single_smooth: Face-centered 9:16 crop
    - split_screen: Dual-speaker stacked layout (Host top / Guest bottom)
    - blur_stack: Full 16:9 centered over ambient blurred/darkened background
    """
    filters = []

    if framing.mode == "single_smooth":
        # Target aspect ratio 9:16
        crop_w = int(framing.video_height * (9 / 16))
        # Ensure center_x keeps crop window within bounds
        cx = framing.smoothed_center_x or (framing.video_width // 2)
        crop_x = max(0, min(cx - crop_w // 2, framing.video_width - crop_w))
        v_filter = f"[0:v]crop={crop_w}:{framing.video_height}:{crop_x}:0,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos,fps={FPS}[base]"

    elif framing.mode == "split_screen" and framing.speaker1_box and framing.speaker2_box:
        # Split-screen stack: Top pane (1080x960), Bottom pane (1080x960)
        s1_x, s1_y, s1_w, s1_h = framing.speaker1_box
        s2_x, s2_y, s2_w, s2_h = framing.speaker2_box
        half_h = OUTPUT_HEIGHT // 2

        v_filter = (
            f"[0:v]split=2[s1_in][s2_in];"
            f"[s1_in]crop={s1_w}:{s1_h}:{s1_x}:0,scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos[top_pane];"
            f"[s2_in]crop={s2_w}:{s2_h}:{s2_x}:0,scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos[bottom_pane];"
            f"[top_pane][bottom_pane]vstack=inputs=2,fps={FPS}[base]"
        )

    else:
        # Blur Stack (Safe default for multi-person panels or wide shots)
        # Background: scale & crop to 1080x1920, heavy boxblur, darkened
        # Foreground: crisp 16:9 centered at 1080x608
        fg_height = int(OUTPUT_WIDTH * (9 / 16))  # 608px
        fg_y = (OUTPUT_HEIGHT - fg_height) // 2   # 656px
        v_filter = (
            f"[0:v]split=2[bg_in][fg_in];"
            f"[bg_in]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},boxblur=25:5,eq=brightness=-0.15:contrast=0.9[bg];"
            f"[fg_in]scale={OUTPUT_WIDTH}:{fg_height}:flags=lanczos[fg];"
            f"[bg][fg]overlay=0:{fg_y},fps={FPS}[base]"
        )

    filters.append(v_filter)

    # Subtitle burn-in layer
    if burn_subtitles and ass_subtitle_path and ass_subtitle_path.exists():
        escaped_ass = sanitize_ffmpeg_path(ass_subtitle_path)
        sub_filter = f"[base]subtitles='{escaped_ass}'[outv]"
        filters.append(sub_filter)
    else:
        # Pass base directly to output
        filters.append("[base]null[outv]")

    return ";".join(filters)

def build_audio_filtergraph() -> str:
    """
    Constructs the broadcast audio mastering chain:
    1. High-pass filter (80Hz) to cut desk bumps/low rumble
    2. Vocal presence lift (3000Hz)
    3. EBU R128 loudness normalization (-14 LUFS, -1.5 dBTP)
    """
    return (
        f"highpass=f={HIGHPASS_FREQ},"
        f"equalizer=f={VOCAL_PRESENCE_FREQ}:width_type=h:width=1000:g=2.5,"
        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA=11"
    )

def render_viral_clip(
    source_video_path: Path,
    start_time: float,
    end_time: float,
    output_clip_path: Path,
    framing: FramingDecision,
    ass_subtitle_path: Optional[Path] = None,
    burn_subtitles: bool = True
) -> Path:
    """
    Executes FFmpeg with exact start/end cut, audio mastering, video framing,
    and subtitle burning.
    """
    duration = end_time - start_time
    output_clip_path.parent.mkdir(parents=True, exist_ok=True)

    video_filters = build_video_filtergraph(
        framing=framing,
        ass_subtitle_path=ass_subtitle_path,
        burn_subtitles=burn_subtitles
    )
    audio_filters = build_audio_filtergraph()

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_time:.2f}",
        "-t", f"{duration:.2f}",
        "-i", str(source_video_path),
        "-filter_complex", video_filters,
        "-map", "[outv]",
        "-filter:a", audio_filters,
        "-c:v", "libx264",
        "-crf", str(VIDEO_CRF),
        "-preset", "medium",
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        "-movflags", "+faststart",
        str(output_clip_path)
    ]

    print(f"[*] Rendering viral clip with FFmpeg: {output_clip_path.name} (Framing: {framing.mode})...")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(f"[-] FFmpeg error:\n{result.stderr[-1000:]}")
        raise RuntimeError(f"FFmpeg failed to render clip {output_clip_path.name}")

    print(f"[+] Clip rendered successfully: {output_clip_path} (Size: {output_clip_path.stat().st_size / (1024*1024):.2f} MB)")
    return output_clip_path
