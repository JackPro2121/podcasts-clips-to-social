import os
import subprocess
import re
from pathlib import Path
from typing import Optional, Dict, Any
from src.config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT, FPS, VIDEO_CRF, AUDIO_BITRATE,
    TARGET_LUFS, TARGET_TRUE_PEAK, HIGHPASS_FREQ, VOCAL_PRESENCE_FREQ, VOCAL_AIR_FREQ, CLIPS_DIR,
    ENABLE_PUNCH_ZOOM, ENABLE_BGM, AUDIO_ASSETS_DIR
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
    - single_smooth: Face-centered 9:16 crop with subtle dynamic punch zoom
    - split_screen: Dual-speaker stacked layout (Host top / Guest bottom)
    - blur_stack: Full 16:9 centered over ambient blurred/darkened background
    """
    filters = []

    # Multi-Stage Broadcast Studio Enhancement & Super-Resolution Chain:
    if framing.video_height and framing.video_height < 720:
        # Aggressive Super-Resolution AI Enhancement for < 720p sources (e.g. 360p / 480p):
        # 1. hqdn3d=2.0:2.0:4.0:4.0 - Eliminates macroblocking and compression artifacts
        # 2. cas=0.60 - High AMD FidelityFX Contrast Adaptive Sharpening restores lost edges & facial details
        # 3. unsharp=lx=7:ly=7:la=1.1:cx=5:cy=5:ca=0.50 - Sharpens micro-contours & text
        # 4. eq=contrast=1.09:brightness=0.01:saturation=1.15 - High-vibrancy OLED mobile grade
        studio_grade = "hqdn3d=2.0:2.0:4.0:4.0,cas=0.60,unsharp=lx=7:ly=7:la=1.1:cx=5:cy=5:ca=0.50,eq=contrast=1.09:brightness=0.01:saturation=1.15"
    else:
        # Broadcast Studio Mastering for >= 720p / 1080p native HD sources:
        # 1. hqdn3d=1.5:1.5:3:3 - Fine luma/chroma noise cleanup
        # 2. cas=0.45 - Subtle AMD FidelityFX Contrast Adaptive Sharpening for razor-sharp edges without halos
        # 3. unsharp=lx=5:ly=5:la=0.75:cx=3:cy=3:ca=0.40 - High-frequency facial & eye clarity
        # 4. eq=contrast=1.07:brightness=0.01:saturation=1.12 - Balanced broadcast color grade
        studio_grade = "hqdn3d=1.5:1.5:3:3,cas=0.45,unsharp=lx=5:ly=5:la=0.75:cx=3:cy=3:ca=0.40,eq=contrast=1.07:brightness=0.01:saturation=1.12"

    # Intelligent Dynamic Punch Zoom (Alex Hormozi / Diary of a CEO style):
    # Starts at 1.0x NORMAL wide crop for the first 4.0s (anchors viewer).
    # Cuts cleanly into 1.12x PUNCH ZOOM on the speaker for 3.0s (peaks retention).
    # Then returns to 1.0x normal crop, alternating every 9.0s cycle.
    # Scaled and centered smoothly so the speaker's eyes remain in the upper-third golden ratio.
    punch_zoom = ",crop='if(between(mod(t,9),4.0,7.0),964,1080)':'if(between(mod(t,9),4.0,7.0),1714,1920)':(iw-ow)/2:(ih-oh)*0.35,scale=1080:1920:flags=lanczos+accurate_rnd" if ENABLE_PUNCH_ZOOM else ""

    if framing.mode == "multi_shot_dynamic" and framing.shots:
        fg_height = int(round(OUTPUT_WIDTH * (9 / 16) / 2) * 2)  # 608px (even number)
        fg_y = (OUTPUT_HEIGHT - fg_height) // 2   # 656px
        target_crop_w = int(framing.video_height * (9 / 16))
        half_h = OUTPUT_HEIGHT // 2

        shot_filters = []
        shot_labels = []

        for i, shot in enumerate(framing.shots):
            label = f"v_shot_{i}"
            shot_labels.append(f"[{label}]")

            if shot.mode == "presentation_slide":
                # Presentation slide: 100% full uncropped 16:9 on ambient blurred background
                shot_f = (
                    f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,split=2[s{i}_fg_in][s{i}_bg_in];"
                    f"[s{i}_bg_in]scale=270:480,boxblur=8:2,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}[s{i}_bg];"
                    f"[s{i}_fg_in]scale={OUTPUT_WIDTH}:{fg_height}:flags=lanczos+accurate_rnd,{studio_grade}[s{i}_fg];"
                    f"[s{i}_bg][s{i}_fg]overlay=0:{fg_y},scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}[{label}]"
                )
            elif shot.mode == "split_screen" and shot.speaker1_box and shot.speaker2_box:
                s1_x, s1_y, s1_w, s1_h = shot.speaker1_box
                s2_x, s2_y, s2_w, s2_h = shot.speaker2_box
                shot_f = (
                    f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,split=2[s{i}_p1][s{i}_p2];"
                    f"[s{i}_p1]crop={s1_w}:{s1_h}:{s1_x}:0,scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd[s{i}_top];"
                    f"[s{i}_p2]crop={s2_w}:{s2_h}:{s2_x}:0,scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd[s{i}_bot];"
                    f"[s{i}_top][s{i}_bot]vstack=inputs=2,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}[{label}]"
                )
            else:
                # Full 9:16 portrait on speaker
                crop_x = max(0, min(shot.crop_x, framing.video_width - target_crop_w))
                shot_f = (
                    f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,"
                    f"crop={target_crop_w}:{framing.video_height}:{crop_x}:0,"
                    f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}[{label}]"
                )
            shot_filters.append(shot_f)

        concat_inputs = "".join(shot_labels)
        concat_f = f"{concat_inputs}concat=n={len(shot_labels)}:v=1:a=0,{studio_grade}[base]"
        v_filter = ";".join(shot_filters) + ";" + concat_f

    elif framing.mode == "dynamic_cut" and framing.crop_x_expr:
        # Target aspect ratio 9:16 with dynamic multi-camera angle switching
        crop_w = int(framing.video_height * (9 / 16))
        v_filter = f"[0:v]crop={crop_w}:{framing.video_height}:'{framing.crop_x_expr}':0,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,{studio_grade}{punch_zoom},fps={FPS}[base]"

    elif framing.mode == "single_smooth":
        # Target aspect ratio 9:16 with intelligent eye-level framing
        crop_w = int(framing.video_height * (9 / 16))
        # Ensure center_x keeps crop window within bounds
        cx = framing.smoothed_center_x or (framing.video_width // 2)
        crop_x = max(0, min(cx - crop_w // 2, framing.video_width - crop_w))
        v_filter = f"[0:v]crop={crop_w}:{framing.video_height}:{crop_x}:0,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,{studio_grade}{punch_zoom},fps={FPS}[base]"

    elif framing.mode == "split_screen" and framing.speaker1_box and framing.speaker2_box:
        # Split-screen stack: Top pane (1080x960), Bottom pane (1080x960)
        s1_x, s1_y, s1_w, s1_h = framing.speaker1_box
        s2_x, s2_y, s2_w, s2_h = framing.speaker2_box
        half_h = OUTPUT_HEIGHT // 2

        v_filter = (
            f"[0:v]split=2[s1_in][s2_in];"
            f"[s1_in]crop={s1_w}:{s1_h}:{s1_x}:0,scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd,{studio_grade}[top_pane];"
            f"[s2_in]crop={s2_w}:{s2_h}:{s2_x}:0,scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd,{studio_grade}[bottom_pane];"
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
            f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},boxblur=30:5,eq=brightness=-0.16:contrast=1.12[bg];"
            f"[fg_in]scale={OUTPUT_WIDTH}:{fg_height}:flags=lanczos+accurate_rnd,{studio_grade}[fg];"
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
    3. Crystal air shelf (10000Hz) for studio podcast sheen
    4. EBU R128 loudness normalization (-14 LUFS, -1.5 dBTP)
    """
    return (
        f"highpass=f={HIGHPASS_FREQ},"
        f"equalizer=f={VOCAL_PRESENCE_FREQ}:width_type=h:width=1000:g=2.5,"
        f"equalizer=f={VOCAL_AIR_FREQ}:width_type=h:width=2500:g=1.8,"
        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA=11"
    )

def render_viral_clip(
    source_video_path: Path,
    start_time: float,
    end_time: float,
    output_clip_path: Path,
    framing: FramingDecision,
    ass_subtitle_path: Optional[Path] = None,
    burn_subtitles: bool = True,
    bgm_path: Optional[Path] = None
) -> Path:
    """
    Executes FFmpeg with exact start/end cut, audio mastering, video framing,
    subtitle burning, and optional ambient BGM ducking.
    """
    duration = end_time - start_time
    output_clip_path.parent.mkdir(parents=True, exist_ok=True)

    video_filters = build_video_filtergraph(
        framing=framing,
        ass_subtitle_path=ass_subtitle_path,
        burn_subtitles=burn_subtitles
    )
    audio_filters = build_audio_filtergraph()

    # Detect if source has audio
    has_audio = True
    try:
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(source_video_path)]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True)
        has_audio = "audio" in probe_res.stdout
    except Exception:
        has_audio = True

    active_bgm = bgm_path or (AUDIO_ASSETS_DIR / "ambient_lofi_loop.mp3")
    use_bgm = ENABLE_BGM and active_bgm and active_bgm.exists()

    input_args = ["-ss", f"{start_time:.2f}", "-t", f"{duration:.2f}", "-i", str(source_video_path)]
    if use_bgm:
        input_args.extend(["-stream_loop", "-1", "-i", str(active_bgm)])

    if has_audio and use_bgm:
        combined_filter = (
            f"{video_filters};"
            f"[0:a]highpass=f={HIGHPASS_FREQ},equalizer=f={VOCAL_PRESENCE_FREQ}:width_type=h:width=1000:g=2.5,equalizer=f={VOCAL_AIR_FREQ}:width_type=h:width=2500:g=1.8[voice];"
            f"[1:a]volume=-22dB[bgm_duck];"
            f"[voice][bgm_duck]amix=inputs=2:duration=first:dropout_transition=2,loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA=11[outa]"
        )
        map_args = ["-map", "[outv]", "-map", "[outa]"]
    elif has_audio:
        combined_filter = f"{video_filters};[0:a]{audio_filters}[outa]"
        map_args = ["-map", "[outv]", "-map", "[outa]"]
    else:
        combined_filter = video_filters
        map_args = ["-map", "[outv]"]

    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", combined_filter,
        *map_args,
        "-c:v", "libx264",
        "-crf", str(VIDEO_CRF),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        "-ar", "48000",
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
