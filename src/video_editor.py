import os
import subprocess
import re
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from src.config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT, FPS, VIDEO_CRF, AUDIO_BITRATE,
    TARGET_LUFS, TARGET_TRUE_PEAK, HIGHPASS_FREQ, VOCAL_PRESENCE_FREQ, VOCAL_AIR_FREQ, CLIPS_DIR,
    ENABLE_PUNCH_ZOOM, ENABLE_BGM, AUDIO_ASSETS_DIR, ENABLE_FILM_GRAIN, IS_CI
)
from src.face_tracker import FramingDecision, ShotPlan

def sanitize_ffmpeg_path(path: Path) -> str:
    """Escapes path for FFmpeg filter arguments across Windows and Unix."""
    p_str = str(path.resolve()).replace("\\", "/")
    # On Linux/Unix, the subtitles filter is extremely picky.
    # The most reliable way is to escape colons and single quotes.
    # But we also need to be careful about the overall wrapping.
    p_str = p_str.replace("'", r"\'").replace(":", r"\:")
    return p_str


def build_video_filtergraph(
    framing: FramingDecision,
    peak_intensity_segments: List[Tuple[float, float]] = [],
    ass_subtitle_path: Optional[Path] = None,
    burn_subtitles: bool = True
) -> str:
    """
    Constructs the complete FFmpeg video filtergraph with professional upgrades:
    - LERP Panning: Smoothly glides between face centers recorded in the timeline.
    - Sentiment-Driven Zoom: Subtle 1.2x zoom during peak intensity moments.
    - High-end Studio Mastering: Noise reduction, sharpening, and color grading.
    """
    filters = []

    if framing.video_height and framing.video_height < 720:
        studio_grade = "hqdn3d=2.0:2.0:4.0:4.0,unsharp=lx=7:ly=7:la=1.1:cx=5:cy=5:ca=0.50,cas=0.60,eq=contrast=1.09:brightness=0.01:saturation=1.15"
    else:
        studio_grade = "hqdn3d=1.5:1.5:3:3,unsharp=lx=5:ly=5:la=0.75:cx=3:cy=3:ca=0.40,cas=0.45,eq=contrast=1.07:brightness=0.01:saturation=1.12"

    if ENABLE_FILM_GRAIN:
        studio_grade += ",noise=alls=1.2:allf=t"

    # Dynamic Zoom Logic (1.2x zoom during peaks)
    # We use the 'zoompan' filter. Since zoompan is complex, we apply it as a 
    # pre-crop effect or a layered effect. For simplicity and stability in 
    # the $0 GH Action environment, we'll implement it via an expression in the crop filter
    # where possible, or a separate zoompan filter.
    

    if ENABLE_FILM_GRAIN:
        studio_grade += ",noise=alls=1.2:allf=t"

    # Dynamic Zoom Logic (1.2x zoom during peaks)
    # We use the 'zoompan' filter. Since zoompan is complex, we apply it as a 
    # pre-crop effect or a layered effect. For simplicity and stability in 
    # the $0 GH Action environment, we'll implement it via an expression in the crop filter
    # where possible, or a separate zoompan filter.
    
    zoom_expr = "1.0"
    if peak_intensity_segments:
        # Create a conditional zoom expression: if time is within any peak, zoom=1.2, else 1.0
        segments_logic = " || ".join([f"(between(t,{s[0]},{s[1]}))" for s in peak_intensity_segments])
        zoom_expr = f"if({segments_logic},1.2,1.0)"

    active_x = getattr(framing, "active_x", 0)
    active_y = getattr(framing, "active_y", 0)
    active_w = getattr(framing, "active_w", framing.video_width)
    active_h = getattr(framing, "active_h", framing.video_height)

    if framing.mode == "multi_shot_dynamic" and framing.shots:
        half_h = OUTPUT_HEIGHT // 2
        shot_filters = []
        shot_labels = []

        for i, shot in enumerate(framing.shots):
            label = f"v_shot_{i}"
            shot_labels.append(f"[{label}]")

            if shot.mode == "presentation_slide":
                s_cx = getattr(shot, "crop_x", None) or active_x
                s_cy = getattr(shot, "crop_y", None) or active_y
                s_cw = getattr(shot, "crop_w", None) or active_w
                s_ch = getattr(shot, "crop_h", None) or active_h
                shot_f = (
                    f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,crop={s_cw}:{s_ch}:{s_cx}:{s_cy},split=2[s{i}_fg_in][s{i}_bg_in];"
                    f"[s{i}_bg_in]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},boxblur=30:5,eq=brightness=-0.16:contrast=1.12[s{i}_bg];"
                    f"[s{i}_fg_in]scale={OUTPUT_WIDTH}:-1:force_original_aspect_ratio=decrease,{studio_grade}[s{i}_fg];"
                    f"[s{i}_bg][s{i}_fg]overlay=(W-w)/2:(H-h)/2,setsar=1:1,fps={FPS}[{label}]"
                )
            elif shot.mode == "split_screen" and shot.speaker1_box and shot.speaker2_box:
                s1_x, s1_y, s1_w, s1_h = shot.speaker1_box
                s2_x, s2_y, s2_w, s2_h = shot.speaker2_box
                # Crop each speaker at native resolution, then scale each pane to half-height.
                # This preserves the tight face-relative crop boxes from face_tracker.
                shot_f = (
                    f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,split=2[s{i}_p1][s{i}_p2];"
                    f"[s{i}_p1]crop={s1_w}:{s1_h}:{s1_x}:{s1_y},scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd[s{i}_top];"
                    f"[s{i}_p2]crop={s2_w}:{s2_h}:{s2_x}:{s2_y},scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd[s{i}_bot];"
                    f"[s{i}_top][s{i}_bot]vstack=inputs=2,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}[{label}]"
                )
            else:
                s_cy = getattr(shot, "crop_y", None)
                if s_cy is None:
                    s_cy = active_y
                s_ch = getattr(shot, "crop_h", None) or active_h
                target_crop_w = int(s_ch * (9 / 16))

                if shot.face_centers_timeline:
                    points = shot.face_centers_timeline
                    t_start, x_start = points[0]
                    t_end, x_end = points[-1]
                    duration = t_end - t_start if t_end > t_start else 1.0
                    
                    cx_expr = f"{x_start} + ({x_end}-{x_start})*(t-{t_start})/{duration}"
                    crop_x_expr = f"max({active_x},min({cx_expr}-{target_crop_w//2},{active_x + active_w}-{target_crop_w}))"
                    
                    shot_f = (
                        f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,"
                        f"crop={target_crop_w}:{s_ch}:'{crop_x_expr}':{s_cy},"
                        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}[{label}]"
                    )
                else:
                    crop_x = max(active_x, min(shot.crop_x, active_x + active_w - target_crop_w))
                    shot_f = (
                        f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,"
                        f"crop={target_crop_w}:{s_ch}:{crop_x}:{s_cy},"
                        f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}[{label}]"
                    )
            shot_filters.append(shot_f)

        concat_inputs = "".join(shot_labels)
        concat_f = f"{concat_inputs}concat=n={len(shot_labels)}:v=1:a=0,{studio_grade}[base]"
        v_filter = ";".join(shot_filters) + ";" + concat_f

    elif framing.mode in ("single_smooth", "dynamic_cut"):
        crop_w = int(active_h * (9 / 16))
        if framing.mode == "dynamic_cut" and framing.crop_x_expr:
            crop_x = f"'{framing.crop_x_expr}'"
        else:
            cx = framing.smoothed_center_x or (active_x + active_w // 2)
            crop_x = max(active_x, min(cx - crop_w // 2, active_x + active_w - crop_w))
        v_filter = f"[0:v]crop={crop_w}:{active_h}:{crop_x}:{active_y},scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,{studio_grade},fps={FPS}[base]"

    elif framing.mode == "split_screen" and framing.speaker1_box and framing.speaker2_box:
        s1_x, s1_y, s1_w, s1_h = framing.speaker1_box
        s2_x, s2_y, s2_w, s2_h = framing.speaker2_box
        half_h = OUTPUT_HEIGHT // 2
        # Crop at native resolution — face boxes are already in source pixel coordinates.
        # Each pane: tight face crop → scale to full output width × half output height.
        v_filter = (
            f"[0:v]split=2[s1_in][s2_in];"
            f"[s1_in]crop={s1_w}:{s1_h}:{s1_x}:{s1_y},scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd,{studio_grade}[top_pane];"
            f"[s2_in]crop={s2_w}:{s2_h}:{s2_x}:{s2_y},scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd,{studio_grade}[bottom_pane];"
            f"[top_pane][bottom_pane]vstack=inputs=2,fps={FPS}[base]"
        )
    else:
        v_filter = (
            f"[0:v]crop={active_w}:{active_h}:{active_x}:{active_y},split=2[bg_in][fg_in];"
            f"[bg_in]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
            f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},boxblur=30:5,eq=brightness=-0.16:contrast=1.12[bg];"
            f"[fg_in]scale={OUTPUT_WIDTH}:-1:force_original_aspect_ratio=decrease,{studio_grade}[fg];"
            f"[bg][fg]overlay=0:(H-h)/2,fps={FPS}[base]"
        )

    filters.append(v_filter) 
    # conditionally or a crop that slightly shrinks the window.
    # For professional result, we apply a subtle zoom overlay or we use the 
    # zoompan filter at the end of the base.
    if peak_intensity_segments and not IS_CI:
        # We'll use a subtle zoompan: z=if(condition, 1.2, 1.0)
        # Note: zoompan is very picky about resolution. We apply it to the scaled base.
        # This is a simplified version that achieves the 1.2x effect.
        segments_logic = " || ".join([f"between(it,{s[0]},{s[1]})" for s in peak_intensity_segments])
        # z=zoom factor, x and y are centers. We keep center since base is already cropped.
        zoom_f = f"[base_in]zoompan=z='if({segments_logic},1.2,1.0)':d=1:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT},fps={FPS}[zoomed]"
        filters[-1] = filters[-1].replace("[base]", "[base_in]") 
        filters[-1] = filters[-1] + f";{zoom_f}"
        base_label = "[zoomed]"
    else:
        base_label = "[base]"


    if burn_subtitles and ass_subtitle_path and ass_subtitle_path.exists():
        escaped_ass = sanitize_ffmpeg_path(ass_subtitle_path)
        sub_filter = f"{base_label}subtitles='{escaped_ass}'[outv]"
        filters.append(sub_filter)
    else:
        filters.append(f"{base_label}null[outv]")

    return ";".join(filters)

def build_audio_processing_filters() -> str:
    return (
        f"aresample=48000,highpass=f={HIGHPASS_FREQ},"
        f"equalizer=f={VOCAL_PRESENCE_FREQ}:width_type=h:width=1000:g=2.5,"
        f"equalizer=f={VOCAL_AIR_FREQ}:width_type=h:width=2500:g=1.8"
    )

def build_audio_mastering_filters() -> str:
    return f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA=11"

def build_audio_filtergraph() -> str:
    return f"{build_audio_processing_filters()},{build_audio_mastering_filters()}"

def render_viral_clip(
    source_video_path: Path,
    start_time: float,
    end_time: float,
    output_clip_path: Path,
    framing: FramingDecision,
    peak_intensity_segments: List[Tuple[float, float]] = [],
    ass_subtitle_path: Optional[Path] = None,
    burn_subtitles: bool = True,
    bgm_path: Optional[Path] = None
) -> Path:
    duration = end_time - start_time
    output_clip_path.parent.mkdir(parents=True, exist_ok=True)

    video_filters = build_video_filtergraph(
        framing=framing,
        peak_intensity_segments=peak_intensity_segments,
        ass_subtitle_path=ass_subtitle_path,
        burn_subtitles=burn_subtitles
    )
    audio_filters = build_audio_filtergraph()

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
