import os
import subprocess
import math
import json
import uuid
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from src.config import (
    OUTPUT_WIDTH, OUTPUT_HEIGHT, FPS, VIDEO_CRF, AUDIO_BITRATE,
    TARGET_LUFS, TARGET_TRUE_PEAK, HIGHPASS_FREQ, VOCAL_PRESENCE_FREQ, VOCAL_AIR_FREQ, ENABLE_BGM, AUDIO_ASSETS_DIR, ENABLE_FILM_GRAIN, IS_CI,
    SFX_ASSETS_DIR, ENABLE_SFX, ENABLE_DYNAMIC_DUCKING
)
from src.face_tracker import FramingDecision

def sanitize_ffmpeg_path(path: Path) -> str:
    """Escapes path for FFmpeg filter arguments across Windows and Unix."""
    p_str = str(path.resolve()).replace("\\", "/")
    # On Linux/Unix, the subtitles filter is extremely picky.
    # The most reliable way is to escape colons and single quotes.
    # But we also need to be careful about the overall wrapping.
    p_str = p_str.replace("'", r"\'").replace(":", r"\:")
    return p_str


def _probe_source_duration(source_video_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(source_video_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except (TypeError, ValueError):
        return None
    return duration if math.isfinite(duration) and duration > 0 else None


def _measure_loudness(path: Path) -> Optional[Dict[str, float]]:
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
                "-map", "0:a:0", "-af",
                f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA=11:print_format=json",
                "-f", "null", "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    payload_start = result.stderr.rfind("{")
    payload_end = result.stderr.rfind("}")
    if payload_start < 0 or payload_end <= payload_start:
        return None
    try:
        payload = json.loads(result.stderr[payload_start:payload_end + 1])
        measurement = {
            "input_i": float(payload["input_i"]),
            "input_tp": float(payload["input_tp"]),
            "input_lra": float(payload["input_lra"]),
            "input_thresh": float(payload["input_thresh"]),
            "target_offset": float(payload["target_offset"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if not all(math.isfinite(value) for value in measurement.values()):
        return None
    return measurement


def _measured_loudnorm_filter(
    measurement: Dict[str, float],
    limiter_limit: float = 0.80,
    linear: bool = True,
) -> str:
    return (
        f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA=11:"
        f"measured_I={measurement['input_i']}:"
        f"measured_TP={measurement['input_tp']}:"
        f"measured_LRA={measurement['input_lra']}:"
        f"measured_thresh={measurement['input_thresh']}:"
        f"offset={measurement['target_offset']}:linear={'true' if linear else 'false'},"
        f"alimiter=limit={limiter_limit:.4f}:attack=2:release=60:level=0"
    )


_LOUDNESS_PASS_LADDER = (
    (True, 0.80),
    (False, 0.80),
    (False, 0.71),
    (False, 0.63),
)


def _apply_measured_loudness(
    source_path: Path,
    destination_path: Path,
    measurement: Dict[str, float],
    limiter_limit: float = 0.80,
    linear: bool = True,
) -> None:
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(source_path),
            "-map", "0:v:0", "-map", "0:a:0", "-c:v", "copy",
            "-af", _measured_loudnorm_filter(measurement, limiter_limit, linear),
            "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-ar", "48000",
            "-movflags", "+faststart", str(destination_path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Measured loudness pass failed: {result.stderr[-1500:]}")


def _validate_rendered_output(path: Path, expected_duration: float) -> None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_streams", "-show_format",
                "-of", "json", str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        raise RuntimeError(f"Output validation probe failed for {path.name}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"Output validation probe failed for {path.name}: {result.stderr[-500:]}")
    try:
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        format_info = payload.get("format", {})
        video = next(stream for stream in streams if stream.get("codec_type") == "video")
        audio = next(stream for stream in streams if stream.get("codec_type") == "audio")
        if video.get("codec_name") != "h264":
            raise RuntimeError("Rendered video codec is not h264")
        if int(video.get("width", 0)) != OUTPUT_WIDTH or int(video.get("height", 0)) != OUTPUT_HEIGHT:
            raise RuntimeError("Rendered video dimensions are invalid")
        frame_rate = video.get("avg_frame_rate", "0/1")
        numerator, denominator = frame_rate.split("/", 1)
        measured_fps = float(numerator) / max(1.0, float(denominator))
        if abs(measured_fps - float(FPS)) > 0.5:
            raise RuntimeError("Rendered video frame rate is invalid")
        if audio.get("codec_name") != "aac" or int(audio.get("sample_rate", 0)) != 48000:
            raise RuntimeError("Rendered audio stream is invalid")
        measured_duration = float(video.get("duration") or format_info.get("duration") or 0.0)
    except (KeyError, StopIteration, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Rendered output metadata is invalid for {path.name}") from exc
    if abs(measured_duration - expected_duration) > max(0.5, 2.0 / FPS):
        raise RuntimeError("Rendered output duration is outside the allowed tolerance")


# A shot whose face timeline could not be resolved into a moving crop used to be
# emitted as a fixed integer crop box. Over statically-held source content that
# renders a literally frozen output, which editorial_qa correctly rejects as a
# freeze_interval error. The presentation_slide branch already drifted; these
# values were measured to clear freezedetect n=0.003 (0.765 mean-abs-diff) on both
# a frozen synthetic frame and the real clip from CI run 36171365900.
#
# The wave is a triangle rather than a sine on purpose: a sine's velocity reaches
# zero at every extremum, which reintroduced sub-threshold runs up to 0.53s. A
# triangle holds a near-constant speed between its turning points, so the longest
# sub-threshold run drops to ~0.07s -- below freezedetect's own d=0.5 report floor.
_STATIC_SHOT_DRIFT_FREQ = 1.8
_STATIC_SHOT_DRIFT_RATIO = 0.05
# 2/pi scales a unit sine into a unit-amplitude triangle.
_STATIC_SHOT_DRIFT_WAVE = "0.6366*asin(sin(t*{freq}))"

# Picture-in-picture inset geometry for the host-over-slide layout.
# Scaled off the output frame so it tracks OUTPUT_WIDTH/HEIGHT, and placed to
# clear both the platform UI safe zone and the caption band.
PIP_INSET_WIDTH = int(OUTPUT_WIDTH * 0.30)
PIP_INSET_HEIGHT = int(PIP_INSET_WIDTH * 16 / 9) // 2 * 2
PIP_INSET_X = OUTPUT_WIDTH - PIP_INSET_WIDTH - int(OUTPUT_WIDTH * 0.09)
PIP_INSET_Y = int(OUTPUT_HEIGHT * 0.17)
# Fraction of the active region the PiP canvas crop spans. Must stay below 1.0 so
# the crop keeps positional slack and the motion guarantee still applies.
PIP_SLIDE_CANVAS_RATIO = 0.94

# Animated punch-zoom. `crop` and `scale` cannot animate width/height per frame
# in this ffmpeg build, so zoompan is the only per-frame zoom tool available.
# It is strictly additive: the drift below is what guarantees motion, and the
# zoom rides on top of an already-moving frame, so a zoom plateau can never
# reintroduce the frozen-output failure that Phase 0 fixed.
PUNCH_ZOOM_AMOUNT = 0.12
PUNCH_ZOOM_RAMP_FRAMES = 36


def _punch_zoom_filter(motion: str, shot_frames: int) -> str:
    """
    zoompan chain for an animated punch-zoom, or "" when none is wanted.

    `motion` is one of drift / push_in / pull_out / hold. The ramp is expressed
    over the first PUNCH_ZOOM_RAMP_FRAMES output frames and then plateaus, which
    is acceptable precisely because the crop drift underneath never stops.
    """
    if motion not in ("push_in", "pull_out"):
        return ""
    ramp = max(2, min(PUNCH_ZOOM_RAMP_FRAMES, max(2, int(shot_frames))))
    if motion == "push_in":
        zoom_expr = f"1+{PUNCH_ZOOM_AMOUNT}*min(1,on/{ramp})"
    else:
        zoom_expr = f"1+{PUNCH_ZOOM_AMOUNT}*(1-min(1,on/{ramp}))"
    return (
        f"zoompan=z='{zoom_expr}':"
        f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={OUTPUT_WIDTH}x{OUTPUT_HEIGHT}:fps={FPS},setsar=1:1"
    )


def _drift_axis_expr(
    base: int,
    low: int,
    high: int,
    extent: int = 0,
    ratio: float = _STATIC_SHOT_DRIFT_RATIO,
) -> Optional[str]:
    """
    Returns a per-frame crop offset keeping continuous motion inside [low, high].

    Three properties are load-bearing:

    1. The clamp must never bind. A bound clamp flattens part of the wave into a
       static hold, which is exactly the freeze this guards against. So the
       oscillation centre is placed far enough from both bounds that the full
       amplitude fits between them.
    2. The sweep is capped at `ratio * extent` of the crop dimension, so motion
       reads as life rather than as a camera mistake.
    3. A subject sitting against a frame edge still gets motion: the centre is
       nudged inward by at most the cap, which is invisible next to the crop
       width, rather than collapsing to zero amplitude.

    Returns None only when the axis has no slack at all, letting the caller keep
    the exact static integer it emitted before.
    """
    if high <= low or base < low or base > high or extent <= 0 or ratio <= 0:
        return None
    max_amp = int(extent * ratio)
    if max_amp < 1:
        return None
    if high - low >= 2 * max_amp:
        centre = min(max(base, low + max_amp), high - max_amp)
    else:
        centre = (low + high) // 2
    amplitude = min(max_amp, centre - low, high - centre)
    if amplitude < 1:
        return None
    wave = _STATIC_SHOT_DRIFT_WAVE.format(freq=_STATIC_SHOT_DRIFT_FREQ)
    return f"max({low},min({centre}+{amplitude}*{wave},{high}))"


def _effective_drift_ratio(motion_gain: float) -> float:
    """Scale the drift ratio for a repair attempt, clamped to a sane sweep."""
    try:
        gain = float(motion_gain)
    except (TypeError, ValueError):
        return _STATIC_SHOT_DRIFT_RATIO
    if not math.isfinite(gain) or gain <= 0:
        return _STATIC_SHOT_DRIFT_RATIO
    return min(0.20, _STATIC_SHOT_DRIFT_RATIO * gain)


def _crop_position(position: int, expr: Optional[str]) -> str:
    """Render a crop x/y argument, quoting only genuine ffmpeg expressions."""
    return f"'{expr}'" if expr else str(position)


def _clamp_crop_box(
    box: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int
) -> Tuple[int, int, int, int]:
    x, y, width, height = (int(value) for value in box)
    x = max(0, min(x, max(0, frame_width - 2)))
    y = max(0, min(y, max(0, frame_height - 2)))
    width = max(2, min(width, frame_width - x))
    height = max(2, min(height, frame_height - y))
    return x, y, width - width % 2, height - height % 2


def build_video_filtergraph(
    framing: FramingDecision,
    peak_intensity_segments: List[Tuple[float, float]] = [],
    ass_subtitle_path: Optional[Path] = None,
    burn_subtitles: bool = True,
    broll_inputs: List[Tuple[int, float, float, float]] = [],
    cover_input_idx: Optional[int] = None,
    cover_duration: float = 0.25,
    motion_gain: float = 1.0
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

    # A repair attempt can ask for a stronger sweep when QA reports a freeze.
    drift_ratio = _effective_drift_ratio(motion_gain)
    video_width = max(2, int(getattr(framing, "video_width", 1920)))
    video_height = max(2, int(getattr(framing, "video_height", 1080)))
    active_x = max(0, min(int(getattr(framing, "active_x", 0)), video_width - 2))
    active_y = max(0, min(int(getattr(framing, "active_y", 0)), video_height - 2))
    active_w = max(2, min(int(getattr(framing, "active_w", video_width)), video_width - active_x))
    active_h = max(2, min(int(getattr(framing, "active_h", video_height)), video_height - active_y))
    active_w -= active_w % 2
    active_h -= active_h % 2

    if framing.mode == "multi_shot_dynamic" and framing.shots:
        half_h = OUTPUT_HEIGHT // 2
        shot_filters = []
        shot_labels = []

        for i, shot in enumerate(framing.shots):
            label = f"v_shot_{i}"
            shot_labels.append(f"[{label}]")
            shot_frames = max(2, int(round((shot.end - shot.start) * FPS)))
            punch = _punch_zoom_filter(str(getattr(shot, "motion", "drift")), shot_frames)

            if shot.mode == "pip_slide" and shot.speaker1_box:
                s_cx, s_cy, s_cw, s_ch = _clamp_crop_box(
                    shot.speaker1_box, video_width, video_height
                )
                # Picture-in-picture: the slide owns the canvas (over a blurred
                # fill so any aspect ratio survives) while the host stays visible
                # in a corner inset. This is the most common podcast visual and
                # the legacy layout enum had no way to express it.
                #
                # The canvas is cropped to PIP_SLIDE_CANVAS_RATIO of the active
                # region rather than all of it: a crop spanning the entire region
                # has zero slack on every axis, so no drift can be emitted and the
                # slide renders frozen. The small inset-zoom is imperceptible on a
                # slide and buys the slack the motion guarantee needs.
                cw = max(2, int(active_w * PIP_SLIDE_CANVAS_RATIO))
                ch = max(2, int(active_h * PIP_SLIDE_CANVAS_RATIO))
                cw -= cw % 2
                ch -= ch % 2
                c_hi_x = active_x + active_w - cw
                c_hi_y = active_y + active_h - ch
                c_x = max(active_x, min(active_x + (active_w - cw) // 2, c_hi_x))
                c_y = max(active_y, min(active_y + (active_h - ch) // 2, c_hi_y))
                canvas_x = _crop_position(c_x, _drift_axis_expr(c_x, active_x, c_hi_x, cw, drift_ratio))
                canvas_y = _crop_position(c_y, _drift_axis_expr(c_y, active_y, c_hi_y, ch, drift_ratio))
                # The host inset has plenty of slack, so it drifts too. A small
                # inset alone may not move enough pixels to clear the QA gate, so
                # this is belt-and-braces behind the canvas drift.
                h_hi_x = active_x + active_w - s_cw
                h_hi_y = active_y + active_h - s_ch
                h_x = _crop_position(s_cx, _drift_axis_expr(s_cx, active_x, h_hi_x, s_cw, drift_ratio))
                h_y = _crop_position(s_cy, _drift_axis_expr(s_cy, active_y, h_hi_y, s_ch, drift_ratio))
                shot_f = (
                    f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,split=2[p{i}_c][p{i}_h];"
                    f"[p{i}_c]crop={cw}:{ch}:{canvas_x}:{canvas_y},"
                    f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
                    f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},boxblur=30:5,"
                    f"eq=brightness=-0.16:contrast=1.12[p{i}_blur];"
                    f"[p{i}_c]crop={cw}:{ch}:{canvas_x}:{canvas_y},"
                    f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=decrease[p{i}_slide];"
                    f"[p{i}_blur][p{i}_slide]overlay=(W-w)/2:(H-h)/2,setsar=1:1,fps={FPS}[p{i}_canvas];"
                    f"[p{i}_h]crop={s_cw}:{s_ch}:{h_x}:{h_y},"
                    f"scale={PIP_INSET_WIDTH}:{PIP_INSET_HEIGHT}:"
                    f"flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}[p{i}_host];"
                    f"[p{i}_canvas][p{i}_host]overlay={PIP_INSET_X}:{PIP_INSET_Y}:format=auto,"
                    f"drawbox=x={PIP_INSET_X - 6}:y={PIP_INSET_Y - 6}:"
                    f"w={PIP_INSET_WIDTH + 12}:h={PIP_INSET_HEIGHT + 12}:"
                    f"color=white@0.85:t=6,setsar=1:1[{label}]"
                )
            elif shot.mode == "presentation_slide":
                s_cx = active_x if getattr(shot, "crop_x", None) is None else shot.crop_x
                s_cy = active_y if getattr(shot, "crop_y", None) is None else shot.crop_y
                s_cw = active_w if getattr(shot, "crop_w", None) in (None, 0) else shot.crop_w
                s_ch = active_h if getattr(shot, "crop_h", None) in (None, 0) else shot.crop_h
                s_cx, s_cy, s_cw, s_ch = _clamp_crop_box(
                    (s_cx, s_cy, s_cw, s_ch), video_width, video_height
                )
                shot_f = (
                    f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,crop={s_cw}:{s_ch}:{s_cx}:{s_cy},split=2[s{i}_fg_in][s{i}_bg_in];"
                    f"[s{i}_bg_in]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},boxblur=30:5,eq=brightness=-0.16:contrast=1.12[s{i}_bg];"
                    f"[s{i}_fg_in]crop=iw*0.92:ih*0.92:x='(iw-iw*0.92)/2+iw*0.03*sin(t*1.8)':y='(ih-ih*0.92)/2',scale={OUTPUT_WIDTH}:-1:force_original_aspect_ratio=decrease,{studio_grade}[s{i}_fg];"
                    f"[s{i}_bg][s{i}_fg]overlay=(W-w)/2:(H-h)/2,setsar=1:1,fps={FPS}[{label}]"
                )
            elif shot.mode == "split_screen" and shot.speaker1_box and shot.speaker2_box:
                s1_x, s1_y, s1_w, s1_h = _clamp_crop_box(
                    shot.speaker1_box, video_width, video_height
                )
                s2_x, s2_y, s2_w, s2_h = _clamp_crop_box(
                    shot.speaker2_box, video_width, video_height
                )
                # Crop each speaker at native resolution, then scale each pane to half-height.
                # This preserves the tight face-relative crop boxes from face_tracker.
                shot_f = (
                    f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,split=2[s{i}_p1][s{i}_p2];"
                    f"[s{i}_p1]crop={s1_w}:{s1_h}:{s1_x}:{s1_y},scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd:force_original_aspect_ratio=decrease,pad={OUTPUT_WIDTH}:{half_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1:1[s{i}_top];"
                    f"[s{i}_p2]crop={s2_w}:{s2_h}:{s2_x}:{s2_y},scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd:force_original_aspect_ratio=decrease,pad={OUTPUT_WIDTH}:{half_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1:1[s{i}_bot];"
                    f"[s{i}_top][s{i}_bot]vstack=inputs=2,scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}[{label}]"
                )
            else:
                s_cy = active_y if getattr(shot, "crop_y", None) is None else shot.crop_y
                s_ch = active_h if getattr(shot, "crop_h", None) in (None, 0) else shot.crop_h
                s_cy = max(active_y, min(int(s_cy), active_y + active_h - 2))
                s_ch = max(2, min(int(s_ch), active_y + active_h - s_cy))
                s_ch -= s_ch % 2
                target_crop_w = max(2, int(s_ch * (9 / 16)))
                target_crop_w -= target_crop_w % 2
                target_crop_w = min(target_crop_w, active_w)
                target_crop_w -= target_crop_w % 2

                zoom = getattr(shot, "zoom_factor", 1.0) or 1.0
                if zoom > 1.01:
                    eff_h = max(2, int(s_ch / zoom))
                    eff_h -= eff_h % 2
                    eff_w = max(2, int(target_crop_w / zoom))
                    eff_w -= eff_w % 2
                    eff_w = min(eff_w, active_w)
                    eff_w -= eff_w % 2
                    eff_y = max(active_y, min(int(s_cy + (s_ch - eff_h) * 0.28), active_y + active_h - eff_h))
                    shot_dur = max(0.5, shot.end - shot.start)
                    points = shot.face_centers_timeline
                    if points and len(points) >= 2 and abs(points[-1][1] - points[0][1]) > 40:
                        x_start = points[0][1]
                        x_end = points[-1][1]
                        cx_expr = f"{x_start} + ({x_end}-{x_start})*min(1.0,max(0.0,t/{shot_dur:.2f}))"
                        crop_x_expr = f"max({active_x},min({cx_expr}-{eff_w//2},{active_x + active_w}-{eff_w}))"
                        shot_f = (
                            f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,"
                            f"crop={eff_w}:{eff_h}:'{crop_x_expr}':{eff_y},"
                            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}{',' + punch if punch else ''}[{label}]"
                        )
                    else:
                        base_cx = shot.crop_x + target_crop_w // 2
                        eff_x = max(active_x, min(base_cx - eff_w // 2, active_x + active_w - eff_w))
                        shot_f = (
                            f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,"
                            f"crop={eff_w}:{eff_h}:"
                            f"{_crop_position(eff_x, _drift_axis_expr(eff_x, active_x, active_x + active_w - eff_w, eff_w, drift_ratio))}:"
                            f"{_crop_position(eff_y, _drift_axis_expr(eff_y, active_y, active_y + active_h - eff_h, eff_h, drift_ratio))},"
                            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}{',' + punch if punch else ''}[{label}]"
                        )
                else:
                    shot_dur = max(0.5, shot.end - shot.start)
                    points = shot.face_centers_timeline
                    if points and len(points) >= 2 and abs(points[-1][1] - points[0][1]) > 40:
                        x_start = points[0][1]
                        x_end = points[-1][1]
                        cx_expr = f"{x_start} + ({x_end}-{x_start})*min(1.0,max(0.0,t/{shot_dur:.2f}))"
                        crop_x_expr = f"max({active_x},min({cx_expr}-{target_crop_w//2},{active_x + active_w}-{target_crop_w}))"
                        shot_f = (
                            f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,"
                            f"crop={target_crop_w}:{s_ch}:'{crop_x_expr}':{s_cy},"
                            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}{',' + punch if punch else ''}[{label}]"
                        )
                    else:
                        crop_x = max(active_x, min(shot.crop_x, active_x + active_w - target_crop_w))
                        shot_f = (
                            f"[0:v]trim=start={shot.start:.2f}:end={shot.end:.2f},setpts=PTS-STARTPTS,"
                            f"crop={target_crop_w}:{s_ch}:"
                            f"{_crop_position(crop_x, _drift_axis_expr(crop_x, active_x, active_x + active_w - target_crop_w, target_crop_w, drift_ratio))}:"
                            f"{_crop_position(s_cy, _drift_axis_expr(s_cy, active_y, active_y + active_h - s_ch, s_ch, drift_ratio))},"
                            f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,setsar=1:1,fps={FPS}{',' + punch if punch else ''}[{label}]"
                        )
            shot_filters.append(shot_f)

        concat_inputs = "".join(shot_labels)
        concat_f = f"{concat_inputs}concat=n={len(shot_labels)}:v=1:a=0,{studio_grade}[base]"
        v_filter = ";".join(shot_filters) + ";" + concat_f

    elif framing.mode in ("single_smooth", "dynamic_cut"):
        crop_w = max(2, int(active_h * (9 / 16)))
        crop_w -= crop_w % 2
        crop_w = min(crop_w, active_w)
        crop_w -= crop_w % 2
        static_crop_x: object
        if framing.mode == "dynamic_cut" and framing.crop_x_expr:
            static_crop_x = f"'{framing.crop_x_expr}'"
        else:
            cx = framing.smoothed_center_x or (active_x + active_w // 2)
            static_crop_x = max(active_x, min(cx - crop_w // 2, active_x + active_w - crop_w))
        v_filter = f"[0:v]crop={crop_w}:{active_h}:{static_crop_x}:{active_y},scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:flags=lanczos+accurate_rnd,{studio_grade},fps={FPS}[base]"

    elif framing.mode == "split_screen" and framing.speaker1_box and framing.speaker2_box:
        s1_x, s1_y, s1_w, s1_h = _clamp_crop_box(
            framing.speaker1_box, video_width, video_height
        )
        s2_x, s2_y, s2_w, s2_h = _clamp_crop_box(
            framing.speaker2_box, video_width, video_height
        )
        half_h = OUTPUT_HEIGHT // 2
        # Crop at native resolution — face boxes are already in source pixel coordinates.
        # Each pane: tight face crop → scale to full output width × half output height.
        v_filter = (
            f"[0:v]split=2[s1_in][s2_in];"
            f"[s1_in]crop={s1_w}:{s1_h}:{s1_x}:{s1_y},scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd:force_original_aspect_ratio=decrease,pad={OUTPUT_WIDTH}:{half_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1:1,{studio_grade}[top_pane];"
            f"[s2_in]crop={s2_w}:{s2_h}:{s2_x}:{s2_y},scale={OUTPUT_WIDTH}:{half_h}:flags=lanczos+accurate_rnd:force_original_aspect_ratio=decrease,pad={OUTPUT_WIDTH}:{half_h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1:1,{studio_grade}[bottom_pane];"
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

    # Overlay Pexels B-roll clips (underneath subtitles, on top of host/guest video)
    if broll_inputs:
        for idx_b, (b_in_idx, b_start, b_end, b_source_offset) in enumerate(broll_inputs):
            b_prep = f"broll_prep_{idx_b}"
            b_out = f"broll_out_{idx_b}"
            broll_duration = max(0.0, b_end - b_start)
            b_source_offset = max(0.0, b_source_offset)
            broll_scale_f = (
                f"[{b_in_idx}:v]trim=start={b_source_offset:.2f}:end={b_source_offset + broll_duration:.2f},"
                f"setpts=PTS-STARTPTS,"
                f"scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,"
                f"crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},setsar=1:1,fps={FPS}[{b_prep}]"
            )
            broll_overlay_f = (
                f"{base_label}[{b_prep}]overlay=enable='between(t,{b_start:.2f},{b_end:.2f})':"
                f"eof_action=pass:repeatlast=0[{b_out}]"
            )
            filters.append(f"{broll_scale_f};{broll_overlay_f}")
            base_label = f"[{b_out}]"

    if burn_subtitles and ass_subtitle_path is not None and not ass_subtitle_path.exists():
        raise FileNotFoundError(f"Requested subtitle file does not exist: {ass_subtitle_path}")

    if burn_subtitles and ass_subtitle_path and ass_subtitle_path.exists():
        escaped_ass = sanitize_ffmpeg_path(ass_subtitle_path)
        from src.config import FONTS_DIR
        if FONTS_DIR.exists():
            escaped_fonts = sanitize_ffmpeg_path(FONTS_DIR)
            sub_filter = f"{base_label}subtitles='{escaped_ass}':fontsdir='{escaped_fonts}'"
        else:
            sub_filter = f"{base_label}subtitles='{escaped_ass}'"
        
        if cover_input_idx is not None:
            sub_filter += "[subs_out]"
            filters.append(sub_filter)
            base_label = "[subs_out]"
        else:
            sub_filter += "[outv]"
            filters.append(sub_filter)
            base_label = "[outv]"
    else:
        if cover_input_idx is None:
            filters.append(f"{base_label}null[outv]")

    # Overlay high-CTR cover frame for first 0.25s as auto video thumbnail for TikTok/Reels/Shorts
    if cover_input_idx is not None:
        cover_prep = "cover_prep"
        cover_scale_f = f"[{cover_input_idx}:v]scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT}:force_original_aspect_ratio=increase,crop={OUTPUT_WIDTH}:{OUTPUT_HEIGHT},setsar=1:1,fps={FPS}[{cover_prep}]"
        cover_overlay_f = f"{base_label}[{cover_prep}]overlay=enable='between(t,0,{cover_duration:.2f})'[outv]"
        filters.append(f"{cover_scale_f};{cover_overlay_f}")

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
    bgm_path: Optional[Path] = None,
    sfx_cues: List[Tuple[float, str]] = [],
    broll_cues: List[Tuple[float, float, Path, str]] = [],
    cover_image_path: Optional[Path] = None,
    cover_duration: float = 0.25,
    motion_gain: float = 1.0
) -> Path:
    requested_duration = end_time - start_time
    if requested_duration <= 0:
        raise ValueError("Clip duration must be greater than zero")

    source_duration = _probe_source_duration(source_video_path)
    if source_duration is not None:
        available_duration = max(0.0, source_duration - start_time)
        if available_duration <= 0:
            raise RuntimeError(f"Requested clip starts after source EOF for {output_clip_path.name}")
        if available_duration + 0.5 < requested_duration:
            raise RuntimeError(
                f"Source does not cover requested clip for {output_clip_path.name}: "
                f"available={available_duration:.2f}s requested={requested_duration:.2f}s"
            )
        duration = min(requested_duration, available_duration)
    else:
        duration = requested_duration
    duration_text = f"{duration:.3f}"

    output_clip_path.parent.mkdir(parents=True, exist_ok=True)

    current_input_idx = 0
    input_args = ["-ss", f"{start_time:.2f}", "-t", f"{duration:.2f}", "-i", str(source_video_path)]

    # Cover image overlay input (embed 0.25s flash thumbnail at start of video)
    cover_input_idx = None
    if cover_image_path and cover_image_path.exists():
        current_input_idx += 1
        cover_input_idx = current_input_idx
        input_args.extend(["-loop", "1", "-t", f"{cover_duration:.2f}", "-i", str(cover_image_path)])

    # Process B-roll stock footage overlays
    broll_inputs: List[Tuple[int, float, float, float]] = []
    for b_start, b_end, b_path, _kw in broll_cues:
        bounded_start = max(0.0, min(float(b_start), duration))
        bounded_end = max(0.0, min(float(b_end), duration))
        if b_path and b_path.exists() and bounded_end > bounded_start:
            current_input_idx += 1
            input_args.extend(["-stream_loop", "-1", "-i", str(b_path)])
            broll_inputs.append((current_input_idx, bounded_start, bounded_end, bounded_start))

    video_filters = build_video_filtergraph(
        framing=framing,
        peak_intensity_segments=peak_intensity_segments,
        ass_subtitle_path=ass_subtitle_path,
        burn_subtitles=burn_subtitles,
        broll_inputs=broll_inputs,
        cover_input_idx=cover_input_idx,
        cover_duration=cover_duration,
        motion_gain=motion_gain,
    )
    try:
        probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(source_video_path)]
        probe_res = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=15)
    except Exception as exc:
        raise RuntimeError(f"Audio probe failed for clip {output_clip_path.name}") from exc
    if probe_res.returncode != 0:
        raise RuntimeError(f"Audio probe failed for clip {output_clip_path.name}: {probe_res.stderr[-500:]}")

    source_has_audio = "audio" in probe_res.stdout
    voice_input_idx: Optional[int] = 0
    if not source_has_audio:
        current_input_idx += 1
        voice_input_idx = current_input_idx
        input_args.extend([
            "-f", "lavfi",
            "-t", f"{duration:.2f}",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        ])

    active_bgm = bgm_path or (AUDIO_ASSETS_DIR / "ambient_lofi_loop.mp3")
    use_bgm = ENABLE_BGM and active_bgm and active_bgm.exists()

    bgm_input_idx = None
    if use_bgm:
        current_input_idx += 1
        bgm_input_idx = current_input_idx
        input_args.extend(["-stream_loop", "-1", "-i", str(active_bgm)])

    # Collect valid SFX cues
    valid_sfx: List[Tuple[float, Path, str]] = []
    if ENABLE_SFX and sfx_cues:
        for cue_t, s_type in sfx_cues:
            if 0.0 <= cue_t < duration:
                sfx_file = SFX_ASSETS_DIR / f"{s_type}.wav"
                if sfx_file.exists() and len(valid_sfx) < 5:
                    valid_sfx.append((cue_t, sfx_file, s_type))

    sfx_indices = []
    for cue_t, sfx_file, s_type in valid_sfx:
        current_input_idx += 1
        input_args.extend(["-i", str(sfx_file)])
        sfx_indices.append((current_input_idx, cue_t, s_type))

    if voice_input_idx is not None:
        audio_subfilters = []
        mix_inputs = ["[voice]"]

        # High-definition vocal chain: 80Hz rumble cut + 3kHz presence + 10kHz air
        voice_filter = (
            f"[{voice_input_idx}:a]aformat=channel_layouts=stereo,"
            f"aresample=48000:async=1:first_pts=0,asetpts=PTS-STARTPTS,"
            f"atrim=duration={duration_text},apad=whole_dur={duration_text},"
            f"highpass=f={HIGHPASS_FREQ},"
            f"equalizer=f={VOCAL_PRESENCE_FREQ}:width_type=h:width=1000:g=2.5,"
            f"equalizer=f={VOCAL_AIR_FREQ}:width_type=h:width=2500:g=1.8"
        )

        if use_bgm and ENABLE_DYNAMIC_DUCKING:
            # Dynamic sidechain ducking: voice triggers downward compression on BGM
            bgm_base = (
                f"[{bgm_input_idx}:a]aformat=channel_layouts=stereo,"
                f"aresample=48000:async=1:first_pts=0,asetpts=PTS-STARTPTS,"
                f"atrim=duration={duration_text},apad=whole_dur={duration_text},"
                f"asetpts=PTS-STARTPTS"
            )
            audio_subfilters.append(f"{voice_filter},asplit=2[voice][voice_sc]")
            audio_subfilters.append(
                f"{bgm_base},volume=-15dB[bgm_pre];"
                f"[bgm_pre][voice_sc]sidechaincompress=threshold=0.07:ratio=6:attack=30:release=350[bgm_duck]"
            )
            mix_inputs.append("[bgm_duck]")
        elif use_bgm:
            bgm_base = (
                f"[{bgm_input_idx}:a]aformat=channel_layouts=stereo,"
                f"aresample=48000:async=1:first_pts=0,asetpts=PTS-STARTPTS,"
                f"atrim=duration={duration_text},apad=whole_dur={duration_text},"
                f"asetpts=PTS-STARTPTS"
            )
            audio_subfilters.append(f"{voice_filter}[voice]")
            audio_subfilters.append(f"{bgm_base},volume=-22dB[bgm_duck]")
            mix_inputs.append("[bgm_duck]")
        else:
            audio_subfilters.append(f"{voice_filter}[voice]")

        # Process SFX cues with millisecond precision positioning via adelay
        for i, (idx, cue_t, s_type) in enumerate(sfx_indices):
            delay_ms = max(0, int(cue_t * 1000))
            vol = "-4dB" if s_type == "whoosh" else "-3dB"
            audio_subfilters.append(
                f"[{idx}:a]aformat=channel_layouts=stereo,"
                f"aresample=48000:async=1:first_pts=0,asetpts=PTS-STARTPTS,"
                f"adelay={delay_ms}|{delay_ms},apad=whole_dur={duration_text},"
                f"atrim=duration={duration_text},asetpts=PTS-STARTPTS,volume={vol}[sfx_{i}]"
            )
            mix_inputs.append(f"[sfx_{i}]")

        mastering_tail = (
            f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TRUE_PEAK}:LRA=11,"
            f"atrim=duration={duration_text},asetpts=PTS-STARTPTS[outa]"
        )
        if len(mix_inputs) > 1:
            mix_str = "".join(mix_inputs)
            audio_subfilters.append(
                f"{mix_str}amix=inputs={len(mix_inputs)}:duration=first:dropout_transition=2,"
                f"{mastering_tail}"
            )
        else:
            audio_subfilters.append(f"[voice]{mastering_tail}")

        combined_filter = f"{video_filters};" + ";".join(audio_subfilters)
        map_args = ["-map", "[outv]", "-map", "[outa]"]
    else:
        combined_filter = video_filters
        map_args = ["-map", "[outv]"]

    temp_output = output_clip_path.with_name(
        f".{output_clip_path.stem}.{uuid.uuid4().hex}.part{output_clip_path.suffix or '.mp4'}"
    )
    normalized_output = output_clip_path.with_name(
        f".{output_clip_path.stem}.{uuid.uuid4().hex}.normalized{output_clip_path.suffix or '.mp4'}"
    )
    output_clip_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        *input_args,
        "-filter_complex", combined_filter,
        *map_args,
        "-t", f"{duration:.2f}",
        "-c:v", "libx264",
        "-crf", str(VIDEO_CRF),
        "-preset", "faster" if IS_CI else "medium",
        "-threads", "2" if IS_CI else "4",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-c:a", "aac",
        "-b:a", AUDIO_BITRATE,
        "-ar", "48000",
        "-movflags", "+faststart",
        str(temp_output)
    ]

    print(f"[*] Rendering viral clip with FFmpeg: {output_clip_path.name} (Framing: {framing.mode})...")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=900,
        )

        if result.returncode != 0:
            print(f"[-] FFmpeg error:\n{result.stderr[-1500:]}")
            print(f"[-] FFmpeg complex filter was:\n{combined_filter}")
            raise RuntimeError(f"FFmpeg failed to render clip {output_clip_path.name}")
        if not temp_output.exists() or temp_output.stat().st_size <= 0:
            raise RuntimeError(f"FFmpeg produced no output for clip {output_clip_path.name}")
        _validate_rendered_output(temp_output, duration)

        loudness_verified = False
        if source_has_audio or use_bgm or sfx_indices:
            measurement = _measure_loudness(temp_output)
            if measurement is None:
                raise RuntimeError(f"Could not measure loudness for clip {output_clip_path.name}")
            source_candidate = temp_output
            attempt_outputs: List[Path] = []
            for attempt, (linear_pass, limiter_limit) in enumerate(_LOUDNESS_PASS_LADDER):
                if attempt == 0:
                    destination = normalized_output
                else:
                    destination = output_clip_path.with_name(
                        f".{output_clip_path.stem}.{uuid.uuid4().hex}"
                        f".retry{attempt}{output_clip_path.suffix or '.mp4'}"
                    )
                if attempt > 0:
                    print(
                        f"[!] Retrying loudness pass for {output_clip_path.name} "
                        f"(attempt {attempt + 1}, linear={linear_pass}, "
                        f"limiter={limiter_limit:.2f})."
                    )
                _apply_measured_loudness(
                    source_candidate, destination, measurement, limiter_limit, linear_pass
                )
                if not destination.exists() or destination.stat().st_size <= 0:
                    raise RuntimeError(f"Loudness pass produced no output for clip {output_clip_path.name}")
                verification = _measure_loudness(destination)
                if verification is None:
                    raise RuntimeError(f"Could not verify loudness for clip {output_clip_path.name}")
                lufs_ok = abs(verification["input_i"] - TARGET_LUFS) <= 1.0
                peak_ok = verification["input_tp"] <= TARGET_TRUE_PEAK + 0.5
                if lufs_ok and peak_ok:
                    loudness_verified = True
                    attempt_outputs.append(destination)
                    break
                print(
                    f"[!] Loudness verification missed target for {output_clip_path.name} "
                    f"(LUFS={verification['input_i']:.2f}, TP={verification['input_tp']:.2f})."
                )
                attempt_outputs.append(destination)
                source_candidate = destination
                measurement = verification
            if not loudness_verified:
                raise RuntimeError(f"Loudness verification failed for clip {output_clip_path.name}")
            os.replace(attempt_outputs[-1], output_clip_path)
            for stale_output in attempt_outputs[:-1]:
                if stale_output.exists():
                    stale_output.unlink()
            if temp_output.exists():
                temp_output.unlink()
        else:
            os.replace(temp_output, output_clip_path)

        metadata = {
            "framing_mode": framing.mode,
            "source_duration": source_duration,
            "requested_duration": requested_duration,
            "output_duration": duration,
            "audio_bed": not source_has_audio,
            "broll_count": len(broll_inputs),
            "sfx_count": len(sfx_indices),
            "loudness_verified": loudness_verified,
        }
        print(f"[render_metadata] {json.dumps(metadata, sort_keys=True)}")
    except Exception:
        for partial_output in list(output_clip_path.parent.glob(f".{output_clip_path.stem}.*")):
            if partial_output.exists():
                try:
                    partial_output.unlink()
                except OSError:
                    pass
        raise

    size_mb = (output_clip_path.stat().st_size / (1024 * 1024)) if output_clip_path.exists() else 0.0
    print(f"[+] Clip rendered successfully: {output_clip_path} (Size: {size_mb:.2f} MB)")
    return output_clip_path
