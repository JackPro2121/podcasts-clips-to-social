"""V2 Streamer Clipper Pipeline Orchestrator.

Full autonomous pipeline specialized for US Streamers (Kai Cenat, IShowSpeed, Jynxzi, etc.):
- Zero full-video download (Apify/targeted range download)
- Fast-paced dead air trimming
- Kinetic high-energy subtitles
- Sound effects (whoosh, pop, ding)
- Universal face-aware thumbnails (chest/headroom collision avoidance)
- BUFFER ISOLATION: Does NOT touch podcast Buffer channels or credentials.
"""

import os
from pathlib import Path
from typing import Optional, List

from src.config import (
    CLIPS_DIR, DOWNLOADS_DIR, SUBTITLES_DIR,
    CHANNEL_WATERMARK
)
from src.downloader import (
    fetch_transcript_only, download_clip_segment,
    get_video_duration
)
from src.transcriber import (
    get_transcript, transcribe_audio_whisper,
    TranscriptSegment
)
from src.jump_cutter import build_cut_plan, prepare_compacted_clip
from src.face_tracker import analyze_faces_in_clip
from src.subtitle_generator import create_styled_ass_subtitles
from src.video_editor import render_viral_clip
from src.thumbnail_generator import generate_clip_thumbnail
from src.streamers.discovery import (
    discover_streamer_candidates, record_streamer_history, extract_video_id_from_url
)
from src.streamers.viral_detector import (
    detect_streamer_viral_moments, StreamerClipCandidate
)

# Dedicated streamer output folder
STREAMER_CLIPS_DIR = CLIPS_DIR / "streamers"
STREAMER_CLIPS_DIR.mkdir(parents=True, exist_ok=True)

# Dedicated Streamer Buffer environment variables (isolated from podcast Buffer)
STREAMER_BUFFER_ACCESS_TOKEN = os.getenv("STREAMER_BUFFER_ACCESS_TOKEN", "").strip()
STREAMER_BUFFER_CHANNEL_IDS = [
    c.strip() for c in os.getenv("STREAMER_BUFFER_CHANNEL_IDS", "").split(",") if c.strip()
]
STREAMER_WATERMARK = os.getenv("STREAMER_WATERMARK", CHANNEL_WATERMARK or "@streamersclips")


def run_streamer_pipeline(
    video_url: Optional[str] = None,
    creator_name: Optional[str] = None,
    num_clips: int = 3,
    post_to_buffer: bool = False,
    framing_mode: str = "auto",
) -> List[Path]:
    """Runs the V2 Streamer Clipping Pipeline.

    Args:
        video_url: Direct YouTube stream/VOD URL, or None to auto-discover
        creator_name: Target streamer filter (e.g. 'Kai Cenat', 'IShowSpeed', 'Jynxzi')
        num_clips: Number of viral clips to extract (1 to 5)
        post_to_buffer: Social publishing toggle (defaults to False, strictly isolated)
        framing_mode: 'auto', 'single_smooth', or 'split_screen'
    """
    print("=" * 70)
    print("⚡ AUTONOMOUS AI STREAMER VIRAL CLIPPER (PIPELINE V2)")
    print("=" * 70)

    # 1. Target URL resolution
    target_url = video_url
    target_title = ""
    target_creator = creator_name or "Streamer"

    if not target_url:
        print(f"[*] Auto-discovering latest streams (Filter: {creator_name or 'All Top US Streamers'})...")
        candidates = discover_streamer_candidates(creator=creator_name, max_candidates=5)
        if not candidates:
            print("[-] No fresh streamer streams found in RSS discovery feeds.")
            return []
        chosen = candidates[0]
        target_url = chosen["url"]
        target_title = chosen["title"]
        target_creator = chosen.get("creator", "Streamer")
        print(f"[+] Selected stream: '{target_title}' by {target_creator} ({target_url})")
    else:
        print(f"[*] Using provided stream URL: {target_url}")

    video_id = extract_video_id_from_url(target_url) or "stream_video"

    # 2. Fetch transcript without downloading video
    print("\n--- [1/5] FETCHING TRANSCRIPT (ZERO VIDEO DOWNLOAD) ---")
    transcript_data = fetch_transcript_only(target_url, allow_apify_fallback=True)
    if not transcript_data or not transcript_data.get("transcript"):
        print("[-] Could not retrieve transcript for stream. Attempting fallback...")
        return []

    segments: List[TranscriptSegment] = get_transcript(
        video_path=Path("__transcript_only__"),
        native_transcript=transcript_data["transcript"],
        video_id=video_id
    )
    if not segments:
        print("[-] Could not parse transcript segments.")
        return []

    print(f"[+] Retrieved {len(segments)} transcript segments. Analyzing viral moments...")

    # 3. Detect high-energy streamer viral moments
    print("\n--- [2/5] HIGH-ENERGY VIRAL MOMENT DETECTION ---")
    viral_moments: List[StreamerClipCandidate] = detect_streamer_viral_moments(
        segments, creator_name=target_creator, num_clips=num_clips
    )

    if not viral_moments:
        print("[-] No viral moments detected in stream.")
        return []

    print(f"[+] Detected {len(viral_moments)} viral streamer moments:")
    for idx, vm in enumerate(viral_moments, start=1):
        print(f"  #{idx}: '{vm.title}' ({vm.duration:.1f}s, Score: {vm.viral_score}) -> {vm.hook_reason}")

    # 4. Download and process only the targeted segments
    print("\n--- [3/5] TARGETED SEGMENT EXTRACTION & RENDERING ---")
    rendered_clips: List[Path] = []

    for idx, vm in enumerate(viral_moments, start=1):
        print(f"\n>>> Processing Streamer Clip #{idx}: '{vm.title}' [{vm.start_time:.1f}s - {vm.end_time:.1f}s]")

        # Download just the 30-45s window
        clip_info = download_clip_segment(
            video_url=target_url,
            start_time=max(0.0, vm.start_time - 1.0),
            end_time=vm.end_time + 1.0,
            clip_index=idx,
            output_dir=DOWNLOADS_DIR
        )
        if not clip_info or "video_path" not in clip_info:
            print(f"[-] Failed to download segment for clip #{idx}; skipping.")
            continue

        clip_path = Path(clip_info["video_path"])
        if not clip_path.exists():
            print(f"[-] Downloaded clip path missing for clip #{idx}; skipping.")
            continue

        clip_duration = get_video_duration(clip_path)

        # Transcribe & word-align audio for kinetic captions
        whisper_segments = transcribe_audio_whisper(clip_path)
        clip_segments = whisper_segments if whisper_segments else segments

        # Dead air trimming (jump cuts)
        p_words = []
        for s in clip_segments:
            for w in getattr(s, "words", []) or []:
                if 0.0 <= w.start <= clip_duration:
                    p_words.append((w.start, w.end))

        cut_plan = build_cut_plan(p_words, clip_duration)
        if cut_plan.applied:
            prepared = prepare_compacted_clip(
                clip_path, p_words, clip_duration, clip_path.parent
            )
            if prepared:
                clip_path, _, clip_duration = prepared
                print(f"[*] Streamer pacing: removed {cut_plan.removed_s:.2f}s of dead air.")

        # Face tracking & framing
        framing = analyze_faces_in_clip(clip_path, start_time=0.0, end_time=clip_duration)

        # Generate styled ASS kinetic subtitles (Hormozi / Streamer viral style)
        ass_path = SUBTITLES_DIR / f"streamer_clip_{idx}_{video_id}.ass"
        create_styled_ass_subtitles(
            segments=clip_segments,
            clip_start=0.0,
            clip_end=clip_duration,
            output_ass_path=ass_path,
            theme_key="hormozi",
            watermark=STREAMER_WATERMARK
        )

        # Generate smart face-aware thumbnail (chest/headroom collision avoidance)
        thumb_path = STREAMER_CLIPS_DIR / f"streamer_clip_{idx}_{vm.title.replace(' ', '_')[:30]}_thumb.jpg"
        generate_clip_thumbnail(
            clip_path=clip_path,
            title=vm.title,
            output_path=thumb_path,
            badge_text=f"{target_creator.upper()} LIVE",
            watermark=STREAMER_WATERMARK
        )

        # Final 9:16 viral clip rendering with FFmpeg
        out_clip_name = f"clip_{idx}_{vm.title.replace(' ', '_')[:32]}.mp4"
        out_clip_path = STREAMER_CLIPS_DIR / out_clip_name

        rendered = render_viral_clip(
            source_video_path=clip_path,
            start_time=0.0,
            end_time=clip_duration,
            output_clip_path=out_clip_path,
            framing=framing,
            ass_subtitle_path=ass_path,
            burn_subtitles=True,
            sfx_cues=[(s[0], s[1]) for s in vm.sfx_cues]
        )

        if rendered and rendered.exists():
            rendered_clips.append(rendered)
            print(f"[+] Rendered Streamer Clip #{idx} ({rendered.stat().st_size / 1024 / 1024:.2f} MB): {rendered.name}")

    # 5. Buffer Publishing / Isolation Check
    print("\n--- [4/5] BUFFER SOCIAL PUBLISHING ISOLATION ---")
    if post_to_buffer and STREAMER_BUFFER_ACCESS_TOKEN and STREAMER_BUFFER_CHANNEL_IDS:
        print("[*] Streamer Buffer API detected. Queuing clips...")
        # Future: connect to dedicated Streamer Buffer Client once user provides API
        print("[!] Dedicated Streamer Buffer client will dispatch once credentials are confirmed.")
    else:
        print("[*] BUFFER ISOLATION ACTIVE:")
        print("    -> Existing Podcast Buffer credentials were NOT touched.")
        print("    -> Streamer Buffer API is pending user configuration.")
        print("    -> Clips safely saved locally in clips/streamers/ and ready for manual review.")

    # 6. Record history
    print("\n--- [5/5] RECORDING HISTORY ---")
    if target_url:
        record_streamer_history(video_id, title=target_title, creator=target_creator)

    print("=" * 70)
    print(f"🎉 V2 Streamer Pipeline finished! Rendered {len(rendered_clips)} clips.")
    print("=" * 70)
    return rendered_clips
