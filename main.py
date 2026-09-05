import sys
import argparse
from pathlib import Path
from typing import Optional

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import CLIPS_DIR, SUBTITLES_DIR
from src.downloader import download_video
from src.transcriber import get_transcript
from src.viral_detector import detect_viral_moments
from src.face_tracker import analyze_faces_in_clip, FramingDecision
from src.subtitle_generator import create_styled_ass_subtitles
from src.video_editor import render_viral_clip
from src.github_uploader import upload_clip_to_github_release
from src.buffer_client import BufferClient

def run_pipeline(
    url_or_path: str,
    num_clips: int = 3,
    framing_mode: str = "auto",
    subtitle_style: str = "hormozi",
    subtitles_mode: str = "auto",
    post_to_buffer: bool = False,
    dry_run: bool = False
):
    print("=" * 70)
    print("🚀 AUTONOMOUS AI PODCAST CLIPPER & BUFFER SOCIAL PUBLISHER ($0)")
    print("=" * 70)
    print(f"[*] Target Source: {url_or_path}")
    print(f"[*] Clips to generate: {num_clips}")
    print(f"[*] Framing Strategy: {framing_mode}")
    print(f"[*] Subtitle Aesthetic: {subtitle_style} (Mode: {subtitles_mode})")
    print(f"[*] Post to Buffer: {post_to_buffer}")
    print(f"[*] Dry Run: {dry_run}")
    print("=" * 70)

    # Step 1: Download video & extract native captions if available
    print("\n--- [1/6] INGESTION & DOWNLOAD ---")
    download_info = download_video(url_or_path)
    video_path = download_info['video_path']
    video_title = download_info['title']
    native_transcript = download_info.get('transcript')
    print(f"[+] Active video file: {video_path}")

    # Step 2: Extract or transcribe word-level transcript
    print("\n--- [2/6] SPEECH-TO-TEXT / TRANSCRIPT EXTRACTION ---")
    segments = get_transcript(video_path, native_transcript)
    if not segments:
        print("[-] Error: Could not obtain transcript for video. Exiting.")
        sys.exit(1)

    # Step 3: AI Viral Moment Detection (Google Gemini Flash)
    print("\n--- [3/6] VIRAL MOMENT HUNTING & HOOK SCORING ---")
    viral_moments = detect_viral_moments(segments, num_clips=num_clips)
    if not viral_moments:
        print("[-] No viral moments detected. Exiting.")
        sys.exit(1)

    print(f"\n[+] Top {len(viral_moments)} Viral Moments Identified:")
    for idx, m in enumerate(viral_moments, 1):
        print(f"   #{idx}: [{m.start_time:.1f}s - {m.end_time:.1f}s] (Virality: {m.viral_score}/100) '{m.title}'")

    if dry_run:
        print("\n[!] Dry run enabled: Skipping video rendering and Buffer upload.")
        return

    # Step 4 & 5: Video Enhancement, Framing & Subtitle Generation
    print("\n--- [4/6 & 5/6] EDITING, FACE TRACKING, AUDIO MASTERING & RENDERING ---")
    rendered_clips = []

    for idx, moment in enumerate(viral_moments, 1):
        print(f"\n>>> Processing Clip #{idx}: {moment.title} ({moment.duration:.1f}s)")
        
        # Determine Framing:
        if framing_mode == "auto":
            print("[*] Running AI Face Detection & Speaker Tracking...")
            framing = analyze_faces_in_clip(video_path, moment.start_time, moment.end_time)
            print(f"[+] Framing decision: '{framing.mode}' (Detected faces: {framing.face_count})")
        elif framing_mode == "split":
            # Force split screen
            framing = FramingDecision(mode="split_screen", face_count=2, speaker1_box=(100,0,900,1080), speaker2_box=(920,0,900,1080))
        elif framing_mode == "crop":
            framing = FramingDecision(mode="single_smooth", face_count=1, smoothed_center_x=960)
        else:
            framing = FramingDecision(mode="blur_stack", face_count=0)

        # Subtitle Generation:
        burn_subtitles = subtitles_mode in ("auto", "burn")
        ass_path = None
        if burn_subtitles:
            ass_path = SUBTITLES_DIR / f"clip_{idx}_{subtitle_style}.ass"
            create_styled_ass_subtitles(
                segments=segments,
                clip_start=moment.start_time,
                clip_end=moment.end_time,
                output_ass_path=ass_path,
                theme_key=subtitle_style,
                layout_mode=framing.mode
            )

        # Video Render
        safe_title = "".join(c for c in moment.title if c.isalnum() or c in (" ", "_", "-")).rstrip()
        safe_title = safe_title.replace(" ", "_")[:30]
        out_clip_path = CLIPS_DIR / f"clip_{idx}_{safe_title}.mp4"

        rendered_path = render_viral_clip(
            source_video_path=video_path,
            start_time=moment.start_time,
            end_time=moment.end_time,
            output_clip_path=out_clip_path,
            framing=framing,
            ass_subtitle_path=ass_path,
            burn_subtitles=burn_subtitles
        )

        rendered_clips.append({
            "path": rendered_path,
            "moment": moment
        })

    # Step 6: Permanent Hosting & Buffer Social Distribution
    print("\n--- [6/6] PERMANENT HOSTING & BUFFER SOCIAL PUBLISHING ---")
    buffer_client = BufferClient() if post_to_buffer else None

    for item in rendered_clips:
        clip_path = item["path"]
        moment = item["moment"]

        # Upload to GitHub Release for $0 permanent high-speed direct video URL
        direct_url = upload_clip_to_github_release(clip_path)

        # If Buffer posting requested and public URL exists:
        if post_to_buffer and direct_url:
            caption_text = f"{moment.title}\n\n{moment.social_caption}\n\n{' '.join(moment.hashtags)}"
            buffer_client.schedule_video_post(
                video_url=direct_url,
                text=caption_text
            )
        elif post_to_buffer and not direct_url:
            print("[-] Cannot post to Buffer because direct video URL is not available.")

    # Step 7: Storage Hygiene: Auto-delete releases older than 5 days
    print("\n--- [7/7] STORAGE HYGIENE: AUTO-CLEANUP RELEASES > 5 DAYS ---")
    try:
        from src.release_cleaner import clean_old_releases
        clean_old_releases(days=5)
    except Exception as e:
        print(f"[-] Auto-cleanup warning: {e}")

    print("\n" + "=" * 70)
    print("✨ ALL CLIPS PROCESSED SUCCESSFULLY!")
    print(f"📂 Output clips saved to: {CLIPS_DIR.resolve()}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous AI Podcast Viral Clipper & Buffer Social Media Publisher ($0 Budget)"
    )
    parser.add_argument(
        "--url", "-u",
        required=True,
        help="YouTube video URL, podcast link, or path to local video file."
    )
    parser.add_argument(
        "--num-clips", "-n",
        type=int,
        default=3,
        help="Number of viral clips to extract (default: 3)."
    )
    parser.add_argument(
        "--framing", "-f",
        choices=["auto", "split", "blur_stack", "crop"],
        default="auto",
        help="Video framing strategy (default: auto with face detection)."
    )
    parser.add_argument(
        "--subtitle-style", "-s",
        choices=["hormozi", "neon_green", "luxury_gold", "cyber_cyan"],
        default="hormozi",
        help="Styling and color aesthetic for burned-in captions (default: hormozi)."
    )
    parser.add_argument(
        "--subtitles",
        choices=["auto", "burn", "skip"],
        default="auto",
        help="Subtitle burn-in mode: auto, burn, or skip if already baked-in."
    )
    parser.add_argument(
        "--post-to-buffer",
        action="store_true",
        help="Automatically dispatch generated clips to connected Buffer social accounts."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform transcript analysis and moment hunting without heavy video rendering."
    )

    args = parser.parse_args()

    run_pipeline(
        url_or_path=args.url,
        num_clips=args.num_clips,
        framing_mode=args.framing,
        subtitle_style=args.subtitle_style,
        subtitles_mode=args.subtitles,
        post_to_buffer=args.post_to_buffer,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
