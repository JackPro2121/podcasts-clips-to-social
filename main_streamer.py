#!/usr/bin/env python3
"""V2 Standalone CLI Entrypoint: Autonomous Streamer Viral Clipper.

Usage:
    # Auto-discover fresh streams across top US streamers:
    python main_streamer.py

    # Target specific streamer:
    python main_streamer.py --streamer ishowspeed --num-clips 3
    python main_streamer.py --streamer kai_cenat --num-clips 3

    # Process specific stream / VOD:
    python main_streamer.py --video-url "https://www.youtube.com/watch?v=VIDEO_ID" --num-clips 3
"""

import sys
import argparse

# Ensure UTF-8 console output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.streamers.pipeline import run_streamer_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V2 AI Streamer Viral Clipper (US Streamers: Kai Cenat, Speed, Jynxzi, CaseOh, xQc)"
    )
    parser.add_argument(
        "--video-url",
        type=str,
        default="",
        help="YouTube Stream/VOD URL. If blank, auto-discovers latest streams."
    )
    parser.add_argument(
        "--streamer",
        type=str,
        default="",
        help="Target streamer filter: kai_cenat, ishowspeed, jynxzi, caseoh, xqc, adin_ross"
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=3,
        help="Number of viral clips to extract (default: 3)"
    )
    parser.add_argument(
        "--framing",
        type=str,
        default="auto",
        choices=["auto", "single_smooth", "split_screen"],
        help="Framing mode: auto, single_smooth, split_screen"
    )
    parser.add_argument(
        "--post-to-buffer",
        action="store_true",
        default=False,
        help="Post to Streamer Buffer account (strictly isolated, requires STREAMER_BUFFER_ACCESS_TOKEN)"
    )

    args = parser.parse_args()

    clips = run_streamer_pipeline(
        video_url=args.video_url.strip() or None,
        creator_name=args.streamer.strip() or None,
        num_clips=args.num_clips,
        post_to_buffer=args.post_to_buffer,
        framing_mode=args.framing
    )

    if clips:
        print(f"\n[+] Successfully generated {len(clips)} streamer clips:")
        for c in clips:
            print(f"    - {c}")
        sys.exit(0)
    else:
        print("\n[-] No clips generated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
