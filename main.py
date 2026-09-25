import sys
import argparse
import traceback
import uuid
from pathlib import Path
from typing import Optional, List, Union, Dict, Any, Tuple

# Ensure UTF-8 output encoding across Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from src.config import (
    CLIPS_DIR, SUBTITLES_DIR, DOWNLOADS_DIR, CLIP_ONLY_MODE, DATA_DIR, FPS, OUTPUT_WIDTH, OUTPUT_HEIGHT, IS_CI,
    MAX_TRANSCRIPT_FALLBACKS, BUFFER_ACCESS_TOKEN,
    UNIVERSAL_EDITOR_SHADOW, UNIVERSAL_EDITOR_ENFORCE_QA,
)
from src.downloader import (
    download_video, fetch_transcript_only, download_clip_segment, extract_youtube_id,
    get_video_dimensions, get_video_duration, APIFY_API_TOKEN,
)
from src.transcriber import (
    align_clip_transcript,
    get_transcript,
    TranscriptSegment,
    WordTimestamp,
    transcribe_audio_whisper,
    verify_audio_language,
)
from src.viral_detector import detect_viral_moments
from src.face_tracker import analyze_faces_in_clip, FramingDecision
from src.subtitle_generator import create_styled_ass_subtitles
from src.video_editor import render_viral_clip
from src.thumbnail_generator import generate_clip_thumbnail
from src.broll_manager import find_broll_cues_for_clip
from src.github_uploader import upload_clip_to_github_release
from src.buffer_client import BufferClient
from src.slack_notifier import SlackNotifier
from src.channel_discovery import record_history
from src.editor_models import StageStatus
from src.editorial_qa import run_editorial_qa, save_qa_report
from src.run_state import RunStateStore
from src.universal_editor import ClipEditorArtifacts, build_clip_editor_artifacts


def _manual_framing(video_path: Path, framing_mode: str) -> FramingDecision:
    width, height = get_video_dimensions(video_path)
    if framing_mode == "split":
        panel_height = max(1, height // 2)
        panel_width = min(width, max(1, int(panel_height * 9 / 8)))
        if panel_width * 2 > width:
            panel_width = max(1, width // 2)
        left_x = max(0, (width - (panel_width * 2)) // 2)
        return FramingDecision(
            mode="split_screen",
            face_count=2,
            speaker1_box=(left_x, 0, panel_width, panel_height),
            speaker2_box=(left_x + panel_width, 0, panel_width, panel_height),
            video_width=width,
            video_height=height,
            active_x=0,
            active_y=0,
            active_w=width,
            active_h=height,
        )
    if framing_mode == "crop":
        return FramingDecision(
            mode="single_smooth",
            face_count=1,
            smoothed_center_x=width // 2,
            video_width=width,
            video_height=height,
            active_x=0,
            active_y=0,
            active_w=width,
            active_h=height,
        )
    return FramingDecision(
        mode="blur_stack",
        face_count=0,
        video_width=width,
        video_height=height,
        active_x=0,
        active_y=0,
        active_w=width,
        active_h=height,
    )


def _print_framing_summary(framing: FramingDecision) -> None:
    print(f"[+] Framing decision: '{framing.mode}' (detected faces: {framing.face_count})")
    for shot in getattr(framing, "shots", []):
        print(
            f"    shot {shot.start:.2f}-{shot.end:.2f}s mode={shot.mode} "
            f"crop_x={shot.crop_x} face_track_points={len(shot.face_centers_timeline)}"
        )


def _extend_moment_to_complete_transcript(segments: List[TranscriptSegment], moment: Any) -> None:
    original_end = float(moment.end_time)
    max_end = original_end + 8.0
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        if segment.end <= original_end + 0.05:
            continue
        if segment.start > max_end:
            break
        if segment.text.rstrip().endswith((".", "?", "!")):
            moment.end_time = min(segment.end, max_end)
            moment.duration = moment.end_time - moment.start_time
            print(f"[+] Extended clip endpoint to complete transcript sentence at {moment.end_time:.2f}s.")
            return


def _build_universal_shadow(
    run_id: str,
    state_store: RunStateStore,
    video_path: Path,
    segments: List[TranscriptSegment],
    clip_start: float,
    clip_end: float,
    clip_index: int,
    source_window: Optional[Tuple[float, float]] = None,
) -> Optional[ClipEditorArtifacts]:
    if not UNIVERSAL_EDITOR_SHADOW or not segments:
        return None
    try:
        artifact_dir = DATA_DIR / "runs" / run_id
        artifacts = build_clip_editor_artifacts(
            video_path=video_path,
            segments=segments,
            clip_start=clip_start,
            clip_end=clip_end,
            artifact_dir=artifact_dir,
            run_id=run_id,
            clip_index=clip_index,
            sample_fps=1.0,
            detect_faces=True,
            source_window=source_window,
        )
        state_store.record_artifact(run_id, f"clip_{clip_index}_source_index", artifact_dir / f"clip_{clip_index}_source_index.json")
        state_store.record_artifact(run_id, f"clip_{clip_index}_edit_plan", artifact_dir / f"clip_{clip_index}_edit_plan.json")
        state_store.record_artifact(run_id, f"clip_{clip_index}_composition_plan", artifact_dir / f"clip_{clip_index}_composition_plan.json")
        print(f"[+] Universal editor shadow artifacts created for clip #{clip_index}.")
        return artifacts
    except Exception as shadow_error:
        print(f"[!] Universal editor shadow analysis unavailable for clip #{clip_index}: {shadow_error}")
        return None


def _run_universal_qa(
    run_id: str,
    state_store: RunStateStore,
    clip_index: int,
    rendered_path: Path,
    artifacts: Optional[ClipEditorArtifacts],
    clip_duration: float,
) -> bool:
    if artifacts is None:
        return True
    qa_report = run_editorial_qa(
        path=rendered_path,
        edit_plan=artifacts.edit_plan,
        composition=artifacts.composition_plan,
        expected_width=OUTPUT_WIDTH,
        expected_height=OUTPUT_HEIGHT,
        expected_fps=float(FPS),
        source_duration=clip_duration,
        report_id=f"{run_id}_clip_{clip_index}",
    )
    qa_path = DATA_DIR / "runs" / run_id / f"clip_{clip_index}_qa_report.json"
    save_qa_report(qa_report, qa_path)
    state_store.record_artifact(run_id, f"clip_{clip_index}_qa_report", qa_path)
    if UNIVERSAL_EDITOR_ENFORCE_QA and not qa_report.passed:
        issue_codes = [issue.code for issue in qa_report.issues]
        print(f"[-] Editorial QA blocked clip #{clip_index}: {issue_codes}")
        return False
    print(f"[+] Editorial QA report for clip #{clip_index}: passed={qa_report.passed}")
    warning_codes = [issue.code for issue in qa_report.issues if not issue.blocks_publish]
    if warning_codes:
        print(f"[!] Editorial QA warnings for clip #{clip_index}: {warning_codes}")
    return True


def run_pipeline(
    url_or_path: Union[str, List[str]],
    num_clips: int = 3,
    framing_mode: str = "auto",
    subtitle_style: str = "hormozi",
    subtitles_mode: str = "auto",
    post_to_buffer: bool = False,
    dry_run: bool = False,
    watermark: Optional[str] = None,
    niche: str = "auto"
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

    candidates = [url_or_path] if isinstance(url_or_path, str) else list(url_or_path)
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    state_store = RunStateStore(DATA_DIR / "runs")
    state_store.create(
        run_id,
        candidates[0],
        settings={
            "num_clips": num_clips,
            "niche": niche,
            "framing": framing_mode,
            "subtitle_style": subtitle_style,
            "subtitles_mode": subtitles_mode,
            "post_to_buffer": post_to_buffer,
            "watermark": watermark,
        },
    )
    state_store.set_stage(run_id, "pipeline", StageStatus.RUNNING)
    active_source_url = candidates[0]
    is_youtube_url = any(
        "youtube.com" in c or "youtu.be" in c for c in candidates
    )
    is_local_file = False
    if isinstance(url_or_path, (str, Path)):
        try:
            p = Path(url_or_path)
            is_local_file = p.exists() and p.is_file()
        except Exception:
            is_local_file = False

    # =========================================================================
    # SMART TRANSCRIPT-FIRST PIPELINE
    # Strategy: Fetch transcript text ONLY (zero video bytes) -> send to Gemini
    # -> identify N viral clip timestamps -> download ONLY those N short segments.
    # =========================================================================

    # ------ Step 1: Fetch Transcript (Text Only - ZERO Video Downloaded) ------
    print("\n--- [1/6] TRANSCRIPT FETCH (ZERO VIDEO DOWNLOAD) ---")

    segments: Optional[List[TranscriptSegment]] = None
    video_id: Optional[str] = extract_youtube_id(active_source_url)
    video_title = f"YouTube_{video_id or 'podcast'}"
    native_transcript = None

    if is_youtube_url and not is_local_file:
        for candidate_index, cand in enumerate(candidates):
            cand_id = extract_youtube_id(cand)
            print(f"[*] Checking candidate for native transcript: {cand}")
            transcript_data = fetch_transcript_only(
                cand,
                allow_apify_fallback=candidate_index < MAX_TRANSCRIPT_FALLBACKS,
            )
            if transcript_data and transcript_data.get("transcript"):
                active_source_url = cand
                video_id = transcript_data.get("video_id", cand_id)
                video_title = f"YouTube_{video_id or 'podcast'}"
                native_transcript = transcript_data["transcript"]
                print(f"[+] Native YouTube transcript fetched ({len(native_transcript)} segments) for '{cand}'. No video downloaded yet.")
                segments = get_transcript(
                    video_path=Path("__transcript_only__"),
                    native_transcript=native_transcript,
                    video_id=video_id
                )
                if segments:
                    break
        if not segments:
            print("[!] No candidate had a native transcript. Will use full-download fallback pipeline.")

    # ------ Step 2: AI Viral Moment Detection BEFORE any video download ------
    print("\n--- [2/6] VIRAL MOMENT HUNTING & HOOK SCORING ---")

    viral_moments = None
    if segments:
        viral_moments = detect_viral_moments(segments, num_clips=num_clips, niche=niche)
        if viral_moments:
            print(f"\n[+] Gemini identified {len(viral_moments)} viral clips:")
            for idx, m in enumerate(viral_moments, 1):
                print(f"   #{idx}: [{m.start_time:.1f}s -> {m.end_time:.1f}s] ({m.duration:.0f}s) | Score: {m.viral_score}/100 | '{m.title}'")
        else:
            print("[!] Gemini found no viral moments from transcript. Falling back to full-download pipeline.")
            viral_moments = None

    if dry_run:
        state_store.set_stage(run_id, "pipeline", StageStatus.COMPLETED)
        state_store.set_status(run_id, "completed")
        print("\n[!] Dry run enabled. Skipping video download, rendering and Buffer upload.")
        return

    rendered_clips: List[Dict[str, Any]] = []

    # ------ BRANCH A: Targeted Clip Segment Downloads ------
    if viral_moments and is_youtube_url and not is_local_file:
        print("\n--- [3/6] TARGETED CLIP SEGMENT DOWNLOADS (Not Full Video!) ---")
        print(f"[*] Downloading only {len(viral_moments)} clip segments (~15-25 MB each) instead of full podcast.")

        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

        for idx, moment in enumerate(viral_moments, 1):
            if segments:
                _extend_moment_to_complete_transcript(segments, moment)
            print(f"\n>>> Clip #{idx}: '{moment.title}' [{moment.start_time:.1f}s -> {moment.end_time:.1f}s]")

            clip_info = download_clip_segment(
                video_url=active_source_url,
                start_time=moment.start_time,
                end_time=moment.end_time,
                clip_index=idx,
                output_dir=DOWNLOADS_DIR,
            )

            if not clip_info:
                print(f"[-] Segment download failed for clip #{idx}. Skipping.")
                continue

            clip_path = clip_info['video_path']
            clip_duration = moment.end_time - moment.start_time
            try:
                verify_audio_language(Path(clip_path))
            except Exception as language_error:
                print(f"[-] Skipping non-English clip #{idx}: {language_error}")
                continue

            render_start = float(clip_info.get("segment_start", 0.0))
            render_end = render_start + clip_duration
            downloaded_duration = get_video_duration(Path(clip_path))
            if downloaded_duration > 0 and downloaded_duration + 0.5 < render_end:
                print(f"[-] Downloaded segment #{idx} is short ({downloaded_duration:.2f}s < {render_end:.2f}s). Skipping before render.")
                continue

            if framing_mode == "auto":
                print("[*] Running AI Face Detection & Speaker Tracking on segment...")
                framing = analyze_faces_in_clip(clip_path, render_start, render_end)
                print(f"[+] Framing decision: '{framing.mode}' (Detected faces: {framing.face_count})")
            else:
                framing = _manual_framing(clip_path, framing_mode)
            _print_framing_summary(framing)

            burn_subtitles = subtitles_mode in ("auto", "burn")
            ass_path = None
            clip_segments: List[TranscriptSegment] = segments or []
            editor_artifacts = _build_universal_shadow(
                run_id=run_id,
                state_store=state_store,
                video_path=Path(clip_path),
                segments=clip_segments,
                clip_start=moment.start_time,
                clip_end=moment.end_time,
                clip_index=idx,
            )
            if burn_subtitles and segments:
                try:
                    aligned_segments = align_clip_transcript(
                        clip_path=Path(clip_path),
                        render_start=render_start,
                        output_start=moment.start_time,
                        output_end=moment.end_time,
                        fallback_segments=segments,
                        model_size="tiny.en" if IS_CI else "base.en",
                    )
                    if aligned_segments is not segments:
                        print(f"[+] Local Whisper alignment refreshed clip #{idx} caption timing.")
                    clip_segments = aligned_segments
                except Exception as alignment_error:
                    print(f"[!] Local caption alignment unavailable for clip #{idx}: {alignment_error}")
                ass_path = SUBTITLES_DIR / f"clip_{idx}_{subtitle_style}.ass"
                create_styled_ass_subtitles(
                    segments=clip_segments,
                    clip_start=moment.start_time,
                    clip_end=moment.end_time,
                    output_ass_path=ass_path,
                    theme_key=subtitle_style,
                    layout_mode=framing.mode,
                    header_title=moment.title,
                    watermark=watermark,
                    shots=framing.shots,
                    keyword_emojis=getattr(moment, "keyword_emojis", None)
                )

            safe_title = "".join(c for c in moment.title if c.isalnum() or c in (" ", "_", "-")).rstrip()
            safe_title = safe_title.replace(" ", "_")[:30]
            out_clip_path = CLIPS_DIR / f"clip_{idx}_{safe_title}.mp4"

            # Pre-generate viral high-CTR thumbnail to embed as 0.25s opening flash cover
            thumb_path = None
            try:
                thumb_target = CLIPS_DIR / f"{out_clip_path.stem}_thumb.jpg"
                thumb_path = generate_clip_thumbnail(
                    clip_path=clip_path,
                    title=moment.title,
                    output_path=thumb_target,
                    extract_time=render_start + min(1.5, clip_duration / 2),
                    badge_text="MUST WATCH",
                    watermark=watermark
                )
            except Exception as te:
                print(f"[!] Thumbnail generation notice for clip #{idx}: {te}")

            # Detect and configure Pexels B-roll stock footage overlays
            broll_cues = []
            if clip_segments:
                clip_words = []
                for s in clip_segments:
                    for w in getattr(s, "words", []):
                        if moment.start_time <= w.start <= moment.end_time:
                            clip_words.append(WordTimestamp(
                                word=w.word,
                                start=w.start - moment.start_time,
                                end=w.end - moment.start_time,
                            ))
                if clip_words:
                    broll_cues = find_broll_cues_for_clip(clip_words, clip_duration=clip_duration)

            try:
                rendered_path = render_viral_clip(
                    source_video_path=clip_path,
                    start_time=render_start,
                    end_time=render_end,
                    output_clip_path=out_clip_path,
                    framing=framing,
                    peak_intensity_segments=getattr(moment, "peak_intensity_segments", []),
                    ass_subtitle_path=ass_path,
                    burn_subtitles=burn_subtitles,
                    sfx_cues=getattr(moment, "sfx_cues", []),
                    broll_cues=broll_cues,
                    cover_image_path=thumb_path
                )
                if not _run_universal_qa(
                    run_id=run_id,
                    state_store=state_store,
                    clip_index=idx,
                    rendered_path=Path(rendered_path),
                    artifacts=editor_artifacts,
                    clip_duration=clip_duration,
                ):
                    continue
                rendered_clips.append({
                    "path": rendered_path,
                    "thumbnail": thumb_path,
                    "moment": moment
                })
            except Exception as e:
                print(f"[-] Rendering failed for clip #{idx} ('{moment.title}'): {e}")
                continue

    # ------ BRANCH B: Fallback Full-Download Pipeline ------
    else:
        # Smart Whisper Probe Fallback:
        # If CLIP_ONLY_MODE is active but no transcript was found, we avoid
        # downloading the full video. Instead, we probe only the first 10 minutes
        # via Apify (a targeted range download), run faster-whisper STT on that
        # short audio probe, detect viral moments from the resulting segments,
        # then download ONLY those clip segments — keeping the zero-full-download contract.
        if CLIP_ONLY_MODE and is_youtube_url and not is_local_file:
            print("\n[!] No transcript found. Attempting smart Whisper probe (first 10 min via Apify)...")
            PROBE_END = 600.0  # 10 minutes — enough for viral moment detection
            probe_info = None
            if APIFY_API_TOKEN:
                try:
                    probe_info = download_clip_segment(
                        video_url=active_source_url,
                        start_time=0.0,
                        end_time=PROBE_END,
                        clip_index=0,
                        output_dir=DOWNLOADS_DIR,
                    )
                except Exception as e:
                    print(f"[-] Apify probe download failed: {e}")
            if probe_info:
                probe_path = Path(probe_info["video_path"])
                print(f"[+] Probe downloaded: {probe_path.name} ({probe_path.stat().st_size / (1024*1024):.1f} MB). Running Whisper STT...")
                try:
                    segments = transcribe_audio_whisper(probe_path)
                except Exception as e:
                    print(f"[-] Whisper STT on probe failed: {e}")
                    segments = []
                if segments:
                    print(f"[+] Whisper produced {len(segments)} segments from probe. Running viral detection...")
                    viral_moments = detect_viral_moments(segments, num_clips=num_clips, niche=niche)
                    if viral_moments:
                        print(f"[+] Viral moments detected from Whisper probe. Downloading {len(viral_moments)} targeted clips...")
                        DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
                        rendered_clips = []
                        for idx, moment in enumerate(viral_moments, 1):
                            try:
                                _extend_moment_to_complete_transcript(segments, moment)
                                clip_info = download_clip_segment(
                                    video_url=active_source_url,
                                    start_time=moment.start_time,
                                    end_time=moment.end_time,
                                    clip_index=idx,
                                    output_dir=DOWNLOADS_DIR,
                                )
                                if not clip_info:
                                    print(f"[-] Targeted clip #{idx} download returned nothing. Skipping.")
                                    continue
                                clip_path = Path(clip_info["video_path"])
                                try:
                                    verify_audio_language(clip_path)
                                except Exception as language_error:
                                    print(f"[-] Skipping non-English probe clip #{idx}: {language_error}")
                                    continue
                                render_start = 0.0
                                render_end = moment.end_time - moment.start_time
                                editor_artifacts = _build_universal_shadow(
                                    run_id=run_id,
                                    state_store=state_store,
                                    video_path=clip_path,
                                    segments=segments,
                                    clip_start=moment.start_time,
                                    clip_end=moment.end_time,
                                    clip_index=idx,
                                )
                                if framing_mode == "auto":
                                    framing = analyze_faces_in_clip(
                                        clip_path,
                                        render_start,
                                        render_end,
                                    )
                                else:
                                    framing = _manual_framing(clip_path, framing_mode)
                                _print_framing_summary(framing)
                                ass_path = SUBTITLES_DIR / f"probe_clip_{idx}.ass"
                                burn_subtitles = subtitles_mode != "skip"
                                if burn_subtitles:
                                    create_styled_ass_subtitles(
                                        segments=segments,
                                        clip_start=moment.start_time,
                                        clip_end=moment.end_time,
                                        output_ass_path=ass_path,
                                        theme_key=subtitle_style,
                                        layout_mode=framing.mode,
                                        header_title=moment.title,
                                        watermark=watermark,
                                        shots=framing.shots,
                                        keyword_emojis=getattr(moment, "keyword_emojis", None)
                                    )
                                safe_title = "".join(c for c in moment.title if c.isalnum() or c in (" ", "_", "-")).rstrip()
                                safe_title = safe_title.replace(" ", "_")[:30]
                                out_clip_path = CLIPS_DIR / f"clip_{idx}_{safe_title}.mp4"

                                thumb_path = None
                                try:
                                    thumb_target = CLIPS_DIR / f"{out_clip_path.stem}_thumb.jpg"
                                    thumb_path = generate_clip_thumbnail(
                                        clip_path=clip_path,
                                        title=moment.title,
                                        output_path=thumb_target,
                                        extract_time=1.5,
                                        badge_text="MUST WATCH",
                                        watermark=watermark
                                    )
                                except Exception as te:
                                    print(f"[!] Thumbnail generation notice for probe clip #{idx}: {te}")

                                rendered_path = render_viral_clip(
                                    source_video_path=clip_path,
                                    start_time=render_start,
                                    end_time=render_end,
                                    output_clip_path=out_clip_path,
                                    framing=framing,
                                    peak_intensity_segments=getattr(moment, "peak_intensity_segments", []),
                                    ass_subtitle_path=ass_path,
                                    burn_subtitles=burn_subtitles,
                                    sfx_cues=getattr(moment, "sfx_cues", []),
                                    cover_image_path=thumb_path
                                )
                                if not _run_universal_qa(
                                    run_id=run_id,
                                    state_store=state_store,
                                    clip_index=idx,
                                    rendered_path=Path(rendered_path),
                                    artifacts=editor_artifacts,
                                    clip_duration=render_end - render_start,
                                ):
                                    continue
                                rendered_clips.append({"path": rendered_path, "thumbnail": thumb_path, "moment": moment})
                            except Exception as e:
                                print(f"[-] Probe clip #{idx} failed: {e}")
                                continue
                        # Jump to the Buffer/upload section by falling through
                        # to the rendered_clips handler below (skip the old full-download branch)
                        pass
                    else:
                        print("[-] Whisper probe found no viral moments. Cannot produce clips without captions or viable transcript.")
                        sys.exit(1)
                else:
                    print("[-] Whisper STT produced no segments from probe. Exiting.")
                    sys.exit(1)
            else:
                raise RuntimeError(
                    "No native captions, Apify transcript, or Apify probe download available. "
                    "Cannot produce clips without a transcript source."
                )
            # Rendered clips from probe path — fall through to Buffer upload section
            if not rendered_clips:
                print("[-] No clips rendered from Whisper probe. Exiting.")
                sys.exit(1)
        else:
            print("\n--- [3/6] FULL VIDEO DOWNLOAD (Fallback: no transcript / local file) ---")

            download_info = None
            for cand_url in candidates:
                try:
                    print(f"[*] Ingesting video candidate: {cand_url}")
                    download_info = download_video(cand_url)
                    if download_info:
                        active_source_url = cand_url
                        break
                except Exception as e:
                    print(f"[-] Candidate '{cand_url}' download failed: {e}. Trying next candidate...")

            if not download_info:
                print("[-] Fatal: All candidate downloads failed. Exiting.")
                sys.exit(1)

            video_path = download_info['video_path']
            video_title = download_info['title']
            native_transcript_fb = download_info.get('transcript')
            print(f"[+] Active video file: {video_path}")

            print("\n--- [2b/6] SPEECH-TO-TEXT / TRANSCRIPT EXTRACTION ---")
            segments = get_transcript(video_path, native_transcript_fb, video_id=download_info.get('video_id'))
            if not segments:
                print("[-] Error: Could not obtain transcript for video. Exiting.")
                sys.exit(1)

            print("\n--- [3b/6] VIRAL MOMENT HUNTING & HOOK SCORING (Fallback) ---")
            viral_moments = detect_viral_moments(segments, num_clips=num_clips, niche=niche)
            if not viral_moments:
                print("[-] No viral moments detected. Exiting.")
                sys.exit(1)

            print(f"\n[+] Top {len(viral_moments)} Viral Moments Identified:")
            for idx, m in enumerate(viral_moments, 1):
                print(f"   #{idx}: [{m.start_time:.1f}s - {m.end_time:.1f}s] (Virality: {m.viral_score}/100) '{m.title}'")

            print("\n--- [4/6 & 5/6] EDITING, FACE TRACKING, AUDIO MASTERING & RENDERING ---")
            try:
                verify_audio_language(Path(video_path))
            except Exception as language_error:
                raise RuntimeError(f"Source audio is not English: {language_error}") from language_error
            for idx, moment in enumerate(viral_moments, 1):
                _extend_moment_to_complete_transcript(segments, moment)
                print(f"\n>>> Processing Clip #{idx}: {moment.title} ({moment.duration:.1f}s)")
                try:
                    if framing_mode == "auto":
                        print("[*] Running AI Face Detection & Speaker Tracking...")
                        framing = analyze_faces_in_clip(video_path, moment.start_time, moment.end_time)
                        print(f"[+] Framing decision: '{framing.mode}' (Detected faces: {framing.face_count})")
                    else:
                        framing = _manual_framing(video_path, framing_mode)
                    _print_framing_summary(framing)

                    editor_artifacts = _build_universal_shadow(
                        run_id=run_id,
                        state_store=state_store,
                        video_path=Path(video_path),
                        segments=segments,
                        clip_start=moment.start_time,
                        clip_end=moment.end_time,
                        clip_index=idx,
                        source_window=(moment.start_time, moment.end_time),
                    )
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
                            layout_mode=framing.mode,
                            header_title=moment.title,
                            watermark=watermark,
                            shots=framing.shots,
                            keyword_emojis=getattr(moment, "keyword_emojis", None)
                        )

                    safe_title = "".join(c for c in moment.title if c.isalnum() or c in (" ", "_", "-")).rstrip()
                    safe_title = safe_title.replace(" ", "_")[:30]
                    out_clip_path = CLIPS_DIR / f"clip_{idx}_{safe_title}.mp4"

                    thumb_path = None
                    try:
                        thumb_target = CLIPS_DIR / f"{out_clip_path.stem}_thumb.jpg"
                        thumb_path = generate_clip_thumbnail(
                            clip_path=video_path,
                            title=moment.title,
                            output_path=thumb_target,
                            extract_time=moment.start_time + 1.5,
                            badge_text="MUST WATCH",
                            watermark=watermark
                        )
                    except Exception as te:
                        print(f"[!] Thumbnail generation notice for clip #{idx}: {te}")

                    rendered_path = render_viral_clip(
                        source_video_path=video_path,
                        start_time=moment.start_time,
                        end_time=moment.end_time,
                        output_clip_path=out_clip_path,
                        framing=framing,
                        peak_intensity_segments=getattr(moment, "peak_intensity_segments", []),
                        ass_subtitle_path=ass_path,
                        burn_subtitles=burn_subtitles,
                        sfx_cues=getattr(moment, "sfx_cues", []),
                        cover_image_path=thumb_path
                    )
                    if not _run_universal_qa(
                        run_id=run_id,
                        state_store=state_store,
                        clip_index=idx,
                        rendered_path=Path(rendered_path),
                        artifacts=editor_artifacts,
                        clip_duration=moment.end_time - moment.start_time,
                    ):
                        continue
                    rendered_clips.append({"path": rendered_path, "thumbnail": thumb_path, "moment": moment})
                except Exception as e:
                    print(f"[-] Full-download clip #{idx} failed: {e}")
                    continue

    if not rendered_clips:
        raise RuntimeError(
            "No clips were rendered. Refusing to report a successful run without publishable output."
        )

    # Step 6: Release Hosting & Buffer Social Distribution
    print("\n--- [6/6] RELEASE HOSTING & BUFFER SOCIAL PUBLISHING ---")
    buffer_client = BufferClient() if post_to_buffer or BUFFER_ACCESS_TOKEN else None
    publish_failed = False
    clips_report = []
    for item in rendered_clips:
        clip_path = item["path"]
        thumb_path = item.get("thumbnail")
        moment = item["moment"]

        direct_url = upload_clip_to_github_release(clip_path)
        thumb_url = upload_clip_to_github_release(thumb_path) if thumb_path and thumb_path.exists() else None

        buffer_status = "Local Only"
        if post_to_buffer and direct_url:
            caption_text = f"{moment.title}\n\n{moment.social_caption}\n\n{' '.join(moment.hashtags)}"
            if buffer_client is None:
                buffer_status = "No Client"
                publish_failed = True
            else:
                schedule_results = buffer_client.schedule_video_post(
                    video_url=direct_url,
                    text=caption_text,
                    title=moment.title,
                    source_url=active_source_url,
                    thumbnail_url=thumb_url
                )
                successful_posts = sum(
                    1
                    for result in schedule_results
                    if (((result.get("response") or {}).get("data") or {}).get("createPost") or {}).get("post")
                )
                if schedule_results and successful_posts == len(schedule_results):
                    buffer_status = "Scheduled"
                elif successful_posts:
                    buffer_status = "Partial"
                    publish_failed = True
                else:
                    buffer_status = "Failed"
                    publish_failed = True
        elif post_to_buffer and not direct_url:
            print("[-] Cannot post to Buffer because direct video URL is not available.")
            buffer_status = "No URL"
            publish_failed = True

        clips_report.append({
            "title": moment.title,
            "virality_score": getattr(moment, "viral_score", 85),
            "duration": moment.end_time - moment.start_time,
            "download_url": direct_url or "",
            "thumbnail_url": thumb_url or "",
            "buffer_status": buffer_status
        })

    # Step 7: Storage Hygiene: Auto-delete releases older than 5 days
    print("\n--- [7/7] STORAGE HYGIENE: AUTO-CLEANUP RELEASES > 5 DAYS ---")
    try:
        from src.release_cleaner import clean_old_releases
        clean_old_releases(days=5, buffer_client=buffer_client)
    except Exception as e:
        print(f"[-] Auto-cleanup warning: {e}")

    # Dispatch Slack Operational Report Card
    try:
        slack = SlackNotifier()
        slack.send_run_report(
            podcast_title=video_title,
            podcast_url=active_source_url,
            niche=niche,
            clips=clips_report
        )
    except Exception as e:
        print(f"[-] Slack alert warning: {e}")

    if publish_failed:
        raise RuntimeError("One or more clips failed required hosting or Buffer publishing.")

    try:
        record_history(video_url=active_source_url, title=video_title, niche=niche)
    except Exception as e:
        print(f"[-] History record warning: {e}")

    state_store.set_stage(run_id, "pipeline", StageStatus.COMPLETED)
    state_store.set_status(run_id, "completed")
    print("\n" + "=" * 70)
    print("✨ ALL CLIPS PROCESSED SUCCESSFULLY!")
    print(f"📂 Output clips saved to: {CLIPS_DIR.resolve()}")
    print("=" * 70)

def main():
    # Debug: print arguments received by the script
    import sys
    print(f"[*] CLI Arguments received: {sys.argv[1:]}")
    
    parser = argparse.ArgumentParser(
        description="Autonomous AI Podcast Viral Clipper & Buffer Social Media Publisher ($0 Budget)"
    )
    parser.add_argument(
        "--url", "-u",
        required=False,
        default=None,
        help="YouTube video URL, podcast link, or path to local video file."
    )
    parser.add_argument(
        "--niche",
        choices=["finance", "business", "ai_tech", "health_longevity", "real_estate", "mindset", "auto"],
        default="finance",
        help="Target high-CPM niche for automatic podcast discovery (default: finance - Personal Wealth & Investing)."
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
    parser.add_argument(
        "--watermark", "-w",
        type=str,
        default=None,
        help="Channel watermark handle to burn at 50%% opacity."
    )

    args = parser.parse_args()

    target_url = args.url
    resolved_niche = args.niche
    if not target_url or target_url.strip().lower() in ("auto", "none", ""):
        print("[*] No URL provided. Activating Automated High-CPM Podcast Discovery...")
        from src.channel_discovery import get_daily_discovery_candidates, resolve_daily_niche
        resolved_niche = resolve_daily_niche(args.niche if args.niche != "auto" else None)
        target_url = get_daily_discovery_candidates(niche=resolved_niche)
    elif resolved_niche == "auto":
        from src.channel_discovery import resolve_daily_niche
        resolved_niche = resolve_daily_niche()

    try:
        run_pipeline(
            url_or_path=target_url,
            num_clips=args.num_clips,
            framing_mode=args.framing,
            subtitle_style=args.subtitle_style,
            subtitles_mode=args.subtitles,
            post_to_buffer=args.post_to_buffer,
            dry_run=args.dry_run,
            watermark=args.watermark,
            niche=resolved_niche
        )
    except Exception as e:
        print("\n" + "!" * 70)
        print("💥 FATAL PIPELINE CRASH DETECTED")
        print("!" * 70)
        traceback.print_exc()
        print("\n" + "!" * 70)

        # Automated error observability: dispatch high-priority alert to Slack
        try:
            from src.slack_notifier import SlackNotifier
            notifier = SlackNotifier()
            notifier.send_error_alert(
                error_message=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-300:]}",
                podcast_url=str(target_url)[:100] if target_url else "N/A"
            )
        except Exception:
            pass

        sys.exit(1)

if __name__ == "__main__":
    main()
