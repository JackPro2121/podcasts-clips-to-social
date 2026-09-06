import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import yt_dlp
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from src.config import DOWNLOADS_DIR, APIFY_API_TOKEN

def extract_youtube_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from various URL formats."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'(?:shorts\/)([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def fetch_youtube_transcript(video_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    Attempt to fetch native YouTube transcript (free & instant).
    Compatible with both youtube-transcript-api 0.x and 1.x.
    Returns list of dicts: [{'text': str, 'start': float, 'duration': float}]
    """
    raw_snippets = None
    try:
        # Try new API (1.x+)
        api = YouTubeTranscriptApi()
        if hasattr(api, "fetch"):
            raw_snippets = api.fetch(video_id)
        elif hasattr(api, "list"):
            transcripts = api.list(video_id)
            for t in transcripts:
                raw_snippets = t.fetch()
                break
    except Exception:
        pass

    if raw_snippets is None:
        try:
            # Fallback to legacy static method (0.x)
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                raw_snippets = YouTubeTranscriptApi.get_transcript(video_id)
        except Exception as e:
            print(f"[-] Native YouTube transcript fetch failed ({e}). Will use speech-to-text fallback.")
            return None

    if not raw_snippets:
        return None

    # Normalize snippets into standard dict format
    normalized = []
    for item in raw_snippets:
        text = getattr(item, "text", None) if not isinstance(item, dict) else item.get("text")
        start = getattr(item, "start", 0.0) if not isinstance(item, dict) else item.get("start", 0.0)
        duration = getattr(item, "duration", 0.0) if not isinstance(item, dict) else item.get("duration", 0.0)
        if text:
            normalized.append({
                "text": str(text),
                "start": float(start),
                "duration": float(duration)
            })

    print(f"[+] Retrieved native YouTube transcript ({len(normalized)} snippets).")
    return normalized

def download_via_apify(
    video_url: str,
    output_dir: Path,
    quality: str = "1080",
    api_token: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Downloads YouTube video using Apify Actor (streamers/youtube-video-downloader).
    Guarantees bypass of bot captchas and datacenter IP blocks on GitHub Actions.
    """
    token = api_token or APIFY_API_TOKEN
    if not token:
        return None

    print(f"[*] Dispatching YouTube download via Apify Actor (streamers/youtube-video-downloader)...")
    endpoint = "https://api.apify.com/v2/acts/streamers~youtube-video-downloader/runs"
    payload = {
        "videos": [{"url": video_url}]
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if res.status_code not in (200, 201):
            print(f"[-] Apify actor start failed: {res.status_code} - {res.text}")
            return None

        run_id = res.json().get("data", {}).get("id")
        if not run_id:
            print("[-] Could not retrieve Apify run ID.")
            return None

        print(f"[*] Apify run started ({run_id}). Polling for download completion (up to 10 mins)...")
        final_run_data = None
        for attempt in range(150):
            time.sleep(4)
            status_res = requests.get(
                f"https://api.apify.com/v2/actor-runs/{run_id}",
                headers=headers,
                timeout=15
            )
            if status_res.status_code == 200:
                final_run_data = status_res.json().get("data", {})
                status = final_run_data.get("status")
                if attempt % 5 == 0:
                    print(f"[*] Apify download status: {status} (elapsed: {(attempt + 1) * 4}s)")
                if status in ("SUCCEEDED", "FAILED", "TIMED-OUT", "ABORTED"):
                    break

        if not final_run_data or final_run_data.get("status") != "SUCCEEDED":
            print(f"[-] Apify actor finished with non-success state: {final_run_data.get('status') if final_run_data else 'unknown'}")
            return None

        dataset_id = final_run_data.get("defaultDatasetId")

        # Fetch output items
        items_res = requests.get(
            f"https://api.apify.com/v2/datasets/{dataset_id}/items",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )
        items = items_res.json()
        if not items or not isinstance(items, list):
            print("[-] No output items found in Apify dataset.")
            return None

        first_item = items[0]
        direct_url = first_item.get("downloadedFileUrl") or first_item.get("output", {}).get("url")
        if not direct_url:
            print("[-] Apify output missing direct video download URL.")
            return None

        vid_id = first_item.get("id") or extract_youtube_id(video_url) or "video"
        out_file = output_dir / f"{vid_id}.mp4"

        print(f"[*] Streaming high-quality video from Apify storage ({out_file.name})...")
        with requests.get(direct_url, headers=headers, stream=True, timeout=180) as stream_res:
            stream_res.raise_for_status()
            with open(out_file, "wb") as f:
                for chunk in stream_res.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        print(f"[+] Download complete via Apify: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
        return {
            "video_path": out_file.resolve(),
            "title": f"YouTube_{vid_id}",
            "duration": float(first_item.get("durationSeconds", 0.0)),
            "is_local": False
        }
    except Exception as e:
        print(f"[-] Apify download encountered an exception: {e}")
        return None

def download_video(url_or_path: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Downloads the video using Apify Actor (preferred) or yt-dlp fallback.
    """
    target_dir = output_dir or DOWNLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if local file
    local_path = Path(url_or_path)
    if local_path.exists() and local_path.is_file():
        return {
            'video_path': local_path.resolve(),
            'title': local_path.stem,
            'duration': 0.0,
            'transcript': None,
            'is_local': True
        }

    # Fetch native transcript if YouTube URL
    video_id = extract_youtube_id(url_or_path)
    transcript = None
    if video_id:
        transcript = fetch_youtube_transcript(video_id)

    # 1. If Apify API token is configured, use Apify for 100% cloud bot-bypass
    if APIFY_API_TOKEN and ("youtube.com" in url_or_path or "youtu.be" in url_or_path):
        apify_result = download_via_apify(url_or_path, target_dir, quality="1080")
        if apify_result:
            apify_result['transcript'] = transcript
            return apify_result
        print("[!] Apify download failed or timed out. Falling back to yt-dlp...")

    # Download with yt-dlp (up to 1080p, AAC audio, mp4 container)
    out_template = str(target_dir / "%(id)s_%(title).50s.%(ext)s")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': out_template,
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'tv']
            }
        },
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }]
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url_or_path, download=True)
        downloaded_file = ydl.prepare_filename(info)
        # Handle possible extension change after merge
        if not os.path.exists(downloaded_file):
            candidate = str(Path(downloaded_file).with_suffix('.mp4'))
            if os.path.exists(candidate):
                downloaded_file = candidate

        return {
            'video_path': Path(downloaded_file).resolve(),
            'title': info.get('title', 'podcast_episode'),
            'duration': float(info.get('duration', 0.0)),
            'transcript': transcript,
            'is_local': False
        }
