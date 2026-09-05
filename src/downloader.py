import os
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from src.config import DOWNLOADS_DIR

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
    Returns list of dicts: [{'text': str, 'start': float, 'duration': float}]
    """
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try finding manually created English transcript, else auto-generated English
        try:
            transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
        except Exception:
            transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
        
        data = transcript.fetch()
        return data
    except Exception as e:
        print(f"[-] Native YouTube transcript not available ({e}). Will use speech-to-text fallback.")
        return None

def download_video(url_or_path: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Downloads the video using yt-dlp or validates existing local file.
    Returns: {
        'video_path': Path,
        'title': str,
        'duration': float,
        'transcript': Optional[List[Dict]],
        'is_local': bool
    }
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

    # Download with yt-dlp (up to 1080p, AAC audio, mp4 container)
    out_template = str(target_dir / "%(id)s_%(title).50s.%(ext)s")
    ydl_opts = {
        'format': 'bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': out_template,
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': True,
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
