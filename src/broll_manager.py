import os
import re
import requests
from pathlib import Path
from urllib.parse import quote
from typing import List, Optional, Tuple, Dict, Any

from src.config import PEXELS_API_KEY, BROLL_DIR, ENABLE_BROLL

# Keywords that trigger high-retention B-roll stock footage overlays
BROLL_KEYWORD_MAP = {
    "money": "money cash",
    "dollar": "dollars cash",
    "cash": "counting cash",
    "wealth": "luxury lifestyle rich",
    "rich": "luxury mansion car",
    "million": "stacks of money",
    "millionaire": "successful businessman luxury",
    "invest": "stock market trading",
    "investing": "stock market charts",
    "stocks": "stock exchange numbers",
    "stock": "stock market trading",
    "crypto": "bitcoin cryptocurrency",
    "bitcoin": "bitcoin gold coin",
    "debt": "stressed person bills",
    "broke": "empty wallet broke",
    "bank": "bank building money",
    "credit": "credit card payment",
    "real estate": "luxury modern house",
    "house": "modern house interior",
    "property": "skyscrapers city skyline",
    "tax": "tax calculation calculator",
    "taxes": "tax forms calculator",
    "jail": "prison bars jail",
    "lawsuit": "courtroom gavel judge",
    "lawyer": "courtroom gavel judge",
    "scam": "handcuffs police crime",
    "fired": "office worker packing box",
    "job": "modern office tech workplace",
    "salary": "handshake business deal",
    "profit": "profit charts growth",
    "growth": "rocket launch success"
}

def search_pexels_broll(
    query: str,
    api_key: Optional[str] = None,
    per_page: int = 3
) -> List[Dict[str, Any]]:
    """
    Queries Pexels Video API for high-resolution stock video footage matching query.
    Prioritizes 9:16 portrait orientation, falling back to all orientations if needed.
    """
    key = api_key or PEXELS_API_KEY
    if not key:
        return []

    headers = {"Authorization": key}
    encoded_q = quote(query.strip())

    # Try portrait orientation first for native 9:16 reels/shorts
    url_portrait = f"https://api.pexels.com/videos/search?query={encoded_q}&orientation=portrait&size=medium&per_page={per_page}"
    try:
        r = requests.get(url_portrait, headers=headers, timeout=12)
        if r.status_code == 200:
            data = r.json()
            videos = data.get("videos", [])
            if videos:
                return videos
    except Exception as e:
        print(f"[-] Pexels portrait search warning: {e}")

    # Fallback to general orientation
    url_fallback = f"https://api.pexels.com/videos/search?query={encoded_q}&size=medium&per_page={per_page}"
    try:
        r = requests.get(url_fallback, headers=headers, timeout=12)
        if r.status_code == 200:
            data = r.json()
            return data.get("videos", [])
    except Exception as e:
        print(f"[-] Pexels fallback search error: {e}")
    return []

def select_best_video_file(video_entry: Dict[str, Any]) -> Optional[str]:
    """Selects the best quality MP4 file URL from a Pexels video result."""
    files = video_entry.get("video_files", [])
    if not files:
        return None

    # Prefer HD (720p or 1080p) MP4 files
    mp4_files = [f for f in files if f.get("file_type") == "video/mp4"]
    if not mp4_files:
        mp4_files = files

    # Sort by height descending up to 1920 (optimal for performance & quality)
    def quality_score(f):
        h = f.get("height") or 0
        w = f.get("width") or 0
        return min(h, 1920) + min(w, 1080)

    mp4_files.sort(key=quality_score, reverse=True)
    return mp4_files[0].get("link")

def download_broll_clip(
    video_url: str,
    keyword: str,
    output_dir: Optional[Path] = None
) -> Optional[Path]:
    """Downloads B-roll video clip and caches locally to prevent redundant downloads."""
    out_dir = output_dir or BROLL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', keyword).lower()
    # Unique filename based on url hash/id
    url_id = video_url.split("/")[-1].split("?")[0]
    if not url_id.endswith(".mp4"):
        url_id = f"pexels_{abs(hash(video_url)) % 1000000}.mp4"
    dest_path = out_dir / f"broll_{safe_name}_{url_id}"

    if dest_path.exists() and dest_path.stat().st_size > 10000:
        return dest_path

    try:
        print(f"[*] Downloading B-roll clip for '{keyword}'...")
        r = requests.get(video_url, stream=True, timeout=20)
        if r.status_code == 200:
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
            print(f"[+] B-roll downloaded: {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB)")
            return dest_path
    except Exception as e:
        print(f"[-] B-roll download failed: {e}")
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
    return None

def find_broll_cues_for_clip(
    words: List[Any],
    clip_duration: float,
    max_brolls: int = 2,
    min_spacing: float = 8.0
) -> List[Tuple[float, float, Path, str]]:
    """
    Identifies high-impact keywords in spoken words and fetches matching Pexels B-roll.
    Returns: List of (start_t, end_t, broll_file_path, keyword)
    """
    if not ENABLE_BROLL or not PEXELS_API_KEY:
        return []

    cues: List[Tuple[float, float, Path, str]] = []
    last_broll_end = 4.0  # Don't place B-roll in the initial 4-second hook window

    for w in words:
        if len(cues) >= max_brolls:
            break

        w_text = getattr(w, "word", "").lower().strip()
        w_text = re.sub(r'[^a-z]', '', w_text)
        w_start = getattr(w, "start", 0.0)

        # Ensure B-roll starts after hook and has adequate spacing
        if w_start < last_broll_end or (w_start + 3.0) > (clip_duration - 1.5):
            continue

        if w_text in BROLL_KEYWORD_MAP:
            search_query = BROLL_KEYWORD_MAP[w_text]
            results = search_pexels_broll(search_query)
            if results:
                best_url = select_best_video_file(results[0])
                if best_url:
                    broll_file = download_broll_clip(best_url, w_text)
                    if broll_file and broll_file.exists():
                        broll_dur = min(3.5, clip_duration - w_start - 0.5)
                        broll_end = w_start + broll_dur
                        cues.append((round(w_start, 2), round(broll_end, 2), broll_file, w_text))
                        last_broll_end = broll_end + min_spacing

    if cues:
        print(f"[+] Configured {len(cues)} Pexels B-roll overlay(s) for clip.")
    return cues
