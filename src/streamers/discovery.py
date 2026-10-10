"""Streamer Discovery Engine (V2).

Discovers fresh high-engagement YouTube VODs & streams from top US creators:
- Kai Cenat
- IShowSpeed
- Jynxzi
- CaseOh
- xQc
- Adin Ross
- FlightReacts

Maintains an isolated history store at data/streamer_history.json so V1 podcast
history is never touched or affected.
"""

import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from src.config import DATA_DIR

STREAMER_HISTORY_FILE = DATA_DIR / "streamer_history.json"

STREAMER_CHANNELS: Dict[str, Dict[str, Any]] = {
    "kai_cenat": {
        "name": "Kai Cenat",
        "category": "just_chatting",
        "channel_urls": [
            "https://www.youtube.com/@KaiCenatLive/videos",
            "https://www.youtube.com/@KaiCenat/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCm03u_jV1-G7u7S9ZgE9sOw",
        ],
        "keywords": ["Kai Cenat", "AMP", "Mafia", "stream", "room", "rage"],
    },
    "ishowspeed": {
        "name": "IShowSpeed",
        "category": "irl",
        "channel_urls": [
            "https://www.youtube.com/@IShowSpeed/videos",
            "https://www.youtube.com/@LiveSpeedy/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCWsD388ETZql143SFkk9VvA",
        ],
        "keywords": ["Speed", "IShowSpeed", "stream", "bark", "rage", "irl"],
    },
    "jynxzi": {
        "name": "Jynxzi",
        "category": "gaming",
        "channel_urls": [
            "https://www.youtube.com/@Jynxzi/videos",
            "https://www.youtube.com/@JynxziClips/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UC0L_q8F102r3gY4U32bF39A",
        ],
        "keywords": ["Jynxzi", "R6", "Rainbow Six", "reaction", "clip", "gaming"],
    },
    "caseoh": {
        "name": "CaseOh",
        "category": "gaming",
        "channel_urls": [
            "https://www.youtube.com/@CaseOh_/videos",
            "https://www.youtube.com/@CaseOhGames/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCp8U9a9_S2lQnQ25h9oT_qA",
        ],
        "keywords": ["CaseOh", "horror", "gameplay", "funny", "rage", "reaction"],
    },
    "xqc": {
        "name": "xQc",
        "category": "just_chatting",
        "channel_urls": [
            "https://www.youtube.com/@xQcOW/videos",
            "https://www.youtube.com/@xQcClips/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCmDTrq0LNgPodDOFZiSbSwA",
        ],
        "keywords": ["xQc", "reacts", "stream", "drama", "debate", "gaming"],
    },
    "adin_ross": {
        "name": "Adin Ross",
        "category": "just_chatting",
        "channel_urls": [
            "https://www.youtube.com/@AdinLive/videos",
        ],
        "rss_feeds": [
            "https://www.youtube.com/feeds/videos.xml?channel_id=UCgs_U0_x2BwE_J_7vH3oFzA",
        ],
        "keywords": ["Adin Ross", "stream", "interview", "funny", "reaction"],
    },
}


def load_streamer_history() -> List[str]:
    """Loads list of previously processed streamer video IDs."""
    if not STREAMER_HISTORY_FILE.exists():
        return []
    try:
        data = json.loads(STREAMER_HISTORY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(v) for v in data]
        if isinstance(data, dict):
            return [str(v) for v in data.get("processed_video_ids", [])]
    except Exception:
        pass
    return []


def record_streamer_history(video_id: str, title: str = "", creator: str = "") -> None:
    """Records a processed video ID into streamer_history.json."""
    STREAMER_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    history = load_streamer_history()
    if video_id not in history:
        history.append(video_id)
    try:
        STREAMER_HISTORY_FILE.write_text(
            json.dumps({"processed_video_ids": history[-500:]}, indent=2),
            encoding="utf-8"
        )
    except Exception as e:
        print(f"[-] Failed to update streamer history: {e}")


def extract_video_id_from_url(url: str) -> Optional[str]:
    """Extracts YouTube 11-char video ID from any standard URL."""
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|\/embed\/|\/live\/|\/shorts\/)([a-zA-Z0-9_-]{11})",
        r"^([a-zA-Z0-9_-]{11})$"
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def fetch_rss_feed_videos(feed_url: str, limit: int = 5) -> List[Dict[str, str]]:
    """Fetches candidate videos from YouTube RSS feed without downloading any video."""
    candidates = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        req = urllib.request.Request(feed_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read()
        root = ET.fromstring(content)
        ns = {"atom": "http://www.w3.org/2005/Atom", "yt": "http://www.youtube.com/xml/schemas/2015"}
        for entry in root.findall("atom:entry", ns)[:limit]:
            vid_elem = entry.find("yt:videoId", ns)
            title_elem = entry.find("atom:title", ns)
            if vid_elem is not None and vid_elem.text:
                vid_id = vid_elem.text.strip()
                title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                candidates.append({
                    "video_id": vid_id,
                    "url": f"https://www.youtube.com/watch?v={vid_id}",
                    "title": title,
                })
    except Exception:
        pass
    return candidates


def discover_streamer_candidates(
    creator: Optional[str] = None,
    category: Optional[str] = None,
    max_candidates: int = 10
) -> List[Dict[str, str]]:
    """Discovers fresh, unprocessed streamer VODs and stream clips."""
    history = set(load_streamer_history())
    selected_streamers: List[Dict[str, Any]] = []

    if creator and creator.lower() in STREAMER_CHANNELS:
        selected_streamers.append(STREAMER_CHANNELS[creator.lower()])
    else:
        for key, s in STREAMER_CHANNELS.items():
            if category and s.get("category") != category.lower():
                continue
            selected_streamers.append(s)

    discovered = []
    for s in selected_streamers:
        feeds = s.get("rss_feeds", [])
        for feed in feeds:
            videos = fetch_rss_feed_videos(feed, limit=6)
            for v in videos:
                vid_id = v["video_id"]
                if vid_id not in history:
                    discovered.append({
                        "creator": s["name"],
                        "category": s.get("category", "streamer"),
                        "video_id": vid_id,
                        "url": v["url"],
                        "title": v["title"],
                    })
                if len(discovered) >= max_candidates:
                    return discovered

    return discovered
