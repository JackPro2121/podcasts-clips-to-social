import os
import sys
import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import YOUTUBE_COOKIES, MAX_DISCOVERY_CANDIDATES
from src.downloader import _write_cookiefile

# Curated High CPM & RPM Podcast Niches
HIGH_CPM_NICHES = {
    "finance": {
        "title": "💰 Personal Finance, Investing & Wealth",
        "estimated_cpm": "$30 - $65+ CPM / RPM",
        "rationale": "Financial institutions, stock brokers, crypto platforms, and credit card companies bid the highest advertising dollars on social media.",
        "channel_urls": [
            "https://www.youtube.com/@TheRamseyShow/videos",
            "https://www.youtube.com/@TheIcedCoffeeHour/videos",
            "https://www.youtube.com/@CalebHammer/videos",
            "https://www.youtube.com/@MyFirstMillionPod/videos",
            "https://www.youtube.com/@TheMoneyGuyShow/videos",
            "https://www.youtube.com/@GrahamStephan/videos",
            "https://www.youtube.com/@BiggerPockets/videos",
            "https://www.youtube.com/@humphreytalks/videos",
            "https://www.youtube.com/@AndreiJikh/videos",
            "https://www.youtube.com/@marktilbury/videos",
            "https://www.youtube.com/@InvestorsPodcast/videos",
            "https://www.youtube.com/@DamienTalksMoney/videos",
            "https://www.youtube.com/@Pensioncraft/videos",
            "https://www.youtube.com/@PBDPodcast/videos",
        ],
        "search_queries": [
            "The Ramsey Show debt free screams episode",
            "Caleb Hammer Financial Audit episode full",
            "The Iced Coffee Hour podcast episode",
            "My First Million podcast full episode",
            "The Money Guy Show full podcast episode",
            "BiggerPockets Money Podcast episode",
            "Graham Stephan podcast full episode",
            "Patrick Bet-David PBD podcast money wealth",
        ]
    },
    "business": {
        "title": "📈 Business, Startups & Entrepreneurship",
        "estimated_cpm": "$25 - $50+ CPM / RPM",
        "rationale": "High-value B2B SaaS, CRM tools (HubSpot/Salesforce), enterprise software, and payment processors advertise heavily here.",
        "channel_urls": [
            "https://www.youtube.com/@TheDiaryOfACEO/videos",
            "https://www.youtube.com/@AlexHormozi/videos",
            "https://www.youtube.com/@VALUETAINMENT/videos",
            "https://www.youtube.com/@TimFerriss/videos",
            "https://www.youtube.com/@MastersofScale/videos",
            "https://www.youtube.com/@joerogan/videos",
            "https://www.youtube.com/@colinandsamir/videos",
            "https://www.youtube.com/@SimonSquibb/videos",
        ],
        "search_queries": [
            "The Diary Of A CEO podcast full episode",
            "The Tim Ferriss Show podcast full",
            "The Game with Alex Hormozi podcast",
            "Masters of Scale podcast episode",
            "Joe Rogan podcast business entrepreneur",
            "Valuetainment Patrick Bet-David podcast",
        ]
    },
    "ai_tech": {
        "title": "🤖 Artificial Intelligence & Tech Trends",
        "estimated_cpm": "$25 - $45+ CPM / RPM",
        "rationale": "AI tool creators, developer tools, cloud providers (AWS/GCP), and venture capital firms target tech audiences.",
        "channel_urls": [
            "https://www.youtube.com/@lexfridman/videos",
            "https://www.youtube.com/@allin/videos",
            "https://www.youtube.com/@DwarkeshPatel/videos",
            "https://www.youtube.com/@YCombinator/videos",
            "https://www.youtube.com/@a16z/videos",
            "https://www.youtube.com/@OpenAI/videos",
        ],
        "search_queries": [
            "Lex Fridman Podcast full episode AI",
            "All-In Podcast with Chamath and Jason",
            "Dwarkesh Podcast full episode",
            "a16z podcast AI technology episode",
            "Y Combinator startup founder podcast",
        ]
    },
    "health_longevity": {
        "title": "🧬 Health, Longevity & Biohacking",
        "estimated_cpm": "$20 - $40+ CPM / RPM",
        "rationale": "Supplements, health tech wearables (WHOOP/Oura), biohacking products, and fitness gear invest massive sponsorship budgets.",
        "channel_urls": [
            "https://www.youtube.com/@hubermanlab/videos",
            "https://www.youtube.com/@PeterAttiaMD/videos",
            "https://www.youtube.com/@FoundMyFitness/videos",
            "https://www.youtube.com/@TheModelHealthShow/videos",
            "https://www.youtube.com/@ThomasDeLauerOfficial/videos",
            "https://www.youtube.com/@MarkHymanMD/videos",
            "https://www.youtube.com/@drmikedoesnews/videos",
            "https://www.youtube.com/@DrBergKetoCourse/videos",
            "https://www.youtube.com/@ZoeOfficial/videos",
        ],
        "search_queries": [
            "Huberman Lab podcast full episode",
            "The Peter Attia Drive podcast",
            "Mark Hyman MD podcast full episode",
            "FoundMyFitness Dr Rhonda Patrick",
            "Dr Mike health science podcast episode",
        ]
    },
    "real_estate": {
        "title": "🏢 Real Estate Investing & Wealth",
        "estimated_cpm": "$30 - $60+ CPM / RPM",
        "rationale": "Mortgage lenders, prop-tech firms, title companies, and real estate masterminds have enormous customer acquisition budgets.",
        "channel_urls": [
            "https://www.youtube.com/@BiggerPockets/videos",
            "https://www.youtube.com/@MeetKevin/videos",
            "https://www.youtube.com/@PaceMorby/videos",
            "https://www.youtube.com/@GrahamStephan/videos",
            "https://www.youtube.com/@PropertyHub/videos",
            "https://www.youtube.com/@SamuelLeeds/videos",
        ],
        "search_queries": [
            "BiggerPockets Real Estate Podcast full episode",
            "Meet Kevin podcast investing real estate",
            "Pace Morby real estate podcast episode",
            "Property Hub UK podcast episode",
            "Australian property investing podcast",
        ]
    },
    "mindset": {
        "title": "Mindset, Motivation & Self-Improvement",
        "estimated_cpm": "$20 - $45+ CPM / RPM",
        "rationale": "Personal development apps, coaching platforms, and premium wellness brands target growth-mindset audiences with huge budgets.",
        "channel_urls": [
            "https://www.youtube.com/@EdMylettShow/videos",
            "https://www.youtube.com/@TomBilyeu/videos",
            "https://www.youtube.com/@JayShettyPodcast/videos",
            "https://www.youtube.com/@mindsetmentorpodcast/videos",
            "https://www.youtube.com/@LewisHowes/videos",
            "https://www.youtube.com/@GaryVee/videos",
            "https://www.youtube.com/@melrobbins/videos",
            "https://www.youtube.com/@TonyRobbins/videos",
            "https://www.youtube.com/@AddictedToSuccess/videos",
        ],
        "search_queries": [
            "Ed Mylett Show podcast full episode",
            "Tom Bilyeu Impact Theory podcast episode",
            "Jay Shetty On Purpose podcast episode",
            "Lewis Howes School of Greatness podcast",
            "Mel Robbins podcast motivation episode",
        ]
    }
}

def extract_youtube_id(url_or_id: str) -> Optional[str]:
    """Extracts canonical 11-character YouTube video ID."""
    if not url_or_id:
        return None
    if len(url_or_id) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", url_or_id):
        return url_or_id
    match = re.search(r"(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", url_or_id)
    return match.group(1) if match else None

def resolve_handle_to_id(handle: str) -> Optional[str]:
    """
    Resolves a @handle (e.g. @hubermanlab) to a UC... channel ID.
    Tries lightweight direct HTTP search first (fast, avoids tab errors),
    then falls back to yt-dlp with quiet logging.
    """
    clean_handle = handle.lstrip('@')
    # Strategy 1: Direct fast HTTP inspection of channel page
    try:
        url = f"https://www.youtube.com/@{clean_handle}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            match = re.search(r'itemprop="channelId"\s+content="(UC[\w-]+)"', res.text)
            if not match:
                match = re.search(r'"channelId":"(UC[\w-]+)"', res.text)
            if not match:
                match = re.search(r'/channel/(UC[\w-]+)', res.text)
            if match:
                return match.group(1)
    except Exception:
        pass

    # Strategy 2: Fallback to yt-dlp
    import yt_dlp
    url = f"https://www.youtube.com/@{clean_handle}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "extractor_args": {"youtubetab": {"skip": ["authcheck"]}}
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("channel_id")
    except Exception:
        return None

def discover_via_rss(channel_id: str) -> List[Dict[str, Any]]:
    """
    Fetches latest videos from YouTube's official RSS feed for a specific channel.
    Low bot-detection risk.
    """
    url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
    try:
        res = requests.get(url, timeout=15)
        if res.status_code != 200:
            return []
        
        root = ET.fromstring(res.content)
        videos = []
        # YouTube RSS uses namespaces
        ns = {'yt': 'http://www.youtube.com/xml/schemas/2015', 'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('atom:entry', ns):
            title_node = entry.find('atom:title', ns)
            link_node = entry.find('atom:link', ns)
            if title_node is None or link_node is None:
                continue
            title = title_node.text
            v_url = link_node.attrib.get("href")
            if not v_url:
                continue
            vid = extract_youtube_id(v_url)
            if vid:
                videos.append({
                    "title": title,
                    "url": v_url,
                    "id": vid
                })
        return videos
    except Exception as e:
        print(f"[-] RSS error for {channel_id}: {e}")
        return []

def load_history(history_file_path: Optional[Path] = None) -> Dict[str, Any]:
    """Loads JSON history of processed video IDs."""
    from src.config import HISTORY_FILE
    target = history_file_path or HISTORY_FILE
    if target.exists():
        try:
            with open(target, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

def record_history(video_url: str, title: str, niche: str, history_file_path: Optional[Path] = None) -> None:
    """Records a processed video to JSON history file."""
    import datetime
    from src.config import HISTORY_FILE
    target = history_file_path or HISTORY_FILE
    vid = extract_youtube_id(video_url)
    if not vid:
        return
    history = load_history(target)
    history[vid] = {
        "title": title,
        "url": f"https://www.youtube.com/watch?v={vid}",
        "niche": niche,
        "processed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_name(f"{target.name}.tmp")
        try:
            with open(temp_target, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_target, target)
        finally:
            if temp_target.exists():
                try:
                    temp_target.unlink()
                except OSError:
                    pass
        print(f"[+] Recorded '{vid}' to persistent history: {target}")
    except Exception as e:
        print(f"[-] Failed to update history: {e}")

DAY_OF_WEEK_NICHES = ["finance"] * 7

def resolve_daily_niche(niche: Optional[str] = None) -> str:
    if niche and niche in HIGH_CPM_NICHES and niche != "auto":
        return niche
    return "finance"

def classify_candidate_entry(entry: Dict[str, Any], processed_ids: set) -> Optional[tuple]:
    if not entry:
        return None
    vid = entry.get("id")
    vurl = entry.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else None)
    if not vurl:
        return None
    # Discovery is for long-form source episodes, never already-short content.
    # Flat yt-dlp search entries often omit duration, so URL shape is the only
    # reliable early signal for rejecting Shorts without an expensive probe.
    if "/shorts/" in vurl.lower():
        return None
    canon_id = extract_youtube_id(vurl or vid or "")
    if not canon_id or canon_id in processed_ids:
        return None
    duration = entry.get("duration")
    if duration is not None and 0 < duration <= 600:
        return None
    title = entry.get("title", "Unknown Title")
    is_confirmed_long = bool(duration and duration > 600)
    return (vurl, title, canon_id, is_confirmed_long)

def filter_fresh_candidates(
    ydl_opts: Dict[str, Any],
    queries: List[str],
    processed_ids: set,
    per_query: int = 6,
    channel_urls: Optional[List[str]] = None,
    cookie_path: Optional[Path] = None,
) -> List[tuple]:
    import yt_dlp
    seen: set = set()
    scored: List[tuple] = []

    current_opts = dict(ydl_opts)
    if cookie_path:
        current_opts['cookiefile'] = str(cookie_path)

    # 1. RSS-Based Discovery (Highest Priority)
    if channel_urls:
        for ch_url in channel_urls:
            # Extract handle from url
            handle_match = re.search(r"@([^\/\s?#]+)", ch_url)
            if handle_match:
                handle = handle_match.group(1)
                ch_id = resolve_handle_to_id(handle)
                if ch_id:
                    rss_videos = discover_via_rss(ch_id)
                    for v in rss_videos:
                        picked = classify_candidate_entry(v, processed_ids)
                        if picked:
                            vurl, title, cid, is_long = picked
                            if cid not in seen:
                                seen.add(cid)
                                scored.append((0 if is_long else 2, vurl, title, cid))

    # 2. Search queries fallback
    with yt_dlp.YoutubeDL(current_opts) as ydl:
        for q in queries:
            try:
                res = ydl.extract_info(f"ytsearch{per_query}:{q}", download=False)
            except Exception as e:
                print(f"[-] Search query error for '{q}': {e}")
                continue
            for entry in (res.get("entries") or []):
                picked = classify_candidate_entry(entry, processed_ids)
                if not picked:
                    continue
                vurl, title, cid, is_long = picked
                if cid in seen:
                    continue
                seen.add(cid)
                scored.append((0 if is_long else 3, vurl, title, cid))

    # 3. Direct /videos tab fallback (Low priority)
    if channel_urls:
        ch_opts = dict(current_opts)
        ch_opts["playlistend"] = 3
        ch_opts["extractor_args"] = {"youtubetab": {"skip": ["authcheck"]}}
        with yt_dlp.YoutubeDL(ch_opts) as ydl:
            for ch_url in channel_urls:
                try:
                    res = ydl.extract_info(ch_url, download=False)
                    for entry in (res.get("entries") or []):
                        picked = classify_candidate_entry(entry, processed_ids)
                        if picked:
                            vurl, title, cid, is_long = picked
                            if cid not in seen:
                                seen.add(cid)
                                scored.append((1 if is_long else 4, vurl, title, cid))
                except Exception:
                    continue

    scored.sort(key=lambda c: c[0])
    return [(v, t, i) for _, v, t, i in scored]

def get_daily_discovery_candidates(niche: Optional[str] = None, history_file: str = "history.txt") -> List[str]:
    import random
    from pathlib import Path
    from src.config import HISTORY_FILE

    history_data = load_history(HISTORY_FILE)
    processed_ids = set(history_data.keys())

    legacy_txt = Path(history_file)
    if legacy_txt.exists():
        try:
            for line in legacy_txt.read_text(encoding="utf-8").splitlines():
                lid = extract_youtube_id(line.strip())
                if lid:
                    processed_ids.add(lid)
        except Exception:
            pass

    selected_niche = resolve_daily_niche(niche)
    niche_data = HIGH_CPM_NICHES[selected_niche]
    queries: List[str] = list(niche_data["search_queries"])
    random.shuffle(queries)
    channel_urls: List[str] = list(niche_data.get("channel_urls", []))
    random.shuffle(channel_urls)

    print(f"[*] Auto-Discovering latest episodes in niche: {niche_data['title']} (Day rotation: {selected_niche})")
    
    ydl_opts = {"quiet": True, "extract_flat": True}
    cookie_path = None
    try:
        if YOUTUBE_COOKIES and YOUTUBE_COOKIES.strip():
            cookie_path = _write_cookiefile(YOUTUBE_COOKIES)
        candidates = filter_fresh_candidates(ydl_opts, queries, processed_ids, channel_urls=channel_urls, cookie_path=cookie_path)
    finally:
        if cookie_path and cookie_path.exists():
            try:
                cookie_path.unlink()
            except Exception:
                pass

    if candidates:
        print(f"[+] Found {len(candidates)} fresh high-CPM podcast candidates:")
        for idx, (vurl, vtitle, _) in enumerate(candidates[:3], 1):
            print(f"    #{idx}: '{vtitle}' ({vurl})")
        return [c[0] for c in candidates[:MAX_DISCOVERY_CANDIDATES]]

    fallback = "https://www.youtube.com/watch?v=UF8uR6Z6KLc"
    print(f"[*] Defaulting to verified episode: {fallback}")
    return [fallback]

def get_daily_discovery_episode(niche: Optional[str] = None, history_file: str = "history.txt") -> str:
    candidates = get_daily_discovery_candidates(niche=niche, history_file=history_file)
    return candidates[0] if candidates else "https://www.youtube.com/watch?v=UF8uR6Z6KLc"

def main():
    parser = argparse.ArgumentParser(description="High-CPM Podcast Discovery Tool")
    parser.add_argument("--niche", "-n", choices=list(HIGH_CPM_NICHES.keys()) + ["all"], default="finance")
    args = parser.parse_args()
    
    niche = args.niche if args.niche != "all" else resolve_daily_niche()
    print(f"[*] Testing discovery for niche: {niche}")
    url = get_daily_discovery_episode(niche=niche)
    print(f"[+] Result: {url}")

if __name__ == "__main__":
    main()
