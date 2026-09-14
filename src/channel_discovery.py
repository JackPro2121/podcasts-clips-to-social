import os
import sys
import argparse
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.config import APIFY_API_TOKEN

# Curated High CPM & RPM Podcast Niches with proven advertiser demand
# Channels verified via DuckDuckGo + web research (US, UK, AU, CA top podcasters)
HIGH_CPM_NICHES = {
    "finance": {
        "title": "💰 Personal Finance, Investing & Wealth",
        "estimated_cpm": "$30 - $65+ CPM / RPM",
        "rationale": "Financial institutions, stock brokers, crypto platforms, and credit card companies bid the highest advertising dollars on social media.",
        "channel_urls": [
            # === US Top Finance Podcasters ===
            "https://www.youtube.com/@TheRamseyShow/videos",
            "https://www.youtube.com/@MyFirstMillionPod/videos",
            "https://www.youtube.com/@TheIcedCoffeeHour/videos",
            "https://www.youtube.com/@TheMoneyGuyShow/videos",
            "https://www.youtube.com/@GrahamStephan/videos",
            "https://www.youtube.com/@InvestorsPodcast/videos",
            "https://www.youtube.com/@YahooFinance/videos",
            "https://www.youtube.com/@PBDPodcast/videos",
            # === UK Finance Podcasters ===
            "https://www.youtube.com/@DamienTalksMoney/videos",
            "https://www.youtube.com/@Pensioncraft/videos",
            "https://www.youtube.com/@marktilbury/videos",
            # === Canada Finance Podcasters ===
            "https://www.youtube.com/@ThePlainBagel/videos",
            "https://www.youtube.com/@BenFelixCSI/videos",
            # === Australia Finance Podcasters ===
            "https://www.youtube.com/@RaskAustralia/videos",
        ],
        "search_queries": [
            "The Ramsey Show podcast episode",
            "My First Million podcast full episode",
            "The Iced Coffee Hour podcast episode",
            "The Money Guy Show full podcast",
            "BiggerPockets Money Podcast",
            "We Study Billionaires investor podcast",
            "Patrick Bet-David PBD podcast episode",
        ]
    },
    "business": {
        "title": "📈 Business, Startups & Entrepreneurship",
        "estimated_cpm": "$25 - $50+ CPM / RPM",
        "rationale": "High-value B2B SaaS, CRM tools (HubSpot/Salesforce), enterprise software, and payment processors advertise heavily here.",
        "channel_urls": [
            # === US/Global Business Podcasters ===
            "https://www.youtube.com/@TheDiaryOfACEO/videos",
            "https://www.youtube.com/@AlexHormozi/videos",
            "https://www.youtube.com/@VALUETAINMENT/videos",
            "https://www.youtube.com/@TimFerriss/videos",
            "https://www.youtube.com/@MastersofScale/videos",
            "https://www.youtube.com/@joerogan/videos",
            "https://www.youtube.com/@colinandsamir/videos",
            # === UK Business Podcasters ===
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
            # === US/Global AI & Tech Podcasters ===
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
            # === US Health & Longevity Podcasters ===
            "https://www.youtube.com/@hubermanlab/videos",
            "https://www.youtube.com/@PeterAttiaMD/videos",
            "https://www.youtube.com/@FoundMyFitness/videos",
            "https://www.youtube.com/@TheModelHealthShow/videos",
            "https://www.youtube.com/@ThomasDeLauerOfficial/videos",
            "https://www.youtube.com/@MarkHymanMD/videos",
            "https://www.youtube.com/@drmikedoesnews/videos",
            "https://www.youtube.com/@DrBergKetoCourse/videos",
            # === UK Health Podcasters ===
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
            # === US Real Estate Podcasters ===
            "https://www.youtube.com/@BiggerPockets/videos",
            "https://www.youtube.com/@MeetKevin/videos",
            "https://www.youtube.com/@PaceMorby/videos",
            "https://www.youtube.com/@GrahamStephan/videos",
            # === UK Real Estate Podcasters ===
            "https://www.youtube.com/@PropertyHub/videos",
            "https://www.youtube.com/@SamuelLeeds/videos",
            # === Australia Real Estate Podcasters ===
            "https://www.youtube.com/@PizzaAndProperty/videos",
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
            # === US Mindset & Motivation Podcasters ===
            "https://www.youtube.com/@EdMylettShow/videos",
            "https://www.youtube.com/@TomBilyeu/videos",
            "https://www.youtube.com/@JayShettyPodcast/videos",
            "https://www.youtube.com/@mindsetmentorpodcast/videos",
            "https://www.youtube.com/@LewisHowes/videos",
            "https://www.youtube.com/@GaryVee/videos",
            "https://www.youtube.com/@melrobbins/videos",
            "https://www.youtube.com/@TonyRobbins/videos",
            # === UK/AU Mindset Podcasters ===
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

def extract_high_cpm_channels(
    niche_key: str = "finance",
    max_results_per_query: int = 2,
    api_token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Uses Apify's YouTube Scraper actor to search and extract verified podcast channels
    and their latest active episodes in high-CPM niches.
    """
    token = api_token or APIFY_API_TOKEN
    if not token:
        print("[-] APIFY_API_TOKEN is required to scrape podcast channels.")
        return []

    if niche_key not in HIGH_CPM_NICHES:
        print(f"[-] Unknown niche '{niche_key}'. Available niches: {list(HIGH_CPM_NICHES.keys())}")
        return []

    niche_info = HIGH_CPM_NICHES[niche_key]
    queries = niche_info["search_queries"]

    print(f"[*] Discovering high-CPM channels for niche: {niche_info['title']} ({niche_info['estimated_cpm']})...")
    endpoint = "https://api.apify.com/v2/acts/streamers~youtube-scraper/runs"
    payload = {
        "searchQueries": queries,
        "maxResults": max_results_per_query
    }
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(endpoint, headers=headers, json=payload, timeout=30)
        if res.status_code not in (200, 201):
            print(f"[-] Failed to start Apify scraper: {res.status_code} - {res.text}")
            return []

        run_id = res.json().get("data", {}).get("id")
        print(f"[*] Apify scraping run started ({run_id}). Polling for channel discovery...")

        final_run_data = None
        for _ in range(30):
            time.sleep(3)
            status_res = requests.get(f"https://api.apify.com/v2/actor-runs/{run_id}", headers=headers, timeout=15)
            if status_res.status_code == 200:
                final_run_data = status_res.json().get("data", {})
                status = final_run_data.get("status")
                if status in ("SUCCEEDED", "FAILED", "TIMED-OUT"):
                    break

        if not final_run_data or final_run_data.get("status") != "SUCCEEDED":
            print("[-] Scraper did not succeed on Apify.")
            return []

        dataset_id = final_run_data.get("defaultDatasetId")
        items_res = requests.get(f"https://api.apify.com/v2/datasets/{dataset_id}/items", headers=headers, timeout=30)
        items = items_res.json()

        # Group and deduplicate by channel
        channels_dict: Dict[str, Dict[str, Any]] = {}
        for item in items:
            ch_name = item.get("channelName") or "Unknown Channel"
            ch_url = item.get("channelUrl") or ""
            video_title = item.get("title") or ""
            video_url = item.get("url") or ""
            view_count = item.get("viewCount", 0)

            if ch_name not in channels_dict:
                channels_dict[ch_name] = {
                    "channel_name": ch_name,
                    "channel_url": ch_url,
                    "niche": niche_info["title"],
                    "estimated_cpm": niche_info["estimated_cpm"],
                    "episodes": []
                }

            channels_dict[ch_name]["episodes"].append({
                "title": video_title,
                "url": video_url,
                "views": view_count
            })

        discovered = list(channels_dict.values())
        print(f"[+] Discovered {len(discovered)} top podcast channels with active audiences!")
        return discovered

    except Exception as e:
        print(f"[-] Exception during channel discovery: {e}")
        return []

def get_latest_high_cpm_podcast_url(
    niche: Optional[str] = None,
    history_file: str = "history.txt"
) -> str:
    """Alias for get_daily_discovery_episode for backwards compatibility."""
    return get_daily_discovery_episode(niche=niche, history_file=history_file)

def extract_youtube_id(url_or_id: str) -> Optional[str]:
    """Extracts canonical 11-character YouTube video ID."""
    if not url_or_id:
        return None
    if len(url_or_id) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", url_or_id):
        return url_or_id
    match = re.search(r"(?:v=|\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", url_or_id)
    return match.group(1) if match else None

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
        with open(target, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"[+] Recorded '{vid}' to persistent history: {target}")
    except Exception as e:
        print(f"[-] Failed to update history: {e}")

# Day-of-Week Smart Auto-Rotation for maximum CPM & audience engagement:
# Monday: Finance & Wealth | Tuesday: AI & Tech | Wednesday: Health & Longevity
# Thursday: Business & Startups | Friday: Real Estate | Saturday: Mindset | Sunday: Health
DAY_OF_WEEK_NICHES = [
    "finance",           # Monday
    "ai_tech",           # Tuesday
    "health_longevity",  # Wednesday
    "business",          # Thursday
    "real_estate",       # Friday
    "mindset",           # Saturday
    "health_longevity"   # Sunday
]

def resolve_daily_niche(niche: Optional[str] = None) -> str:
    """Resolves target niche from explicit argument or day-of-week rotation."""
    if niche and niche in HIGH_CPM_NICHES:
        return niche
    import datetime
    day_idx = datetime.datetime.now(datetime.timezone.utc).weekday()
    return DAY_OF_WEEK_NICHES[day_idx]

def classify_candidate_entry(
    entry: Dict[str, Any],
    processed_ids: set,
) -> Optional[tuple]:
    """Pure filter: return (url, title, canon_id, is_confirmed_long) for a fresh
    long-form candidate, else None.

    Flat search entries usually omit ``duration``. The old code required
    ``duration > 600`` and therefore discarded EVERY flat entry -- discovery was
    silently inert and always fell back to one hardcoded video. We now treat
    unknown duration as acceptable and reject only entries we *know* are short.
    """
    if not entry:
        return None
    vid = entry.get("id")
    vurl = entry.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else None)
    if not vurl:
        return None
    canon_id = extract_youtube_id(vurl or vid or "")
    if not canon_id or canon_id in processed_ids:
        return None
    duration = entry.get("duration")
    if duration is not None and 0 < duration <= 600:
        return None  # confirmed Short / clip, skip.
    title = entry.get("title", "Unknown Title")
    is_confirmed_long = bool(duration and duration > 600)
    return (vurl, title, canon_id, is_confirmed_long)


def filter_fresh_candidates(
    ydl_opts: Dict[str, Any],
    queries: List[str],
    processed_ids: set,
    per_query: int = 6,
    channel_urls: Optional[List[str]] = None,
) -> List[tuple]:
    """Run extraction across official channel /videos tabs first (guaranteed newest uploads)
    and then fallback to queries, returning de-duplicated fresh candidates."""
    import yt_dlp
    seen: set = set()
    scored: List[tuple] = []  # (rank, url, title, id)

    # 1. Official channel /videos endpoints (Top priority: fresh trending episodes)
    if channel_urls:
        ch_opts = dict(ydl_opts)
        ch_opts["playlistend"] = 3
        with yt_dlp.YoutubeDL(ch_opts) as ydl:
            for ch_url in channel_urls:
                try:
                    res = ydl.extract_info(ch_url, download=False)
                    for entry in (res.get("entries") or []):
                        picked = classify_candidate_entry(entry, processed_ids)
                        if not picked:
                            continue
                        vurl, title, cid, is_long = picked
                        if cid in seen:
                            continue
                        seen.add(cid)
                        # Rank 0: Latest official release from top channel
                        scored.append((0, vurl, title, cid))
                except Exception as e:
                    print(f"[-] Channel extraction error for '{ch_url}': {e}")
                    continue

    # 2. General search queries fallback
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
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
                scored.append((1 if is_long else 2, vurl, title, cid))

    scored.sort(key=lambda c: c[0])
    return [(v, t, i) for _, v, t, i in scored]


def get_daily_discovery_candidates(
    niche: Optional[str] = None,
    history_file: str = "history.txt"
) -> List[str]:
    """
    Discovers fresh high-CPM podcast episodes automatically and returns a list
    of candidates in priority order (fresh unclipped episodes).
    """
    import random
    import yt_dlp
    from pathlib import Path
    from src.config import HISTORY_FILE

    # Load persistent history (both JSON and legacy TXT)
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

    queries = niche_data["search_queries"].copy()
    random.shuffle(queries)

    channel_urls = niche_data.get("channel_urls", []).copy()
    random.shuffle(channel_urls)

    print(f"[*] Auto-Discovering latest episodes in niche: {niche_data['title']} (Day rotation: {selected_niche})")
    print(f"[*] Known processed videos count: {len(processed_ids)}")

    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
    }

    candidates = filter_fresh_candidates(
        ydl_opts=ydl_opts,
        queries=queries,
        processed_ids=processed_ids,
        channel_urls=channel_urls
    )

    if candidates:
        print(f"[+] Found {len(candidates)} fresh high-CPM podcast candidates:")
        for idx, (vurl, vtitle, _) in enumerate(candidates[:3], 1):
            print(f"    #{idx}: '{vtitle}' ({vurl})")
        return [c[0] for c in candidates]

    # Fallback to popular evergreen business episode if nothing found
    fallback = "https://www.youtube.com/watch?v=UF8uR6Z6KLc"
    print(f"[*] Defaulting to verified episode: {fallback}")
    return [fallback]

def get_daily_discovery_episode(
    niche: Optional[str] = None,
    history_file: str = "history.txt"
) -> str:
    """Discovers top high-CPM podcast episode URL automatically."""
    candidates = get_daily_discovery_candidates(niche=niche, history_file=history_file)
    return candidates[0] if candidates else "https://www.youtube.com/watch?v=UF8uR6Z6KLc"

def main():
    parser = argparse.ArgumentParser(
        description="High-CPM Podcast Channel Discovery Tool powered by Apify"
    )
    parser.add_argument(
        "--niche", "-n",
        choices=list(HIGH_CPM_NICHES.keys()) + ["all"],
        default="finance",
        help="Niche category to scrape for high CPM/RPM podcasts (default: finance)."
    )
    parser.add_argument(
        "--max-per-query", "-m",
        type=int,
        default=2,
        help="Max video/channel results per query (default: 2)."
    )
    parser.add_argument(
        "--output-json", "-o",
        help="Optional path to save discovery results as JSON."
    )

    args = parser.parse_args()

    niches_to_run = list(HIGH_CPM_NICHES.keys()) if args.niche == "all" else [args.niche]
    all_results = {}

    for n in niches_to_run:
        niche_data = HIGH_CPM_NICHES[n]
        print("\n" + "=" * 75)
        print(f"CATEGORY: {niche_data['title']}")
        print(f"ESTIMATED CPM/RPM: {niche_data['estimated_cpm']}")
        print(f"WHY IT PAYS HIGH: {niche_data['rationale']}")
        print("=" * 75)

        channels = extract_high_cpm_channels(n, max_results_per_query=args.max_per_query)
        all_results[n] = channels

        for idx, ch in enumerate(channels, 1):
            print(f"\n{idx}. 🎙️ {ch['channel_name']}")
            print(f"   🔗 Channel URL: {ch['channel_url']}")
            print(f"   🎬 Latest Podcast Episodes to Clip:")
            for ep in ch["episodes"][:2]:
                print(f"      - {ep['title']}")
                print(f"        👉 {ep['url']}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[+] Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
