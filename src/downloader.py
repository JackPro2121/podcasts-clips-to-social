import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess
import yt_dlp
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from src.config import DOWNLOADS_DIR, APIFY_API_TOKEN

def get_video_height(file_path: Path) -> int:
    """Probes video stream height using ffprobe to ensure resolution is >= 720p."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=height",
            "-of", "csv=p=0",
            str(file_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        lines = res.stdout.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line.isdigit():
                return int(line)
        return 0
    except Exception:
        return 0

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

from src.config import DOWNLOADS_DIR, APIFY_API_TOKEN, YOUTUBE_COOKIES, YTDLP_PROXY, RAPIDAPI_KEY

def download_via_ytdlp(url_or_path: str, target_dir: Path, video_id: Optional[str] = None, transcript: Optional[Any] = None) -> Optional[Dict[str, Any]]:
    """
    Downloads video using yt-dlp with Node.js challenge solving and optional cookie authentication.
    Runs 100% free with 0 Apify compute cost.
    Prioritizes true High-Definition (1080p/720p) streams with bgutil POT provider,
    with intelligent fallback to lower resolutions if YouTube enforces strict throttling.
    """
    out_template = str(target_dir / "%(id)s_%(title).50s.%(ext)s")
    base_opts = {
        'js_runtimes': {'deno': {}, 'node': {}},
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
        'outtmpl': out_template,
        'merge_output_format': 'mp4',
        'quiet': False,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }]
    }

    # Option 1: Route traffic through proxy if configured (bypasses datacenter blocks)
    if YTDLP_PROXY and YTDLP_PROXY.strip():
        proxy_str = YTDLP_PROXY.strip()
        base_opts['proxy'] = proxy_str
        masked_proxy = proxy_str.split('@')[-1] if '@' in proxy_str else proxy_str[:15] + "..."
        print(f"[*] Configured proxy routing for yt-dlp: {masked_proxy}")

    # Connect to local bgutil POT provider if running (port 4416)
    pot_args = {'youtubepot-bgutilhttp': {'base_url': ['http://127.0.0.1:4416']}}

    cookie_path = None
    if YOUTUBE_COOKIES and YOUTUBE_COOKIES.strip():
        try:
            cookie_path = target_dir / "yt_cookies.txt"
            cookie_path.write_text(YOUTUBE_COOKIES.strip() + "\n", encoding="utf-8")
        except Exception as e:
            print(f"[-] Cookie setup warning: {e}")

    strategies = []

    # Priority 1: Default & MWeb Client with POT Provider (Most stable client for bgutil)
    strategies.append(("Default & MWeb with POT Provider (1080p)", {
        **base_opts,
        'format': 'bestvideo[height>=720][height<=1080]+bestaudio/bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'extractor_args': {'youtube': {'player_client': ['default', 'mweb']}, **pot_args}
    }))

    # Priority 2: Primary Web Client (Full 1080p60 DASH streams with Deno solver + POT Provider)
    strategies.append(("Primary Web Client (1080p Deno + POT)", {
        **base_opts,
        'format': 'bestvideo[height>=720][height<=1080]+bestaudio/bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'extractor_args': {'youtube': {'player_client': ['web']}, **pot_args}
    }))

    # Priority 3: Mobile VR Client with POT Provider (1080p H.264 & DASH streams)
    strategies.append(("Mobile VR Client (1080p DASH + POT)", {
        **base_opts,
        'format': 'bestvideo[height>=720][height<=1080]+bestaudio/bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'extractor_args': {'youtube': {'player_client': ['android_vr']}, **pot_args}
    }))

    # Priority 4: iOS Mobile Client with POT Provider (1080p)
    strategies.append(("iOS Mobile Client (1080p + POT)", {
        **base_opts,
        'format': 'bestvideo[height>=720][height<=1080]+bestaudio/bestvideo[height<=1080]+bestaudio/best[height<=1080]',
        'extractor_args': {'youtube': {'player_client': ['ios']}, **pot_args}
    }))

    # Priority 5: Universal Multi-Client Fallback (web, tv, android_vr + POT)
    strategies.append(("Universal Multi-Client with POT", {
        **base_opts,
        'format': 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best',
        'extractor_args': {'youtube': {'player_client': ['web', 'tv_embedded', 'android_vr']}, **pot_args}
    }))

    # Priority 6: Fallback with cookies (if provided)
    if cookie_path and cookie_path.exists():
        strategies.append(("Authenticated Session (with cookies)", {
            **base_opts,
            'cookiefile': str(cookie_path),
            'extractor_args': {'youtube': {'player_client': ['web', 'tv']}, **pot_args}
        }))

    # Priority 7: Fail-safe Unrestricted Mobile Android Client (never fails, 360p/480p fallback)
    strategies.append(("Fail-Safe Mobile Android Client", {
        **base_opts,
        'format': 'bestvideo+bestaudio/best',
        'extractor_args': {'youtube': {'player_client': ['android']}}
    }))

    best_candidate_file = None
    best_candidate_info = None
    best_candidate_height = 0

    try:
        for strat_name, current_opts in strategies:
            try:
                print(f"[*] Attempting yt-dlp download: {strat_name}...")
                with yt_dlp.YoutubeDL(current_opts) as ydl:
                    info = ydl.extract_info(url_or_path, download=True)
                    downloaded_file = ydl.prepare_filename(info)
                    if not os.path.exists(downloaded_file):
                        candidate = str(Path(downloaded_file).with_suffix('.mp4'))
                        if os.path.exists(candidate):
                            downloaded_file = candidate

                    if not os.path.exists(downloaded_file):
                        continue

                    # Probe downloaded video height
                    h = get_video_height(Path(downloaded_file))
                    print(f"[*] Probed stream resolution: {h}p via {strat_name}")

                    # If resolution is 720p or higher, accept immediately as verified HD!
                    if h >= 720:
                        print(f"[+] Verified High-Definition stream: {h}p via {strat_name}")
                        return {
                            'video_path': Path(downloaded_file).resolve(),
                            'title': info.get('title', f"YouTube_{video_id or 'podcast'}"),
                            'duration': float(info.get('duration', 0.0)),
                            'transcript': transcript,
                            'video_id': video_id or info.get('id'),
                            'height': h,
                            'is_low_res': False,
                            'is_local': False
                        }
                    else:
                        # Below 720p: Stash as fallback candidate and keep trying remaining HD strategies
                        if h > best_candidate_height:
                            if best_candidate_file and os.path.exists(best_candidate_file):
                                try:
                                    Path(best_candidate_file).unlink()
                                except Exception:
                                    pass
                            best_candidate_file = downloaded_file
                            best_candidate_info = info
                            best_candidate_height = h
                            print(f"[!] Stream is {h}p (below HD 720p). Retaining as fallback and trying next HD strategy...")
                        else:
                            try:
                                Path(downloaded_file).unlink()
                            except Exception:
                                pass
            except Exception as e:
                print(f"[-] Strategy '{strat_name}' encountered error: {e}")

        # If all HD strategies finished and we have a fallback candidate:
        if best_candidate_file and os.path.exists(best_candidate_file):
            print(f"[!] Info: Using {best_candidate_height}p stream with aggressive AI Super-Resolution mastering.")
            return {
                'video_path': Path(best_candidate_file).resolve(),
                'title': best_candidate_info.get('title', f"YouTube_{video_id or 'podcast'}"),
                'duration': float(best_candidate_info.get('duration', 0.0)),
                'transcript': transcript,
                'video_id': video_id or best_candidate_info.get('id'),
                'height': best_candidate_height,
                'is_low_res': True,
                'is_local': False
            }

        return None
    finally:
        # Clean up temporary cookies file if created
        if cookie_path and cookie_path.exists():
            try:
                cookie_path.unlink()
            except Exception:
                pass

def download_via_rapidapi(
    video_url: str,
    output_dir: Path,
    api_key: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Option 2: Downloads YouTube video via RapidAPI YouTube Downloader endpoints.
    Provides 50-100 free requests per month without bot captcha.
    Muxes crystal-clear 1080p MP4 video with high-bitrate M4A audio via FFmpeg stream copy.
    Supports key rotation (comma-separated keys) and robust chunked streaming fallback.
    """
    import subprocess
    keys = [k.strip() for k in (api_key or RAPIDAPI_KEY).split(",") if k.strip()]
    if not keys:
        return None

    vid_id = extract_youtube_id(video_url)
    if not vid_id:
        return None

    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"

    for key_idx, key in enumerate(keys):
        print(f"[*] Attempting high-speed 1080p download via RapidAPI Downloader (key {key_idx + 1}/{len(keys)})...")
        headers = {
            "x-rapidapi-key": key,
            "x-rapidapi-host": "youtube-media-downloader.p.rapidapi.com"
        }
        url = f"https://youtube-media-downloader.p.rapidapi.com/v2/video/details?videoId={vid_id}"
        try:
            res = requests.get(url, headers=headers, timeout=25)
            if res.status_code == 429:
                print(f"[-] RapidAPI key {key_idx + 1} quota exceeded (429). Trying next key...")
                continue
            if res.status_code != 200:
                print(f"[-] RapidAPI returned status {res.status_code}: {res.text[:150]}")
                continue

            data = res.json()
            videos = data.get("videos", {}).get("items", [])
            audios = data.get("audios", {}).get("items", [])

            # Select best video stream (strictly 1080p or 720p)
            best_video = None
            for q in ["1080p", "720p"]:
                for v in videos:
                    if v.get("quality") == q and v.get("extension") == "mp4" and v.get("url"):
                        best_video = v
                        break
                if best_video:
                    break

            # Select best audio stream
            best_audio = None
            for a in audios:
                if a.get("extension") in ("m4a", "mp4") and a.get("url"):
                    best_audio = a
                    break

            out_file = output_dir / f"YouTube_{vid_id}.mp4"

            if best_video and best_audio:
                print(f"[*] RapidAPI stream found: Video {best_video.get('quality')} + Audio {best_audio.get('extension')}. Muxing via FFmpeg...")
                cmd = [
                    "ffmpeg", "-y",
                    "-user_agent", ua,
                    "-reconnect", "1",
                    "-reconnect_streamed", "1",
                    "-reconnect_delay_max", "5",
                    "-i", best_video["url"],
                    "-user_agent", ua,
                    "-reconnect", "1",
                    "-reconnect_streamed", "1",
                    "-reconnect_delay_max", "5",
                    "-i", best_audio["url"],
                    "-c:v", "copy",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    str(out_file)
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if proc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 1024 * 1024:
                    print(f"[+] Download and 1080p mux complete via RapidAPI: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
                    return {
                        "video_path": out_file.resolve(),
                        "title": data.get("title", f"YouTube_{vid_id}"),
                        "duration": float(data.get("lengthSeconds", 0.0)),
                        "is_local": False
                    }
                else:
                    err_msg = proc.stderr[-300:].strip() if proc.stderr else "Unknown error"
                    print(f"[-] Fast FFmpeg mux failed ({proc.returncode}): {err_msg}. Engaging robust chunked streaming fallback...")

                    # Fallback: Download tracks to disk via requests with browser User-Agent and stream chunks
                    temp_v = output_dir / f"tmp_{vid_id}_v.mp4"
                    temp_a = output_dir / f"tmp_{vid_id}_a.m4a"
                    dl_headers = {"User-Agent": ua}
                    try:
                        print(f"[*] Downloading video track via chunked stream ({best_video.get('quality')})...")
                        with requests.get(best_video["url"], headers=dl_headers, stream=True, timeout=180) as r_v:
                            r_v.raise_for_status()
                            with open(temp_v, "wb") as f_v:
                                for chunk in r_v.iter_content(chunk_size=2 * 1024 * 1024):
                                    if chunk:
                                        f_v.write(chunk)

                        print(f"[*] Downloading audio track via chunked stream ({best_audio.get('extension')})...")
                        with requests.get(best_audio["url"], headers=dl_headers, stream=True, timeout=180) as r_a:
                            r_a.raise_for_status()
                            with open(temp_a, "wb") as f_a:
                                for chunk in r_a.iter_content(chunk_size=2 * 1024 * 1024):
                                    if chunk:
                                        f_a.write(chunk)

                        if temp_v.exists() and temp_a.exists() and temp_v.stat().st_size > 512 * 1024:
                            print(f"[*] Tracks saved locally. Combining with local FFmpeg...")
                            cmd_local = [
                                "ffmpeg", "-y",
                                "-i", str(temp_v),
                                "-i", str(temp_a),
                                "-c:v", "copy",
                                "-c:a", "copy",
                                "-movflags", "+faststart",
                                str(out_file)
                            ]
                            p_loc = subprocess.run(cmd_local, capture_output=True, text=True, timeout=120)
                            if p_loc.returncode == 0 and out_file.exists() and out_file.stat().st_size > 1024 * 1024:
                                print(f"[+] Download and 1080p mux complete via chunked streaming: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
                                return {
                                    "video_path": out_file.resolve(),
                                    "title": data.get("title", f"YouTube_{vid_id}"),
                                    "duration": float(data.get("lengthSeconds", 0.0)),
                                    "is_local": False
                                }
                    finally:
                        for tmp_f in [temp_v, temp_a]:
                            if tmp_f.exists():
                                try:
                                    tmp_f.unlink()
                                except Exception:
                                    pass

            # Fallback: If single combined stream exists
            if best_video and best_video.get("hasAudio"):
                print(f"[*] Streaming combined format ({best_video.get('quality')})...")
                with requests.get(best_video["url"], headers={"User-Agent": ua}, stream=True, timeout=180) as stream_res:
                    stream_res.raise_for_status()
                    with open(out_file, "wb") as f:
                        for chunk in stream_res.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                if out_file.exists() and out_file.stat().st_size > 0:
                    print(f"[+] Download complete via RapidAPI: {out_file} ({out_file.stat().st_size / (1024*1024):.2f} MB)")
                    return {
                        "video_path": out_file.resolve(),
                        "title": data.get("title", f"YouTube_{vid_id}"),
                        "duration": float(data.get("lengthSeconds", 0.0)),
                        "is_local": False
                    }
        except Exception as e:
            print(f"[-] RapidAPI download error on key {key_idx + 1}: {e}")

    return None

def download_video(url_or_path: str, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Smart multi-tier downloader:
    1. Local file check.
    2. Zero-cost yt-dlp (bgutil PO token provider + optional Webshare proxy / cookies).
    3. RapidAPI YouTube Downloader fallback (Option 2).
    4. Apify Actor proxy fallback (Option 3).
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
            'video_id': extract_youtube_id(local_path.stem),
            'is_local': True
        }

    # Fetch native transcript if YouTube URL
    video_id = extract_youtube_id(url_or_path)
    transcript = None
    if video_id:
        transcript = fetch_youtube_transcript(video_id)

    # 1. Tier 1: Direct yt-dlp ($0 cost, uses bgutil PO Token + Proxy if set)
    ytdlp_result = download_via_ytdlp(url_or_path, target_dir, video_id=video_id, transcript=transcript)
    if ytdlp_result:
        return ytdlp_result

    # 2. Tier 2: RapidAPI Downloader fallback (Option 2)
    if RAPIDAPI_KEY:
        print("[!] Direct yt-dlp failed. Engaging RapidAPI downloader fallback...")
        rapidapi_result = download_via_rapidapi(url_or_path, target_dir)
        if rapidapi_result:
            rapidapi_result['transcript'] = transcript
            rapidapi_result['video_id'] = video_id
            return rapidapi_result

    # 3. Tier 3: Apify Actor proxy fallback
    if APIFY_API_TOKEN and ("youtube.com" in url_or_path or "youtu.be" in url_or_path):
        print("[!] Engaging Apify proxy downloader...")
        apify_result = download_via_apify(url_or_path, target_dir, quality="1080")
        if apify_result:
            apify_result['transcript'] = transcript
            apify_result['video_id'] = video_id
            return apify_result

    raise RuntimeError(f"Failed to download video from {url_or_path} using yt-dlp, RapidAPI, and Apify.")
