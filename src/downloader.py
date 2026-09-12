import os
import re
import time
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import subprocess
import yt_dlp
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from src.config import (
    DOWNLOADS_DIR, APIFY_API_TOKEN, YOUTUBE_COOKIES, YTDLP_PROXY, RAPIDAPI_KEY,
    COBALT_API_URL, COBALT_INSTANCES, MIN_VIDEO_HEIGHT, MAX_VIDEO_HEIGHT,
)

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------
# The old downloader treated every failure identically ("try the next strategy"),
# which meant a permanently-dead video (private/removed/geo-blocked) burned the
# full 7-strategy x full-download budget before giving up. We now distinguish
# permanent failures (abort immediately) from transient ones (rotate client).

class VideoUnavailableError(Exception):
    """Raised when a video is permanently unfetchable (private/removed/geo/age).

    Signals the orchestrator to STOP: no client rotation or paid tier will help.
    """


# Phrases that mean "this video will never download, stop trying".
_PERMANENT_MARKERS = (
    "video unavailable",
    "private video",
    "this video is private",
    "has been removed",
    "account associated with this video has been terminated",
    "video has been removed",
    "who has blocked it",
    "not available in your country",
    "not made this video available",
    "members-only",
    "this live event",
    "premieres in",
    "video is not available",
    "content is not available",
)

# Phrases that mean "this specific client got blocked/throttled, rotate".
_TRANSIENT_MARKERS = (
    "sign in to confirm",
    "confirm you're not a bot",
    "http error 429",
    "too many requests",
    "unable to download webpage",
    "read timed out",
    "connection reset",
    "temporarily unavailable",
    "failed to extract",
    "requested format is not available",
    "no video formats found",
    "unable to extract",
    "precondition check failed",
)


def classify_ytdlp_error(exc: Exception) -> str:
    """Return one of: 'permanent' | 'transient' | 'unknown' for a yt-dlp error."""
    msg = str(exc).lower()
    for marker in _PERMANENT_MARKERS:
        if marker in msg:
            return "permanent"
    for marker in _TRANSIENT_MARKERS:
        if marker in msg:
            return "transient"
    return "unknown"


def best_available_height(info: Dict[str, Any]) -> int:
    """Given a yt-dlp info dict (from a metadata-only probe), return the tallest
    downloadable video height on offer. 0 if none / unknown."""
    best = 0
    for f in (info.get("formats") or []):
        if f.get("vcodec") in (None, "none"):
            continue
        h = f.get("height") or 0
        try:
            h = int(h)
        except (TypeError, ValueError):
            h = 0
        if h > best:
            best = h
    # Some extractors expose a top-level height instead of per-format.
    top = info.get("height") or 0
    try:
        top = int(top)
    except (TypeError, ValueError):
        top = 0
    return max(best, top)


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

def get_video_duration(file_path: Path) -> float:
    """Probes video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(file_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        val = res.stdout.strip()
        return float(val) if val else 0.0
    except Exception:
        return 0.0

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

        # SECURITY: only attach the Apify bearer token when the download URL is
        # actually on Apify's domain. The URL comes from the actor's dataset and
        # could point anywhere (redirect/SSRF); we must not leak credentials.
        stream_headers = {}
        if "apify.com" in direct_url or "apify.dev" in direct_url:
            stream_headers = {"Authorization": f"Bearer {token}"}

        print(f"[*] Streaming high-quality video from Apify storage ({out_file.name})...")
        with requests.get(direct_url, headers=stream_headers, stream=True, timeout=180) as stream_res:
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

# Deduplicated format selectors (were copy-pasted across 7 strategies).
_HD_FORMAT = ('bestvideo[height>=720][height<=1080]+bestaudio/'
              'bestvideo[height<=1080]+bestaudio/best[height<=1080]')
_ANY_FORMAT = 'bestvideo[height<=1080]+bestaudio/best[height<=1080]/best'
_FAILSAFE_FORMAT = 'bestvideo+bestaudio/best'
# Local bgutil POT provider (started as a sidecar in the CI workflow).
_POT_ARGS = {'youtubepot-bgutilhttp': {'base_url': ['http://127.0.0.1:4416']}}


def _ytdlp_client_variants(cookie_path: Optional[Path] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """Generates ordered yt-dlp client extraction options prioritizing bot-bypassing clients."""
    def _client(clients: List[str]) -> Dict[str, Any]:
        return {'extractor_args': {'youtube': {'player_client': clients}, **_POT_ARGS}}

    variants: List[Tuple[str, Dict[str, Any]]] = [
        ("android_vr", {'extractor_args': {'youtube': {'player_client': ['android_vr']}}}),
        ("tv_embedded", {'extractor_args': {'youtube': {'player_client': ['tv_embedded', 'tv']}}}),
        ("web-pot", _client(['web'])),
        ("default+mweb", _client(['default', 'mweb'])),
        ("ios", {'extractor_args': {'youtube': {'player_client': ['ios']}}}),
        ("universal", _client(['android_vr', 'web', 'tv_embedded'])),
    ]
    if cookie_path:
        cookie_extra = _client(['web', 'tv'])
        cookie_extra['cookiefile'] = str(cookie_path)
        variants.append(("cookies", cookie_extra))
    # Fail-safe: android client without POT (no HD guarantee, but usually ungated).
    variants.append(("android-failsafe", {'extractor_args': {'youtube': {'player_client': ['android']}}}))
    return variants


def _write_cookiefile(cookies: str) -> Optional[Path]:
    """Write cookies to a private (0600) temp file OUTSIDE the artifact dir.

    The old code wrote cookies into the downloads/ folder, which is uploaded as a
    CI artifact -- leaking the session. mkstemp + chmod keeps them off the artifact
    path and unreadable to other users; the caller unlinks it in a finally block.
    """
    try:
        fd, tmp = tempfile.mkstemp(prefix="ytc_", suffix=".txt")
        os.write(fd, (cookies.strip() + "\n").encode("utf-8"))
        os.close(fd)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass  # Windows: best-effort.
        return Path(tmp)
    except Exception as e:
        print(f"[-] Cookie setup warning: {e}")
        return None


def _do_download(
    url: str,
    base_opts: Dict[str, Any],
    extra: Dict[str, Any],
    fmt: str,
    probed_info: Optional[Dict[str, Any]],
    video_id: Optional[str],
    transcript: Optional[Any],
    label: str,
    low_res: bool = False,
) -> Optional[Dict[str, Any]]:
    """Perform exactly one real download with a chosen client + format."""
    opts = {
        **base_opts,
        'format': fmt,
        'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}],
        **extra,
    }
    try:
        print(f"[*] Downloading via '{label}'...")
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            path = ydl.prepare_filename(info)
            if not os.path.exists(path):
                alt = str(Path(path).with_suffix('.mp4'))
                path = alt if os.path.exists(alt) else path
            if not os.path.exists(path):
                print(f"[-] '{label}' reported success but produced no file.")
                return None
        # Probe the real file. h==0 means the probe failed (missing ffprobe /
        # odd container) -- NOT that the video is low-res; fall back to the
        # resolution the metadata probe advertised instead of mislabelling HD.
        h = get_video_height(Path(path))
        real_h = h if h > 0 else (best_available_height(probed_info) if probed_info else 0)
        print(f"[+] Downloaded {real_h or '?'}p via '{label}': {Path(path).name}")
        return {
            'video_path': Path(path).resolve(),
            'title': info.get('title', f"YouTube_{video_id or 'podcast'}"),
            'duration': float(info.get('duration', 0.0) or 0.0),
            'transcript': transcript,
            'video_id': video_id or info.get('id'),
            'height': real_h,
            'is_low_res': low_res or (0 < real_h < 720),
            'is_local': False,
        }
    except VideoUnavailableError:
        raise
    except Exception as e:
        if classify_ytdlp_error(e) == "permanent":
            raise VideoUnavailableError(str(e))
        print(f"[-] Download via '{label}' failed: {e}")
        return None


def download_via_ytdlp(
    url_or_path: str,
    target_dir: Path,
    video_id: Optional[str] = None,
    transcript: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Probe-first yt-dlp downloader ($0, no Apify compute).

    The previous implementation ran up to SEVEN full downloads of the same
    2-hour video, probing resolution only AFTER each download -- catastrophic on
    a timeboxed/metered CI runner (disk blow-up + 90-min timeout).

    This version inverts the loop:
      1. For each client, run a METADATA-ONLY probe (``download=False`` -- seconds,
         zero video bytes) and classify failures. A *permanent* failure
         (private/removed/geo) raises ``VideoUnavailableError`` and aborts; a
         *transient* block rotates to the next client.
      2. The first client advertising a >=720p stream is downloaded ONCE.
      3. If nobody offers HD, the best sub-HD client is downloaded ONCE.
    Net: one download in the happy path, one in the worst case -- never seven.
    """
    base_opts = {
        'js_runtimes': {'deno': {}, 'node': {}},
        'outtmpl': str(target_dir / "%(id)s_%(title).50s.%(ext)s"),
        'merge_output_format': 'mp4',
        'quiet': True,
        'no_warnings': True,
    }
    if YTDLP_PROXY and YTDLP_PROXY.strip():
        base_opts['proxy'] = YTDLP_PROXY.strip()
        print("[*] Routing yt-dlp through configured proxy.")

    cookie_path = (
        _write_cookiefile(YOUTUBE_COOKIES)
        if (YOUTUBE_COOKIES and YOUTUBE_COOKIES.strip())
        else None
    )

    clients = _ytdlp_client_variants(cookie_path)

    try:
        best_sub_hd: Optional[Tuple[str, Dict[str, Any], int]] = None
        for label, extra in clients:
            probe_opts = {**base_opts, 'skip_download': True, **extra}
            try:
                print(f"[*] Probing client '{label}' (metadata only)...")
                with yt_dlp.YoutubeDL(probe_opts) as ydl:
                    info = ydl.extract_info(url_or_path, download=False)
            except VideoUnavailableError:
                raise
            except Exception as e:
                kind = classify_ytdlp_error(e)
                if kind == "permanent":
                    raise VideoUnavailableError(str(e))
                print(f"[-] Client '{label}' probe failed ({kind}): {e}")
                continue

            height = best_available_height(info)
            print(f"[*] Client '{label}' offers up to {height}p.")
            is_failsafe = label == "android-failsafe"
            if height >= 720:
                fmt = _FAILSAFE_FORMAT if is_failsafe else _HD_FORMAT
                result = _do_download(url_or_path, base_opts, extra, fmt, info,
                                      video_id, transcript, label)
                if result:
                    return result
                continue  # HD probe but download died -> next client.
            elif height > 0 and (best_sub_hd is None or height > best_sub_hd[2]):
                best_sub_hd = (label, extra, height)

        # No HD anywhere: take the tallest sub-HD client with a single download.
        if best_sub_hd:
            label, extra, height = best_sub_hd
            print(f"[!] No HD stream available; downloading best {height}p via '{label}'.")
            fmt = _FAILSAFE_FORMAT if label == "android-failsafe" else _ANY_FORMAT
            return _do_download(url_or_path, base_opts, extra, fmt, None,
                                video_id, transcript, label, low_res=True)

        return None
    finally:
        if cookie_path and cookie_path.exists():
            try:
                cookie_path.unlink()
            except Exception:
                pass

def download_via_cobalt(
    video_url: str,
    output_dir: Path,
    quality: str = "1080",
    instances: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Tier 2 ($0, No Cookies): Downloads YouTube video via Cobalt Engine API.
    Cobalt is an open-source media ingestion service designed specifically to bypass
    YouTube's BotGuard, SABR, and PoToken constraints without requiring user cookies.
    Supports multi-instance fallback, tunnel/redirect streams, and chunked streaming.
    """
    candidate_instances = instances or COBALT_INSTANCES or [COBALT_API_URL]
    vid_id = extract_youtube_id(video_url) or "video"
    out_file = output_dir / f"YouTube_{vid_id}.mp4"

    ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": ua
    }
    payload = {
        "url": video_url,
        "videoQuality": str(quality),
        "youtubeVideoCodec": "h264",
        "downloadMode": "auto"
    }

    for inst_idx, base_url in enumerate(candidate_instances):
        inst_url = base_url.rstrip("/")
        target_endpoint = inst_url if inst_url.endswith("/api") else f"{inst_url}/"
        print(f"[*] Attempting 1080p download via Cobalt instance {inst_idx + 1}/{len(candidate_instances)} ({inst_url})...")
        try:
            res = requests.post(target_endpoint, headers=headers, json=payload, timeout=25)
            if res.status_code not in (200, 201):
                print(f"[-] Cobalt instance {inst_url} returned status {res.status_code}: {res.text[:120]}")
                continue

            data = res.json()
            status = data.get("status")
            stream_url = data.get("url")

            # Direct stream, tunnel, or redirect URL
            if status in ("tunnel", "redirect", "stream") and stream_url:
                print(f"[*] Cobalt returned '{status}' stream. Downloading to disk ({out_file.name})...")
                with requests.get(stream_url, headers={"User-Agent": ua}, stream=True, timeout=180) as stream_res:
                    stream_res.raise_for_status()
                    with open(out_file, "wb") as f:
                        for chunk in stream_res.iter_content(chunk_size=2 * 1024 * 1024):
                            if chunk:
                                f.write(chunk)

                if out_file.exists() and out_file.stat().st_size > 1024 * 1024:
                    h = get_video_height(out_file)
                    dur = get_video_duration(out_file)
                    print(f"[+] Cobalt download complete: {out_file.name} ({h}p, {out_file.stat().st_size / (1024*1024):.2f} MB)")
                    return {
                        "video_path": out_file.resolve(),
                        "title": f"YouTube_{vid_id}",
                        "duration": dur,
                        "height": h,
                        "is_low_res": (0 < h < MIN_VIDEO_HEIGHT),
                        "is_local": False
                    }
                else:
                    print(f"[-] Cobalt stream produced empty or truncated file.")
            elif status == "picker":
                picker_items = data.get("picker", [])
                best_url = None
                for item in picker_items:
                    if item.get("type") in ("video", None) and item.get("url"):
                        best_url = item["url"]
                        break
                if best_url:
                    print(f"[*] Cobalt returned picker items. Downloading stream to disk...")
                    with requests.get(best_url, headers={"User-Agent": ua}, stream=True, timeout=180) as stream_res:
                        stream_res.raise_for_status()
                        with open(out_file, "wb") as f:
                            for chunk in stream_res.iter_content(chunk_size=2 * 1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    if out_file.exists() and out_file.stat().st_size > 1024 * 1024:
                        h = get_video_height(out_file)
                        dur = get_video_duration(out_file)
                        return {
                            "video_path": out_file.resolve(),
                            "title": f"YouTube_{vid_id}",
                            "duration": dur,
                            "height": h,
                            "is_low_res": (0 < h < MIN_VIDEO_HEIGHT),
                            "is_local": False
                        }
            else:
                err_info = data.get("error", {})
                err_code = err_info.get("code") if isinstance(err_info, dict) else str(err_info)
                print(f"[-] Cobalt instance {inst_url} error ({status}): {err_code}")
        except Exception as e:
            print(f"[-] Cobalt instance {inst_url} exception: {e}")

    return None

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
    raw_keys = api_key or RAPIDAPI_KEY or ""
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
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
    Multi-tier resilient downloader with strict 720p - 1080p Quality Gate:
    1. Local file check.
    2. Tier 1: Zero-cost yt-dlp (Cloudflare WARP + android_vr / tv_embedded / bgutil POT).
    3. Tier 2: Cobalt Engine Ingestion ($0, no cookies, bypasses BotGuard & PoTokens).
    4. Tier 3: RapidAPI 1080p Muxer fallback (Key rotation).
    5. Tier 4: Apify Actor residential proxy fallback.
    """
    target_dir = output_dir or DOWNLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if local file
    local_path = Path(url_or_path)
    if local_path.exists() and local_path.is_file():
        h = get_video_height(local_path)
        dur = get_video_duration(local_path)
        return {
            'video_path': local_path.resolve(),
            'title': local_path.stem,
            'duration': dur,
            'transcript': None,
            'video_id': extract_youtube_id(local_path.stem),
            'height': h,
            'is_low_res': (0 < h < MIN_VIDEO_HEIGHT),
            'is_local': True
        }

    # Fetch native transcript if YouTube URL
    video_id = extract_youtube_id(url_or_path)
    transcript = None
    if video_id:
        transcript = fetch_youtube_transcript(video_id)

    fallback_candidate: Optional[Dict[str, Any]] = None

    # Tier 1: Direct yt-dlp ($0 cost, uses bgutil PO Token + android_vr + Proxy if set).
    try:
        ytdlp_result = download_via_ytdlp(url_or_path, target_dir, video_id=video_id, transcript=transcript)
        if ytdlp_result:
            h = ytdlp_result.get('height', 0)
            if h >= MIN_VIDEO_HEIGHT:
                print(f"[+] Tier 1 (yt-dlp) succeeded with verified HD quality ({h}p).")
                return ytdlp_result
            else:
                print(f"[!] Tier 1 result is {h}p (below {MIN_VIDEO_HEIGHT}p). Retaining as fallback and trying Tier 2 (Cobalt)...")
                fallback_candidate = ytdlp_result
    except VideoUnavailableError as e:
        print(f"[-] yt-dlp reports video permanently unavailable: {e}")
    except Exception as e:
        print(f"[-] Tier 1 (yt-dlp) failed: {e}")

    # Tier 2: Cobalt Engine API ($0, Zero Cookies, Bot-Bypass)
    print("[*] Engaging Tier 2: Cobalt Engine API ($0, bot-bypass)...")
    try:
        cobalt_result = download_via_cobalt(url_or_path, target_dir, quality="1080")
        if cobalt_result:
            cobalt_result['transcript'] = transcript
            cobalt_result['video_id'] = video_id
            h = cobalt_result.get('height', 0)
            if h >= MIN_VIDEO_HEIGHT:
                print(f"[+] Tier 2 (Cobalt) succeeded with verified HD quality ({h}p).")
                return cobalt_result
            elif not fallback_candidate or h > fallback_candidate.get('height', 0):
                fallback_candidate = cobalt_result
    except Exception as e:
        print(f"[-] Tier 2 (Cobalt) failed: {e}")

    # Tier 3: RapidAPI Downloader fallback (Option 2)
    if RAPIDAPI_KEY:
        print("[!] Engaging Tier 3: RapidAPI 1080p Downloader fallback...")
        try:
            rapidapi_result = download_via_rapidapi(url_or_path, target_dir)
            if rapidapi_result:
                rapidapi_result['transcript'] = transcript
                rapidapi_result['video_id'] = video_id
                h = rapidapi_result.get('height', 0)
                if h >= MIN_VIDEO_HEIGHT:
                    print(f"[+] Tier 3 (RapidAPI) succeeded with verified HD quality ({h}p).")
                    return rapidapi_result
                elif not fallback_candidate or h > fallback_candidate.get('height', 0):
                    fallback_candidate = rapidapi_result
        except Exception as e:
            print(f"[-] Tier 3 (RapidAPI) failed: {e}")

    # Tier 4: Apify Actor proxy fallback
    if APIFY_API_TOKEN and ("youtube.com" in url_or_path or "youtu.be" in url_or_path):
        print("[!] Engaging Tier 4: Apify Actor proxy downloader...")
        try:
            apify_result = download_via_apify(url_or_path, target_dir, quality="1080")
            if apify_result:
                apify_result['transcript'] = transcript
                apify_result['video_id'] = video_id
                return apify_result
        except Exception as e:
            print(f"[-] Tier 4 (Apify) failed: {e}")

    # If HD wasn't achieved but sub-HD candidate exists:
    if fallback_candidate:
        print(f"[!] Warning: All HD tiers exhausted. Using fallback candidate ({fallback_candidate.get('height', '?')}p).")
        return fallback_candidate

    raise RuntimeError(f"Failed to download video from {url_or_path} across all tiers (yt-dlp, Cobalt, RapidAPI, Apify).")


def fetch_transcript_only(url: str) -> Optional[Dict]:
    """
    Step 0 of the Smart Pipeline: Fetches ONLY the transcript (text) from a
    YouTube video — ZERO video bytes downloaded. Returns transcript + video_id.
    This is the trigger for Gemini to identify viral moments BEFORE any download.
    """
    video_id = extract_youtube_id(url)
    if not video_id:
        return None
    transcript = fetch_youtube_transcript(video_id)
    if not transcript:
        print("[!] No native transcript found. Targeted clip download unavailable; will fall back to full download.")
        return None
    return {"video_id": video_id, "transcript": transcript}


def download_clip_segment(
    video_url: str,
    start_time: float,
    end_time: float,
    clip_index: int,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Targeted Range Downloader: Downloads ONLY the seconds [start_time, end_time]
    of a YouTube video using yt-dlp's --download-sections flag.

    Instead of downloading a full 2GB podcast, this grabs ONLY the 15-25 MB
    clip window identified by Gemini — saving bandwidth, time, and avoiding
    YouTube's rate-limiter that triggers on long continuous streams.

    The downloaded clip starts at t=0s, so render_viral_clip must be called
    with start_time=0.0 and end_time=(end_time - start_time).

    Returns a dict compatible with download_video() results.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_duration = end_time - start_time
    out_file = output_dir / f"clip_{clip_index}_{int(start_time)}s_{int(end_time)}s.mp4"

    # yt-dlp --download-sections takes "*START-END" notation (seconds)
    section_spec = f"*{start_time:.2f}-{end_time:.2f}"
    print(f"[*] Targeted clip download: segment {section_spec} (~{clip_duration:.0f}s) → {out_file.name}")

    base_opts = {
        'js_runtimes': {'deno': {}, 'node': {}},
        'outtmpl': str(output_dir / f"clip_{clip_index}_{int(start_time)}s_{int(end_time)}s.%(ext)s"),
        'merge_output_format': 'mp4',
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
        'quiet': True,
        'no_warnings': True,
    }
    if YTDLP_PROXY and YTDLP_PROXY.strip():
        base_opts['proxy'] = YTDLP_PROXY.strip()

    # Try each client variant in priority order (android_vr first = no BotGuard)
    clients = _ytdlp_client_variants(cookie_path=None)
    for label, extra in clients:
        opts = {
            **base_opts,
            'format': _HD_FORMAT,
            'postprocessors': [{'key': 'FFmpegVideoConvertor', 'preferedformat': 'mp4'}],
            **extra,
        }
        try:
            print(f"  [*] Segment download via client '{label}'...")
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(video_url, download=True)

            # yt-dlp may produce the file with .mp4 directly or via post-processor
            produced = out_file
            if not produced.exists():
                # Search for any file matching our clip prefix
                candidates = sorted(output_dir.glob(f"clip_{clip_index}_{int(start_time)}s_{int(end_time)}s.*"))
                if candidates:
                    produced = candidates[0]

            if produced.exists() and produced.stat().st_size > 500_000:
                h = get_video_height(produced)
                dur = get_video_duration(produced)
                print(f"  [+] Segment downloaded ({h or '?'}p, {produced.stat().st_size / (1024*1024):.1f} MB) via '{label}'")
                return {
                    'video_path': produced.resolve(),
                    'title': f"clip_{clip_index}",
                    'duration': dur or clip_duration,
                    'height': h,
                    'is_low_res': (0 < h < MIN_VIDEO_HEIGHT),
                    'segment_start': 0.0,            # clip starts at t=0 in the file
                    'segment_duration': clip_duration,
                    'is_local': False,
                }
            else:
                print(f"  [-] Client '{label}' produced no usable file. Trying next...")
        except Exception as e:
            kind = classify_ytdlp_error(e)
            if kind == "permanent":
                print(f"  [-] Permanent error on segment download: {e}")
                return None
            print(f"  [-] Client '{label}' segment download failed ({kind}): {e}")

    print(f"[-] All yt-dlp clients failed for segment {section_spec}.")
    return None
