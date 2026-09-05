# AGENTS.md — Autonomous AI Podcast Clipper & Social Publisher

Welcome agent / developer. This document establishes the operational rules, technical architecture, and execution procedures for the **Autonomous AI Podcast Viral Clipper & Buffer Social Media Publisher**.

---

## 🎯 Repository Purpose & Mission

This repository provides a **$0-budget, automated, broadcast-grade pipeline** that:
1. Ingests full-length YouTube podcasts via Apify Actor proxies (bypassing cloud datacenter bot-detection).
2. Transcribes dialogue with word-level precision (native transcript or `faster-whisper` CPU int8).
3. Evaluates high-retention viral moments (30–60s) using Google Gemini 1.5 Flash Free Tier.
4. Detects active speakers and frames them in 9:16 portrait (1080x1920) with smooth panning, dynamic host/guest split-screen stacking, or blurred-stack layouts.
5. Burns studio-grade animated karaoke subtitles (`hormozi`, `neon_green`, `luxury_gold`, `cyber_cyan`) placed strictly inside social media UI safe zones.
6. Masters audio to broadcast standards: 80Hz rumble cut, voice presence boost (3kHz), and EBU R128 (-14 LUFS / -1.5 dBTP) normalization.
7. Uploads clips permanently for $0 to GitHub Releases as downloadable MP4 assets.
8. Automatically schedules posts across TikTok, Instagram Reels, YouTube Shorts, and X/Twitter via Buffer GraphQL API.
9. Automatically auto-deletes releases and assets older than 5 days to prevent storage bloat.

---

## 🏗️ Architecture & Component Directory

```
PODCASTS-CLIPS-TO-SOCIAL/
├── .github/workflows/
│   ├── podcast_clipper.yml          # Primary workflow: Clip, enhance, release, post
│   └── cleanup_old_releases.yml     # Cron workflow: Purges releases older than 5 days
├── src/
│   ├── config.py                    # Safe zones, audio targets, subtitle color themes
│   ├── downloader.py                # Apify actor downloader & yt-dlp fallback
│   ├── transcriber.py               # Native YouTube captions & faster-whisper int8
│   ├── viral_detector.py            # Gemini 1.5 Flash hook finder & viral scoring
│   ├── face_tracker.py              # OpenCV face tracking & split-screen decisions
│   ├── subtitle_generator.py        # Word-by-word karaoke ASS subtitle engine
│   ├── video_editor.py              # FFmpeg filtergraph, loudness normalization
│   ├── github_uploader.py           # GitHub Releases permanent public asset uploader
│   ├── buffer_client.py             # Buffer GraphQL social media publishing client
│   ├── channel_discovery.py         # Apify scraper for high-CPM podcast discovery
│   └── release_cleaner.py           # Auto-deletes releases older than N days
├── tests/
│   └── test_pipeline.py             # Automated unit & integration tests
├── main.py                          # Unified CLI entry point
├── requirements.txt                 # Dependencies
├── .env.example                     # Environment template
├── README.md                        # Project overview
├── ARCHITECTURE_AND_METHODOLOGY.md  # Detailed technical specifications
├── HIGH_CPM_NICHES_GUIDE.md         # Playbook on high-earning niches
└── DEPLOYMENT_AND_OPERATIONS.md     # Production setup & run guide
```

---

## 🔑 Environment Variables & Secrets

| Variable | Scope | Description |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | GitHub Secret / `.env` | Free key from Google AI Studio for viral moment detection. |
| `APIFY_API_TOKEN` | GitHub Secret / `.env` | Apify key for proxy-based YouTube downloading and channel discovery. |
| `BUFFER_ACCESS_TOKEN` | GitHub Secret / `.env` | Buffer access token to queue clips to connected social accounts. |
| `BUFFER_CHANNEL_IDS` | Optional Secret | Comma-separated list of channel IDs (empty = all connected channels). |
| `GITHUB_TOKEN` | Automatic | Provided by GitHub Actions (`contents: write`) for Release creation. |
| `GITHUB_REPOSITORY` | Automatic | Target repository in `owner/repo` format. |

---

## 🛠️ CLI Commands & Usage

### 1. Run Complete Pipeline (Local or CLI)
```bash
python main.py --url "https://www.youtube.com/watch?v=..." --num-clips 3
```

### 2. Custom Framing & Subtitle Aesthetics
```bash
# Split-screen (Host top / Guest bottom) + Toxic Neon subtitles:
python main.py --url "..." --framing split --subtitle-style neon_green

# Luxury Gold subtitles + Skip burn-in if source already has subtitles:
python main.py --url "..." --subtitle-style luxury_gold --subtitles skip
```

### 3. Automatically Dispatch to Buffer
```bash
python main.py --url "..." --post-to-buffer
```

### 4. Discover High-CPM Podcast Channels
```bash
# Scrape top Finance podcasts:
python -m src.channel_discovery --niche finance

# Scrape Business & AI podcasts:
python -m src.channel_discovery --niche business
```

### 5. Manually Clean Old Releases (>5 Days)
```bash
python -m src.release_cleaner --days 5
```

### 6. Run Test Suite
```bash
python -m unittest discover -s tests
```

---

## 🛡️ Critical Engineering Rules for Agents

1. **Zero-Cost Constraint**: All components must remain within free tier allocations (Gemini Flash Free API, GitHub Actions runners, GitHub Releases asset hosting, Apify cheap pay-per-event).
2. **Safe-Zone Compliance**: Never place subtitles outside the safe zone (Y: 60–72% or center divider on split screen). UI buttons on TikTok/Reels will obscure anything in the lower 20% or right 15%.
3. **Audio Integrity**: Always master audio to EBU R128 (-14 LUFS, -1.5 dBTP) with 80Hz highpass rumble cut. Never push volume above -1.0 dBTP to avoid distortion upon social media re-compression.
4. **Permanent Direct Video URLs**: Buffer's API requires a publicly accessible video URL. Video assets uploaded to GitHub Releases provide permanent, high-bandwidth public direct URLs.
5. **Storage Hygiene**: Always ensure the 5-day release cleanup routine is active to avoid accumulation of multi-gigabyte video files in GitHub Releases.
