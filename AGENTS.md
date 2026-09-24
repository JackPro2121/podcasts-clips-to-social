# AGENTS.md — Autonomous AI Podcast Clipper & Social Publisher

Welcome agent / developer. This document establishes the operational rules, technical architecture, and execution procedures for the **Autonomous AI Podcast Viral Clipper & Buffer Social Media Publisher**.

---

## 🎯 Repository Purpose & Mission

This repository provides a **$0-budget, automated, broadcast-grade pipeline** that:
1. **Hyper-Focuses on Finance & Wealth**: Locks channel auto-discovery, Gemini prompts, and hashtags to top-earning personal finance, debt drama, and wealth creation podcasts (Caleb Hammer, Ramsey, Graham Stephan, Humphrey Yang, My First Million).
2. **Targeted Segment Extraction**: YouTube inputs with usable transcripts avoid full-video downloads and stream only the selected 30–60s window; other inputs may use a full-download fallback.
3. **Studio Sound Design & Sidechain Ducking**: Mixes vocal presence (+2.5dB at 3kHz), 80Hz rumble cut, dynamic voice sidechain ducking on BGM (`sidechaincompress`), millisecond-accurate SFX cues (`whoosh`, `pop`, `ding`), and EBU R128 (-14 LUFS / -1.5 dBTP) normalization.
4. **Kinetic Subtitles & Hook Caps**: Burns word-by-word spring-bounce subtitles (`\t(0,70,\fscx118\fscy118)`) with layout-specific safe-zone placement.
5. **AI Active Speaker Tracking**: Employs OpenCV YuNet on CPU to detect active speaker mouth motion, dynamically choosing between solo 9:16 portrait and dual-speaker split screen.
6. **Release Hosting & Buffer Posting**: Uploads clips to public GitHub Releases for the configured retention period and dispatches direct video URLs across connected social accounts via Buffer GraphQL API.
7. **Storage Hygiene**: Automatically deletes releases and assets older than 5 days.

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
│   ├── github_uploader.py           # GitHub Releases public asset uploader
│   ├── buffer_client.py             # Buffer GraphQL social media publishing client
│   ├── channel_discovery.py         # Apify scraper for high-CPM podcast discovery
│   └── release_cleaner.py           # Auto-deletes releases older than N days
├── tests/
│   └── test_pipeline.py             # Automated unit & integration tests
├── main.py                          # Unified CLI entry point
├── requirements.txt                 # Dependencies
├── .env.example                     # Environment template
├── README.md                        # Project overview
├── YOUTUBE_BYPASS_GUIDE.md          # YouTube datacenter IP bypass post-mortem & guide
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
3. **Audio Integrity & Sidechain Balancing**: Always master audio to EBU R128 (-14 LUFS, -1.5 dBTP) with 80Hz highpass rumble cut. Ensure background music is dynamically ducked when voice is present and maintain stereo stream formatting across all mixed inputs.
4. **Deterministic FFmpeg Stream Indexing**: Never compute `-filter_complex` stream indices with heuristic division. Maintain an integer counter that increments with each `-i` flag to strictly match FFmpeg's 0-indexed input stream references (`[0:a]`, `[1:a]`, `[2:a]`, ...).
5. **Targeted Segment Extraction**: Evaluate transcripts first for YouTube inputs and extract only the 30–60s clip window via Apify or range-limited yt-dlp. Use a documented full-download fallback for unsupported sources.
6. **Isolated Per-Clip Failure Recovery**: Always wrap clip rendering in isolated `try/except` handlers so a failure in a single clip does not abort the entire batch of clips.
7. **Temporary Direct Video URLs**: Buffer requires a publicly accessible video URL. GitHub Release assets must be in a public repository and are removed by the retention cleanup workflow.
8. **Storage Hygiene**: Always ensure the 5-day release cleanup routine is active to avoid accumulation of multi-gigabyte video files in GitHub Releases.
