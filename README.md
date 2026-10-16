# Autonomous AI Viral Video Clipper & Social Publisher

[![Podcast Viral Clips (V1)](https://github.com/JackPro2121/podcasts-clips-to-social/actions/workflows/podcast_clipper.yml/badge.svg)](https://github.com/JackPro2121/podcasts-clips-to-social/actions/workflows/podcast_clipper.yml)
[![Streamer Viral Clips (V2)](https://github.com/JackPro2121/streamer-clips-to-social/actions/workflows/streamer_clipper.yml/badge.svg)](https://github.com/JackPro2121/streamer-clips-to-social/actions/workflows/streamer_clipper.yml)

An enterprise-grade, **$0-budget autonomous AI media engine** that converts long-form podcasts and interviews into viral 9:16 vertical videos for **TikTok, YouTube Shorts, and Instagram Reels**.

This repository is the **V1 Podcast pipeline**. The high-energy **V2 Streamer pipeline** (Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc) now lives in its own repository — the two were split so each can ship without risking the other.

---

## 🚀 Two Repositories, One Upstream Engine

| | This repo | Streamer repo |
| :--- | :--- | :--- |
| Pipeline | **V1 — Podcasts** | **V2 — Streamers** |
| Repository | [`JackPro2121/podcasts-clips-to-social`](https://github.com/JackPro2121/podcasts-clips-to-social) | [`JackPro2121/streamer-clips-to-social`](https://github.com/JackPro2121/streamer-clips-to-social) |
| Entrypoint | `python main.py` | `python main_streamer.py` |
| Workflow | `podcast_clipper.yml` | `streamer_clipper.yml` |
| Code | `core/` + `v1/` | `core/` + `streamers/` |
| Publishing token | `BUFFER_ACCESS_TOKEN` | `STREAMER_BUFFER_ACCESS_TOKEN` |

This repo mounts the shared engine ([`JackPro2121/clips-core`](https://github.com/JackPro2121/clips-core))
as a git submodule at `core/`. The streamer repo builds on the same upstream but keeps its **own
vendored copy** of it, so it ships independently — an engine fix is landed upstream once and then
pulled into both. Nothing else is shared: separate entrypoints, separate workflows, separate Buffer
credentials. A change in one pipeline is never a change in the other.

---

## 🌟 Feature Breakdown

### 🎙️ Pipeline V1: Podcast Viral Clipper
* **Hyper-Focused Niche**: Tuned for high-CPM personal finance, debt confessions, net worth reveals, and entrepreneurship.
* **Dual-Speaker Split Screen**: Active mouth motion detection dynamically stacks host and guest vertically (1080x960 each) with zero black side bars.
* **Studio Audio Mastering**: EBU R128 (-14.0 LUFS / -1.5 dBTP), 80Hz rumble cut, +2.5dB vocal presence boost at 3kHz, and dynamic sidechain compression on background music.
* **Kinetic Subtitles**: Word-by-word spring-bounce animation (`hormozi`, `neon_green`, `luxury_gold`).

### ⚡ Pipeline V2: Streamer Viral Clipper → [moved to its own repository](https://github.com/JackPro2121/streamer-clips-to-social)
* Kai Cenat, IShowSpeed, Jynxzi, CaseOh, xQc, Adin Ross.
* SFace ArcFace identity locking, 3-tier off-screen state machine, auto-stacked webcam/gameplay layout, scream-triggered punch-zooms, TikTok meme hook box, and strict 2–4 hashtags.
* Full documentation: [`streamer-clips-to-social`](https://github.com/JackPro2121/streamer-clips-to-social).

---

## 🛠️ Quick Start (Running via GitHub Actions - $0)

No local setup or expensive GPU servers required! Runs entirely in GitHub Actions.

### 1. Add Repository Secrets
In your GitHub repo: **Settings** → **Secrets and variables** → **Actions**:

| Secret Name | Required By | Description |
| :--- | :---: | :--- |
| `GEMINI_API_KEY` | V1 | Free API key from [Google AI Studio](https://aistudio.google.com/app/apikey). |
| `APIFY_API_TOKEN` | Optional | Residential proxy fallback for YouTube datacenter blocks. |
| `BUFFER_ACCESS_TOKEN` | V1 | Buffer token for Podcast accounts. |
| `BUFFER_CHANNEL_IDS` | Optional | Specific podcast social channels in Buffer. |

The streamer-only secrets (`STREAMER_BUFFER_ACCESS_TOKEN`, `STREAMER_BUFFER_CHANNEL_IDS`, `FIRECRAWL_API_KEY`, `STREAMER_WATERMARK`) belong to the [streamer repo](https://github.com/JackPro2121/streamer-clips-to-social).

The full V1 reference — LLM fallback keys, download-resilience keys (Cobalt, RapidAPI, Chocodata), Slack run alerts, and optional repository variables — lives in [`AGENTS.md`](AGENTS.md).

### 2. Trigger Workflow
1. Navigate to the **Actions** tab in GitHub.
2. Select **Podcast Viral Clips to Social (Buffer)**.
3. Click **Run workflow**, paste the video URL (or leave blank to auto-discover), and choose the number of clips (`1` to `5`).

---

## 💻 Local Installation & Usage

### Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available in system PATH.
- A Bash shell (Linux/macOS, or Git Bash / WSL on Windows) for the shared installer.

```bash
git clone https://github.com/JackPro2121/podcasts-clips-to-social.git
cd podcasts-clips-to-social
git submodule update --init --recursive   # pulls the V1 core/ engine (submodule)

# Deterministic dependency install — do NOT `pip install -r requirements.txt`
# directly; the shared installer resolves one canonical OpenCV build:
bash core/tools/install_deps.sh           # runtime only
bash core/tools/install_deps.sh --dev     # + dev/test tooling (pytest, ruff, mypy)
```

### Running Pipeline V1 (Podcasts)
```bash
# Extract 3 viral clips with Hormozi subtitles:
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --num-clips 3

# Force Dual-Speaker Split Screen:
python main.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --framing split
```

### Running Pipeline V2 (Streamers)
The streamer pipeline is no longer in this repository:

```bash
git clone https://github.com/JackPro2121/streamer-clips-to-social.git
cd streamer-clips-to-social
bash core/tools/install_deps.sh           # deterministic install (vendored engine installer)
python main_streamer.py --streamer kai_cenat --num-clips 3
```

---

## 📖 Operational Documentation

* [`AGENTS.md`](AGENTS.md) — Canonical rules, architecture, and guidelines for AI agents and developers.
