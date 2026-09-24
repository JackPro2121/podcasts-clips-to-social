# 🎙️ Autonomous AI Podcast Viral Clipper & Buffer Social Publisher ($0 Budget)

An automated, broadcast-grade pipeline that extracts high-retention viral moments from long-form podcasts, applies AI multi-speaker face tracking, dynamic 9:16 framing (TikTok/Reels/Shorts), studio audio mastering, and animated subtitles, hosts clips on public GitHub Releases for $0, and can publish/schedule them to social media via **Buffer**.

---

## 🌟 Key Capabilities

1. **$0 Budget & Zero-Download Architecture**:
   - **Targeted Segment Extraction**: Never downloads entire 2GB–4GB long podcasts. Transcripts are evaluated first, and only the 30–50s viral snippet (~15–25MB) is streamed using rotating residential proxies via Apify (`vidkraken/youtube-video-audio-downloader-reliable`), bypassing datacenter bot blocks (`Sign in to confirm you're not a bot`).
   - **AI Brain**: Google Gemini 1.5/2.5 Flash Free Tier for hook detection and viral scoring.
   - **Face Tracking & Layouts**: Runs on lightweight CPU OpenCV / YuNet, zero GPU server costs.
   - **Free Hosting for Buffer**: Uploads clips to public **GitHub Releases** for direct MP4 URLs during the configured retention period.
   - **Zero-Cost Cloud Runner**: Runs entirely on **GitHub Actions** (2,000 free minutes/month).

2. **Studio-Grade Video Editing & Audio Mastering**:
   - **Dynamic Audio Sidechain Ducking**: Voice automatically triggers downward compression on background music (`sidechaincompress`), ensuring crisp speech clarity with cinema-level polish.
   - **Precision SFX Sound Design**: Millisecond-accurate sound effects (`whoosh.wav`, `pop.wav`, `ding.wav`) synchronized with kinetic hooks and text pops.
   - **Broadcast Audio Standards**: 80Hz rumble highpass filter, +2.5dB vocal presence boost at 3kHz, and EBU R128 (-14 LUFS / -1.5 dBTP) normalization.

3. **Kinetic Bounce Subtitles**:
   - **Kinetic Pop Animation**: High-energy spring scaling (`\t(0,70,\fscx118\fscy118)`) for maximum retention.
   - **Safe-Zone Compliance**: Formatted strictly within mobile UI safe boundaries (avoiding buttons and handles).
   - Themes: `hormozi` (Yellow & White), `neon_green`, `luxury_gold`, and `cyber_cyan`.

4. **100% Hyper-Focused Niche: Personal Finance & Wealth**:
   - Locked entirely around top-earning US/UK finance content: Caleb Hammer (*Financial Audit*), The Ramsey Show, The Iced Coffee Hour, My First Million, The Money Guy Show, Humphrey Yang, and BiggerPockets.
   - Viral detection prompt specifically tuned for debt confessions, net worth reveals, and high-stakes financial drama.

5. **AI Active Speaker & Dynamic Framing**:
   - **Single Speaker**: Smooth camera tracking with exponential moving average (EMA).
   - **Active Speaker Mouth Tracking**: CPU-based mouth motion detection dynamically selects between solo framing and dual-speaker split screen.
   - **Two Speakers in Wide Shot**: Dynamic Split-Screen (Host top 1080x960, Guest bottom 1080x960) with divider bar.
   - **Group Panels**: Ambient Blurred-Stack layout (16:9 centered over blurred canvas).

---

## 🚀 Quick Start (Running via GitHub Actions - $0)

You do **not** need to install anything on your personal machine to run this!

### Step 1: Fork or Push this Repository to GitHub
Push this codebase to a public GitHub repository when using GitHub Release URLs with Buffer. Private repositories require authenticated asset access, which Buffer may not support.

### Step 2: Add Repository Secrets
Navigate to your GitHub repository:
**Settings** $\rightarrow$ **Secrets and variables** $\rightarrow$ **Actions** $\rightarrow$ **New repository secret**:

| Secret Name | How to get it (100% Free) |
| :--- | :--- |
| `GEMINI_API_KEY` | Grab your free API key at [Google AI Studio](https://aistudio.google.com/app/apikey). |
| `APIFY_API_TOKEN` | Required for targeted YouTube transcript/segment ingestion. |
| `BUFFER_ACCESS_TOKEN` | Required only when publishing to Buffer. |
| `BUFFER_CHANNEL_IDS` | Optional comma-separated target channel IDs. |

> Note: `GITHUB_TOKEN` is automatically provided by GitHub Actions with release write permissions.

### Step 3: Run the Workflow
1. Go to the **Actions** tab in your GitHub repository.
2. Click **Podcast Viral Clips to Social (Buffer)** in the left sidebar.
3. Click **Run workflow**:
   - **Video URL**: Paste any YouTube link, podcast URL, or direct MP4 link.
   - **Number of clips**: Choose `1` to `5`.
   - **Framing**: `auto` (smart face detection), `split`, `blur_stack`, or `crop`.
   - **Subtitle Style**: `hormozi`, `neon_green`, `luxury_gold`, or `cyber_cyan`.
   - **Post to Buffer**: Check to schedule to your social queue automatically.
4. Click **Run workflow**.

---

## 💻 Running Locally

### 1. Prerequisites
- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and in your system PATH.

### 2. Installation
```bash
git clone https://github.com/your-username/podcasts-clips-to-social.git
cd podcasts-clips-to-social
pip install -r requirements.txt
cp .env.example .env
```
For local runs, copy `.env.example` to `.env` and add only the integrations you use. Never commit `.env`; GitHub Actions credentials must be stored in repository Secrets.

### 3. Usage Examples

#### Extract 3 viral clips with auto face tracking and Hormozi subtitles:
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --num-clips 3
```

#### Force Split-Screen layout with Neon Green subtitles:
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --framing split --subtitle-style neon_green
```

#### Skip subtitle burn-in if source video already has baked-in captions:
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --subtitles skip
```

#### Auto-publish generated clips directly to Buffer:
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --post-to-buffer
```

#### Dry-run test (analyze viral moments and hooks without video rendering):
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --dry-run
```

---

## 📁 Project Architecture

```
PODCASTS-CLIPS-TO-SOCIAL/
├── .github/
│   └── workflows/
│       └── podcast_clipper.yml   # Complete GitHub Actions automation
├── src/
│   ├── config.py                 # Safe zones, audio targets, subtitle presets
│   ├── downloader.py             # yt-dlp & instant YouTube transcript extractor
│   ├── transcriber.py            # faster-whisper CPU fallback transcriber
│   ├── viral_detector.py         # Google Gemini Flash viral moment analyzer
│   ├── face_tracker.py           # OpenCV face tracking & split-screen decision engine
│   ├── subtitle_generator.py     # ASS karaoke subtitle generator
│   ├── video_editor.py           # FFmpeg 9:16 layout, audio mastering & burn-in
│   ├── github_uploader.py        # Permanent $0 release asset hosting for Buffer
│   └── buffer_client.py          # Buffer GraphQL API scheduler
├── tests/
│   └── test_pipeline.py          # Automated unit & integration tests
├── main.py                       # Unified CLI entrypoint
├── requirements.txt              # Minimal lightweight dependencies
├── .env.example                  # Environment credentials template
├── YOUTUBE_BYPASS_GUIDE.md       # YouTube IP Block Bypass Guide & Post-Mortem
└── README.md                     # Documentation
```

---

## 📚 Complete Guides & Specifications

* [YOUTUBE_BYPASS_GUIDE.md](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/YOUTUBE_BYPASS_GUIDE.md) — Comprehensive guide on how YouTube datacenter IP blocking works, what failed (Tor, WARP, Railway, etc.), and what works ($0).
* [ARCHITECTURE_AND_METHODOLOGY.md](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/ARCHITECTURE_AND_METHODOLOGY.md) — Deep architectural deep dive on face tracking, ASS subtitles, and FFmpeg filtergraphs.
* [HIGH_CPM_NICHES_GUIDE.md](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/HIGH_CPM_NICHES_GUIDE.md) — Playbook on Personal Finance & Wealth viral content.
* [DEPLOYMENT_AND_OPERATIONS.md](file:///d:/Workspace/PODCASTS-CLIPS-TO-SOCIAL/DEPLOYMENT_AND_OPERATIONS.md) — Setup and deployment manual for GitHub Actions and Buffer.

