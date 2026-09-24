# Deployment, Automation & Operations Guide

A step-by-step operations runbook for configuring, deploying, running, and maintaining the **Autonomous AI Podcast Viral Clipper & Buffer Social Media Publisher** on GitHub Actions for **$0**.

---

## 1. Initial Setup: Adding Repository Secrets

Navigate to your GitHub repository:
**Settings** $\rightarrow$ **Secrets and variables** $\rightarrow$ **Actions** $\rightarrow$ **New repository secret**:

| Secret Name | Required? | How to Obtain for $0 |
| :--- | :--- | :--- |
| `APIFY_API_TOKEN` | **Yes for targeted YouTube ingestion** | Apify account integrations. |
| `GEMINI_API_KEY` | **Yes for AI clip detection** | Google AI Studio API key. |
| `BUFFER_ACCESS_TOKEN` | Optional* | Buffer developer/settings access token. Required for publishing. |
| `BUFFER_CHANNEL_IDS` | Optional | Comma-separated Buffer channel IDs. Leave blank to auto-discover. |
| `CHOCODATA_API_KEY` | Optional | Chocodata transcript fallback. |
| `GROQ_API_KEY` | Optional | Groq LLM fallback. |
| `OPENROUTER_API_KEY` | Optional | OpenRouter LLM fallback. |
| `OLLAMA_API_KEY` | Optional | Ollama Cloud fallback. |
| `PEXELS_API_KEY` | Optional | Pexels B-roll; disabled by default. |
| `RAPIDAPI_KEY` | Optional | RapidAPI downloader fallback. |
| `SLACK_WEBHOOK_URL` | Optional | Slack run/error notifications. |
| `YOUTUBE_COOKIES` | Optional | Store only if a fresh Netscape cookie file is explicitly required. |

> **Note**: `GITHUB_TOKEN` is automatically provisioned by GitHub Actions with `contents: write` permissions.

---

## 2. Triggering Workflows via GitHub Actions

### Workflow 1: Clip & Publish New Podcast Episode
1. Go to the **Actions** tab in your GitHub repository.
2. Select **Podcast Viral Clips to Social (Buffer)** in the left sidebar.
3. Click **Run workflow**:
   - **Video URL**: Paste any YouTube podcast URL, podcast RSS URL, or direct MP4 link.
   - **Number of clips**: Select `1`, `2`, `3`, `4`, or `5`.
   - **Framing**:
     - `auto` (Default): Evaluates faces with OpenCV. Uses centered single-speaker panning, host/guest split-screen stacking, or blurred-stack layouts.
     - `split`: Forces dual-speaker split screen (top host / bottom guest).
     - `blur_stack`: Centers 16:9 over blurred background (safe for multi-person panel shows).
     - `crop`: Centers a 9:16 crop directly.
   - **Subtitle Style**:
     - `hormozi` (Default): Electric Yellow highlight + White text + 4px black outline.
     - `neon_green`: Toxic Neon Green highlight + White text.
     - `luxury_gold`: Warm Gold accent + Ivory text.
     - `cyber_cyan`: Electric Cyan accent + Pure White text.
   - **Subtitles Mode**: `auto` (burn captions), `skip` (if video already has captions baked-in).
   - **Niche**: `finance` (Default, locked for maximum CPM/RPM), or specify another category.
   - **Post to Buffer**: Check `true` to auto-schedule across your social channels.
> **Targeted segment path:** YouTube inputs with usable transcripts download only the selected clip range. Other source types or transcript failures may use the full-download fallback.
4. Click **Run workflow**.

---

## 3. Storage Maintenance: 5-Day Auto-Cleanup

To prevent video files from hoarding space on GitHub Releases:

1. **Automatic Inline Cleanup**:
   - Every time `main.py` finishes, it automatically scans releases tagged `clips-*` and deletes matching releases/assets older than **5 days**, while preserving active Buffer posts.
2. **Automated Daily Cron Workflow (`cleanup_old_releases.yml`)**:
   - Runs automatically every single day at **03:00 UTC**.
   - Identifies any video releases older than 5 days and deletes them via GitHub REST API.
3. **Manual Run on Demand**:
   - Go to **Actions** $\rightarrow$ select **Auto-Cleanup Old Video Releases** $\rightarrow$ choose days threshold (e.g. `5`) $\rightarrow$ Click **Run workflow**.

---

## 4. Discovering High-CPM Podcast Channels

Use the built-in Apify scraper tool to locate high-earning podcast episodes:

```bash
# Discover Finance podcasts:
python -m src.channel_discovery --niche finance

# Discover Business & Entrepreneurship podcasts:
python -m src.channel_discovery --niche business

# Discover AI & Tech podcasts:
python -m src.channel_discovery --niche ai_tech

# Discover a category and print candidates:
python -m src.channel_discovery --niche finance
```

---

## 5. Buffer Social Channel Setup

1. Create a free account at [Buffer](https://buffer.com).
2. Connect your social channels:
   - TikTok Profile
   - Instagram Professional / Creator Account
   - YouTube Channel (for YouTube Shorts)
   - Facebook Page (for Facebook Reels)
   - X / Twitter Account
3. Get your **Buffer Access Token**:
   - Open Developer Settings or Account Settings $\rightarrow$ Access Tokens.
   - Add it to GitHub Secrets as `BUFFER_ACCESS_TOKEN`.
4. When `post_to_buffer` is enabled, the pipeline queries your connected channels and automatically adds each viral video clip, hook title, caption, and hashtags into your posting queue!

---

## 6. Local Workstation Development

### Setup:
```bash
git clone https://github.com/JackPro2121/podcasts-clips-to-social.git
cd podcasts-clips-to-social
pip install -r requirements.txt
cp .env.example .env
```
For local runs, copy `.env.example` to `.env` and provide only the integrations you use. Never commit `.env`; GitHub Actions must receive credentials through repository Secrets.

### Dry-Run Test (Zero Video Rendering):
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --dry-run
```

### Full Local Test Run:
```bash
python main.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --num-clips 1
```

### Run Unit Tests:
```bash
python -m unittest discover -s tests
```

---

## 7. Troubleshooting & FAQ

### Issue: "Sign in to confirm you're not a bot"
- **Cause**: YouTube blocking cloud runner datacenter IPs.
- **Solution**: The pipeline automatically uses your `APIFY_API_TOKEN` to route downloads through Apify's residential proxies, completely bypassing this block.

### Issue: "Buffer rejected video upload"
- **Cause**: Video URL was unreachable or private.
- **Solution**: The pipeline uploads clips to a public GitHub Release and passes the public asset URL to Buffer. Releases are temporary and are cleaned after the configured retention period.

### Issue: "Subtitles overlapping creator handle on TikTok"
- **Cause**: Subtitles placed too low on the screen.
- **Solution**: The subtitle engine applies layout-specific safe-zone margins and keeps captions above mobile platform controls.
