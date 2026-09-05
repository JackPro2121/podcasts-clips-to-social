# Deployment, Automation & Operations Guide

A step-by-step operations runbook for configuring, deploying, running, and maintaining the **Autonomous AI Podcast Viral Clipper & Buffer Social Media Publisher** on GitHub Actions for **$0**.

---

## 1. Initial Setup: Adding Repository Secrets

Navigate to your GitHub repository:
**Settings** $\rightarrow$ **Secrets and variables** $\rightarrow$ **Actions** $\rightarrow$ **New repository secret**:

| Secret Name | Required? | How to Obtain for $0 |
| :--- | :--- | :--- |
| `APIFY_API_TOKEN` | **Yes** | Sign up at [Apify](https://console.apify.com) $\rightarrow$ Settings $\rightarrow$ Integrations $\rightarrow$ Copy API Token. Bypasses datacenter IP blocks. |
| `GEMINI_API_KEY` | **Yes** | Sign up at [Google AI Studio](https://aistudio.google.com/app/apikey) $\rightarrow$ Create API Key (Free tier: 15 RPM / 1M tokens/min). |
| `BUFFER_ACCESS_TOKEN` | Optional* | From [Buffer Developer Portal](https://developers.buffer.com) or Settings. *(Required only if posting to socials)*. |
| `BUFFER_CHANNEL_IDS` | Optional | Comma-separated list of target channel IDs. Leave blank to auto-post to all connected channels. |

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
   - **Post to Buffer**: Check `true` to auto-schedule across your social channels.
4. Click **Run workflow**.

---

## 3. Storage Maintenance: 5-Day Auto-Cleanup

To prevent video files from hoarding space on GitHub Releases:

1. **Automatic Inline Cleanup**:
   - Every time `main.py` finishes, it automatically scans your releases and deletes any release and asset older than **5 days**.
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

# Scrape all 5 categories to JSON:
python -m src.channel_discovery --niche all --output-json top_podcasts.json
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
Fill in your `GEMINI_API_KEY`, `APIFY_API_TOKEN`, and `BUFFER_ACCESS_TOKEN` in `.env`.

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
- **Solution**: The pipeline uploads clips to **GitHub Releases**, creating permanent public direct download links (`https://github.com/.../releases/download/.../clip.mp4`) that Buffer can fetch instantly without authentication.

### Issue: "Subtitles overlapping creator handle on TikTok"
- **Cause**: Subtitles placed too low on the screen.
- **Solution**: Our subtitle engine strictly enforces safe-zone margins (`MarginV = 420` out of 1920), keeping captions in the sweet-spot center-lower third well above all mobile platform interface buttons.
