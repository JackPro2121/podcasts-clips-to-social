# 🍪 YouTube Cookie Exporter for yt-dlp & Clipper

A lightweight, 100% private Manifest V3 Chrome extension designed to export your YouTube session cookies in standard Netscape format for **yt-dlp**, **curl**, and **GitHub Actions**.

---

## ⚡ Why Use This?
GitHub Actions cloud runners operate on Microsoft Azure datacenter IP addresses. YouTube frequently blocks datacenter IPs with bot verification (`Sign in to confirm you're not a bot`).

When you export your YouTube cookies and save them as the `YOUTUBE_COOKIES` secret in your GitHub repository:
- YouTube treats the GitHub Actions runner as a **real logged-in human user**.
- All video downloads run at maximum speed with **$0 compute/proxy cost**.
- Zero Apify credits are needed for video downloads!

---

## 🚀 How to Install in Google Chrome (Takes 30 Seconds)

1. Open Google Chrome and navigate to:
   ```
   chrome://extensions
   ```
2. Enable **Developer mode** toggle in the top-right corner.
3. Click the **"Load unpacked"** button in the top-left corner.
4. Select the extension directory:
   ```
   d:\Workspace\PODCASTS-CLIPS-TO-SOCIAL\tools\youtube-cookie-exporter
   ```
5. Pin the **YouTube Cookie Exporter** icon to your Chrome toolbar.

---

## 📋 How to Export Cookies to GitHub Secrets

1. Make sure you are logged into YouTube in your browser ([youtube.com](https://www.youtube.com)).
2. Click the **YouTube Cookie Exporter** extension icon in your Chrome toolbar.
3. It will display:
   - `YouTube Session: ✅ Active & Logged In`
   - `Cookies Found: XX`
4. Click the purple button: **"Copy Cookies to Clipboard"**.
5. Go to your GitHub repository:
   - [JackPro2121/podcasts-clips-to-social Secrets](https://github.com/JackPro2121/podcasts-clips-to-social/settings/secrets/actions)
   - Click **"New repository secret"**.
   - **Name:** `YOUTUBE_COOKIES`
   - **Secret:** *(Paste the copied text from your clipboard)*
   - Click **"Add secret"**.

Done! Your autonomous pipeline will now automatically use your YouTube cookies on every scheduled run.
